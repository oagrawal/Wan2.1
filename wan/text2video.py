# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                               get_sampling_sigmas, retrieve_timesteps)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


which_gpu = 0


def test_batchability(model, x, t, context, seq_len, clip_fea=None, y=None, batch_size=2):
    print("=== ORIGINAL DATA ===")
    print(f"Original x: {len(x)} videos, shapes: {[u.shape for u in x]}")
    print(f"Original t: {t.shape}, value: {t}")
    print(f"Original context: {len(context)} texts, shapes: {[u.shape for u in context]}")
    
    # Create batch test data
    x_batch, t_batch, context_batch = create_batch_test_data(x, t, context, batch_size)
    
    print("\n=== BATCH TEST DATA ===")
    print(f"Batch x: {len(x_batch)} videos, shapes: {[u.shape for u in x_batch]}")
    print(f"Batch t: {t_batch.shape}, values: {t_batch}")
    print(f"Batch context: {len(context_batch)} texts, shapes: {[u.shape for u in context_batch]}")

    # Timing events
    start_original = torch.cuda.Event(enable_timing=True)
    end_original = torch.cuda.Event(enable_timing=True)
    start_batch = torch.cuda.Event(enable_timing=True)
    end_batch = torch.cuda.Event(enable_timing=True)
    
    # Test original vs batch processing
    print("\n=== RUNNING ORIGINAL MODEL ===")
    start_original.record()
    output_original = model.forward(x, t, context, seq_len, clip_fea, y)
    end_original.record()
    
    print("\n=== RUNNING BATCH MODEL ===")
    start_batch.record()
    output_batch = model.forward(x_batch, t_batch, context_batch, seq_len, clip_fea, y)
    end_batch.record()
    
    torch.cuda.synchronize()
    original_time = start_original.elapsed_time(end_original)
    batch_time = start_batch.elapsed_time(end_batch)
    
    print("\n=== TIMING RESULTS ===")
    print(f"Original processing time: {original_time:.2f}ms")
    print(f"Batch processing time: {batch_time:.2f}ms")
    print(f"Speed ratio: {original_time*batch_size/batch_time:.2f}x")

    print("\n=== OUTPUT COMPARISON ===")
    print(f"Original output: {len(output_original)} tensors")
    for i, out in enumerate(output_original):
        print(f"  output[{i}] shape: {out.shape}")
    
    print(f"Batch output: {len(output_batch)} tensors")
    for i, out in enumerate(output_batch):
        print(f"  output[{i}] shape: {out.shape}")
    
    # Verify consistency
    print("\n=== CONSISTENCY CHECK ===")
    expected_batch_count = len(output_original) * batch_size
    actual_batch_count = len(output_batch)
    
    if actual_batch_count != expected_batch_count:
        print(f"✗ Batch size mismatch: Expected {expected_batch_count}, got {actual_batch_count}")
        return output_original, output_batch

    print("✓ Output length scales correctly with batch size")
    
    max_diffs = []
    mean_diffs = []
    matches = 0
    
    for batch_idx in range(batch_size):
        original_idx = batch_idx % len(output_original)
        batch_tensor = output_batch[batch_idx]
        original_tensor = output_original[original_idx]
        
        print(f"\n--- Comparing Batch[{batch_idx}] vs Original[{original_idx}] ---")
        
        if batch_tensor.shape != original_tensor.shape:
            print(f"✗ Shape mismatch at batch index {batch_idx}")
            continue
        
        # Basic tensor info
        orig_flat = original_tensor.flatten()
        batch_flat = batch_tensor.flatten()
        print(f"Tensor shapes: {original_tensor.shape}")
        print(f"Original - First 10 values: {orig_flat[:10].tolist()}")
        print(f"Batch[{batch_idx}] - First 10 values: {batch_flat[:10].tolist()}")
        
        # Statistics comparison
        print(f"Original - Min: {original_tensor.min().item():.6f}, Max: {original_tensor.max().item():.6f}, Mean: {original_tensor.mean().item():.6f}")
        print(f"Batch[{batch_idx}] - Min: {batch_tensor.min().item():.6f}, Max: {batch_tensor.max().item():.6f}, Mean: {batch_tensor.mean().item():.6f}")
        
        # Difference analysis
        diff = torch.abs(batch_tensor - original_tensor)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        max_diffs.append(max_diff)
        mean_diffs.append(mean_diff)
        
        # Relative difference context
        tensor_range = original_tensor.max().item() - original_tensor.min().item()
        relative_max_diff = (max_diff / tensor_range * 100) if tensor_range > 0 else 0
        relative_mean_diff = (mean_diff / abs(original_tensor.mean().item()) * 100) if original_tensor.mean().item() != 0 else 0
        
        print(f"Max difference: {max_diff:.8f} ({relative_max_diff:.4f}% of tensor range)")
        print(f"Mean difference: {mean_diff:.8f} ({relative_mean_diff:.4f}% of mean value)")
        
        # Tolerance check
        is_close = torch.allclose(batch_tensor, original_tensor, atol=1e-3, rtol=1e-3)
        print(f"Within tolerance (1e-3): {'✓' if is_close else '✗'}")
        
        if is_close:
            matches += 1

    print(f"\n=== CONSISTENCY SUMMARY ===")
    print(f"Batch elements matching original: {matches}/{batch_size}")
    if max_diffs:
        print(f"Max differences: {max(max_diffs):.6f} (avg {sum(max_diffs)/len(max_diffs):.6f})")
        print(f"Mean differences: {max(mean_diffs):.6f} (avg {sum(mean_diffs)/len(mean_diffs):.6f})")
    
    return output_original, output_batch

def create_batch_test_data(x, t, context, batch_size=2):
    """
    Create batch test data based on observed shapes:
    - x: list with 1 tensor [16, 21, 60, 104] 
    - t: tensor [1]
    - context: list with 1 tensor [20/126, 4096]
    """
    # For x: duplicate the tensor within the list to simulate multiple videos
    x_batch = x * batch_size  # Creates [tensor1, tensor1] for batch_size=2
    
    # For t: create proper batch dimension 
    t_batch = t.repeat(batch_size)  # [1] -> [1, 1] for batch_size=2
    
    # For context: duplicate the tensor within the list
    context_batch = context * batch_size  # Creates [tensor1, tensor1] for batch_size=2
    
    return x_batch, t_batch, context_batch



class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import \
                get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (usp_attn_forward,
                                                            usp_dit_forward)
            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn)
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt
        


    def generate(self,
             input_prompt,
             size=(1280, 720),
             frame_num=81,
             shift=5.0,
             sample_solver='unipc',
             sampling_steps=50,
             guide_scale=5.0,
             n_prompt="",
             seed=-1,
             offload_model=True):   

        # Initialize timing dictionary
        timings = {
            "t5_encoder": 0.0,
            "wan_model_conditional": 0.0,
            "wan_model_unconditional": 0.0,
            "vae_decode": 0.0
        }
        
        # Original preprocessing code...
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])
        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        # Time T5 encoder
        start_t5 = torch.cuda.Event(enable_timing=True)
        end_t5 = torch.cuda.Event(enable_timing=True)
        
        start_t5.record()
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            if isinstance(input_prompt, str):
                context = self.text_encoder([input_prompt], self.device)
                context_null = self.text_encoder([n_prompt], self.device)
            else:
                context = self.text_encoder(input_prompt, self.device)
                context_null = self.text_encoder(n_prompt, self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]
        end_t5.record()
        torch.cuda.synchronize()
        timings["t5_encoder"] = start_t5.elapsed_time(end_t5)

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
            # Original scheduler setup...
            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            # Time WanModel in diffusion loop
            for i, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]
                timestep = torch.stack(timestep)

                self.model.to(self.device)
                
                # test_batchability(self.model, latent_model_input, timestep, arg_c['context'], arg_c['seq_len'], batch_size=2)

                # Time conditional model call
                start_cond = torch.cuda.Event(enable_timing=True)
                end_cond = torch.cuda.Event(enable_timing=True)
                
                start_cond.record()
                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                end_cond.record()
                torch.cuda.synchronize()
                timings["wan_model_conditional"] += start_cond.elapsed_time(end_cond)
                
                # Time unconditional model call
                start_uncond = torch.cuda.Event(enable_timing=True)
                end_uncond = torch.cuda.Event(enable_timing=True)
                
                start_uncond.record()
                noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
                end_uncond.record()
                torch.cuda.synchronize()
                timings["wan_model_unconditional"] += start_uncond.elapsed_time(end_uncond)

                # Rest of the step logic
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.empty_cache()
            
            # Time VAE decoding
            if self.rank == which_gpu:
                start_vae = torch.cuda.Event(enable_timing=True)
                end_vae = torch.cuda.Event(enable_timing=True)
                
                start_vae.record()
                videos = self.vae.decode(x0)
                end_vae.record()
                torch.cuda.synchronize()
                timings["vae_decode"] = start_vae.elapsed_time(end_vae)

        # Original cleanup code
        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        # Print timing summary
        if self.rank == which_gpu:
            print("\n===== Model Timing Summary =====")
            print(f"T5 Text Encoder: {timings['t5_encoder']:.2f} ms")
            print(f"WanModel (Conditional): {timings['wan_model_conditional']:.2f} ms")
            print(f"WanModel (Unconditional): {timings['wan_model_unconditional']:.2f} ms")
            total_model_time = timings['wan_model_conditional'] + timings['wan_model_unconditional']
            print(f"WanModel Total: {total_model_time:.2f} ms")
            print(f"Average time per diffusion step: {total_model_time/len(timesteps):.2f} ms")
            print(f"VAE Decoding: {timings['vae_decode']:.2f} ms")
            print(f"Total model time: {timings['t5_encoder'] + total_model_time + timings['vae_decode']:.2f} ms")
            print("==============================\n")

        return videos[0] if self.rank == which_gpu else None
