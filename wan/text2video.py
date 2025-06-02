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
    print(f"Original t: {t.shape}")
    print(f"Original context: {len(context)} texts, shapes: {[u.shape for u in context]}")
    
    # Create batch test data
    x_batch, t_batch, context_batch = create_batch_test_data(x, t, context, batch_size)
    
    print("\n=== BATCH TEST DATA ===")
    print(f"Batch x: {len(x_batch)} videos, shapes: {[u.shape for u in x_batch]}")
    print(f"Batch t: {t_batch.shape}, values: {t_batch}")
    print(f"Batch context: {len(context_batch)} texts, shapes: {[u.shape for u in context_batch]}")
    
    # Test original vs batch processing
    print("\n=== RUNNING ORIGINAL MODEL ===")
    output_original = model.forward(x, t, context, seq_len, clip_fea, y)
    
    print("\n=== RUNNING BATCH MODEL ===")
    output_batch = model.forward(x_batch, t_batch, context_batch, seq_len, clip_fea, y)
    
    print("\n=== OUTPUT COMPARISON ===")
    print(f"Original output: {len(output_original)} tensors")
    for i, out in enumerate(output_original):
        print(f"  output[{i}] shape: {out.shape}")
    
    print(f"Batch output: {len(output_batch)} tensors")
    for i, out in enumerate(output_batch):
        print(f"  output[{i}] shape: {out.shape}")
    
    # Verify consistency (check if batch output contains duplicated original output)
    print("\n=== CONSISTENCY CHECK ===")
    if len(output_batch) == len(output_original) * batch_size:
        print("✓ Output length scales correctly with batch size")
        
        # Check if outputs are duplicated correctly
        original_tensor = output_original[0]
        batch_tensor_1 = output_batch[0]
        batch_tensor_2 = output_batch[1]
            
        if batch_tensor_1.shape == original_tensor.shape and batch_tensor_2.shape == original_tensor.shape:
            print(f"✓ Output[{i}] shapes match")
            
            # Check numerical similarity (within floating point tolerance)
            # Check numerical similarity (within floating point tolerance)
            if torch.allclose(batch_tensor_1, original_tensor, atol=1e-5) and torch.allclose(batch_tensor_2, original_tensor, atol=1e-5):
                print(f"✓ Output values match (duplicated correctly)")
            else:
                print(f"✗ Output values don't match - model may not be deterministic or has batching issues")
                
                # Print tensors here
                print("\n=== TENSOR VALUE COMPARISON ===")
                
                # Original tensor info
                orig_flat = original_tensor.flatten()
                print(f"Original tensor:")
                print(f"  Shape: {original_tensor.shape}")
                print(f"  Dtype: {original_tensor.dtype}")
                print(f"  First 10 values: {orig_flat[:10].tolist()}")
                print(f"  Min: {original_tensor.min().item():.6f}, Max: {original_tensor.max().item():.6f}")
                print(f"  Mean: {original_tensor.mean().item():.6f}, Std: {original_tensor.std().item():.6f}")
                
                # Batch tensor 1 info
                batch1_flat = batch_tensor_1.flatten()
                print(f"\nBatch tensor 1:")
                print(f"  Shape: {batch_tensor_1.shape}")
                print(f"  Dtype: {batch_tensor_1.dtype}")
                print(f"  First 10 values: {batch1_flat[:10].tolist()}")
                print(f"  Min: {batch_tensor_1.min().item():.6f}, Max: {batch_tensor_1.max().item():.6f}")
                print(f"  Mean: {batch_tensor_1.mean().item():.6f}, Std: {batch_tensor_1.std().item():.6f}")
                
                # Batch tensor 2 info
                batch2_flat = batch_tensor_2.flatten()
                print(f"\nBatch tensor 2:")
                print(f"  Shape: {batch_tensor_2.shape}")
                print(f"  Dtype: {batch_tensor_2.dtype}")
                print(f"  First 10 values: {batch2_flat[:10].tolist()}")
                print(f"  Min: {batch_tensor_2.min().item():.6f}, Max: {batch_tensor_2.max().item():.6f}")
                print(f"  Mean: {batch_tensor_2.mean().item():.6f}, Std: {batch_tensor_2.std().item():.6f}")
                
                # Difference analysis
                diff_1_orig = torch.abs(batch_tensor_1 - original_tensor)
                diff_2_orig = torch.abs(batch_tensor_2 - original_tensor)
                diff_1_2 = torch.abs(batch_tensor_1 - batch_tensor_2)
                
                print(f"\n=== DIFFERENCE ANALYSIS ===")
                print(f"Max difference (batch1 vs original): {diff_1_orig.max().item():.8f}")
                print(f"Mean difference (batch1 vs original): {diff_1_orig.mean().item():.8f}")
                print(f"Max difference (batch2 vs original): {diff_2_orig.max().item():.8f}")
                print(f"Mean difference (batch2 vs original): {diff_2_orig.mean().item():.8f}")
                print(f"Max difference (batch1 vs batch2): {diff_1_2.max().item():.8f}")
                print(f"Mean difference (batch1 vs batch2): {diff_1_2.mean().item():.8f}")
                
                # Check if differences are close to tolerance
                print(f"\n=== TOLERANCE CHECK (atol=1e-5) ===")
                print(f"Batch1 vs Original within tolerance: {torch.allclose(batch_tensor_1, original_tensor, atol=1e-5, rtol=1e-5)}")
                print(f"Batch2 vs Original within tolerance: {torch.allclose(batch_tensor_2, original_tensor, atol=1e-5, rtol=1e-5)}")
                print(f"Batch1 vs Batch2 within tolerance: {torch.allclose(batch_tensor_1, batch_tensor_2, atol=1e-5, rtol=1e-5)}")

        else:
            print(f"✗ Output shapes don't match")

    else:
        print("✗ Output length doesn't scale correctly - possible batching issue")
    
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
                
                test_batchability(self.model, latent_model_input, timestep, arg_c['context'], arg_c['seq_len'])

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


    # TOTAL MODEL TIME
    # def generate(self,
    #              input_prompt,
    #              size=(1280, 720),
    #              frame_num=81,
    #              shift=5.0,
    #              sample_solver='unipc',
    #              sampling_steps=50,
    #              guide_scale=5.0,
    #              n_prompt="",
    #              seed=-1,
    #              offload_model=True):
    #     r"""
    #     Generates video frames from text prompt using diffusion process.

    #     Args:
    #         input_prompt (`str`):
    #             Text prompt for content generation
    #         size (tupele[`int`], *optional*, defaults to (1280,720)):
    #             Controls video resolution, (width,height).
    #         frame_num (`int`, *optional*, defaults to 81):
    #             How many frames to sample from a video. The number should be 4n+1
    #         shift (`float`, *optional*, defaults to 5.0):
    #             Noise schedule shift parameter. Affects temporal dynamics
    #         sample_solver (`str`, *optional*, defaults to 'unipc'):
    #             Solver used to sample the video.
    #         sampling_steps (`int`, *optional*, defaults to 40):
    #             Number of diffusion sampling steps. Higher values improve quality but slow generation
    #         guide_scale (`float`, *optional*, defaults 5.0):
    #             Classifier-free guidance scale. Controls prompt adherence vs. creativity
    #         n_prompt (`str`, *optional*, defaults to ""):
    #             Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
    #         seed (`int`, *optional*, defaults to -1):
    #             Random seed for noise generation. If -1, use random seed.
    #         offload_model (`bool`, *optional*, defaults to True):
    #             If True, offloads models to CPU during generation to save VRAM

    #     Returns:
    #         torch.Tensor:
    #             Generated video frames tensor. Dimensions: (C, N H, W) where:
    #             - C: Color channels (3 for RGB)
    #             - N: Number of frames (81)
    #             - H: Frame height (from size)
    #             - W: Frame width from size)
    #     """

    #     start_overall = torch.cuda.Event(enable_timing=True)
    #     end_overall = torch.cuda.Event(enable_timing=True)
    #     start_overall.record()
    


    #     # preprocess
    #     F = frame_num
    #     target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
    #                     size[1] // self.vae_stride[1],
    #                     size[0] // self.vae_stride[2])

    #     seq_len = math.ceil((target_shape[2] * target_shape[3]) /
    #                         (self.patch_size[1] * self.patch_size[2]) *
    #                         target_shape[1] / self.sp_size) * self.sp_size

    #     if n_prompt == "":
    #         n_prompt = self.sample_neg_prompt
    #     seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    #     seed_g = torch.Generator(device=self.device)
    #     seed_g.manual_seed(seed)

    #     if not self.t5_cpu:
    #         self.text_encoder.model.to(self.device)
    #         context = self.text_encoder([input_prompt], self.device)
    #         context_null = self.text_encoder([n_prompt], self.device)
    #         if offload_model:
    #             self.text_encoder.model.cpu()
    #     else:
    #         context = self.text_encoder([input_prompt], torch.device('cpu'))
    #         context_null = self.text_encoder([n_prompt], torch.device('cpu'))
    #         context = [t.to(self.device) for t in context]
    #         context_null = [t.to(self.device) for t in context_null]

    #     noise = [
    #         torch.randn(
    #             target_shape[0],
    #             target_shape[1],
    #             target_shape[2],
    #             target_shape[3],
    #             dtype=torch.float32,
    #             device=self.device,
    #             generator=seed_g)
    #     ]

    #     @contextmanager
    #     def noop_no_sync():
    #         yield

    #     no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    #     # evaluation mode
    #     with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

    #         if sample_solver == 'unipc':
    #             sample_scheduler = FlowUniPCMultistepScheduler(
    #                 num_train_timesteps=self.num_train_timesteps,
    #                 shift=1,
    #                 use_dynamic_shifting=False)
    #             sample_scheduler.set_timesteps(
    #                 sampling_steps, device=self.device, shift=shift)
    #             timesteps = sample_scheduler.timesteps
    #         elif sample_solver == 'dpm++':
    #             sample_scheduler = FlowDPMSolverMultistepScheduler(
    #                 num_train_timesteps=self.num_train_timesteps,
    #                 shift=1,
    #                 use_dynamic_shifting=False)
    #             sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
    #             timesteps, _ = retrieve_timesteps(
    #                 sample_scheduler,
    #                 device=self.device,
    #                 sigmas=sampling_sigmas)
    #         else:
    #             raise NotImplementedError("Unsupported solver.")

    #         # sample videos
    #         latents = noise

    #         arg_c = {'context': context, 'seq_len': seq_len}
    #         arg_null = {'context': context_null, 'seq_len': seq_len}

    #         for _, t in enumerate(tqdm(timesteps)):
    #             latent_model_input = latents
    #             timestep = [t]

    #             timestep = torch.stack(timestep)

    #             self.model.to(self.device)
    #             noise_pred_cond = self.model(
    #                 latent_model_input, t=timestep, **arg_c)[0]
    #             noise_pred_uncond = self.model(
    #                 latent_model_input, t=timestep, **arg_null)[0]

    #             noise_pred = noise_pred_uncond + guide_scale * (
    #                 noise_pred_cond - noise_pred_uncond)

    #             temp_x0 = sample_scheduler.step(
    #                 noise_pred.unsqueeze(0),
    #                 t,
    #                 latents[0].unsqueeze(0),
    #                 return_dict=False,
    #                 generator=seed_g)[0]
    #             latents = [temp_x0.squeeze(0)]

    #         x0 = latents
    #         if offload_model:
    #             self.model.cpu()
    #             torch.cuda.empty_cache()
    #         if self.rank == 0:
    #             videos = self.vae.decode(x0)

    #     del noise, latents
    #     del sample_scheduler
    #     if offload_model:
    #         gc.collect()
    #         torch.cuda.synchronize()
    #     if dist.is_initialized():
    #         dist.barrier()

    #     end_overall.record()
    #     torch.cuda.synchronize()
    #     total_time = start_overall.elapsed_time(end_overall)

    #     # Print timing summary
    #     if self.rank == 0:
    #         print("\n===== Model Timing Summary =====")
    #         print(f"Total End-to-End Time: {total_time:.2f} ms")
    #         print("==============================\n")


    #     return videos[0] if self.rank == 0 else None

    


    # def generate(self,
    #              input_prompt,
    #              size=(1280, 720),
    #              frame_num=81,
    #              shift=5.0,
    #              sample_solver='unipc',
    #              sampling_steps=50,
    #              guide_scale=5.0,
    #              n_prompt="",
    #              seed=-1,
    #              offload_model=True):
    #     r"""
    #     Generates video frames from text prompt using diffusion process.

    #     Args:
    #         input_prompt (`str`):
    #             Text prompt for content generation
    #         size (tupele[`int`], *optional*, defaults to (1280,720)):
    #             Controls video resolution, (width,height).
    #         frame_num (`int`, *optional*, defaults to 81):
    #             How many frames to sample from a video. The number should be 4n+1
    #         shift (`float`, *optional*, defaults to 5.0):
    #             Noise schedule shift parameter. Affects temporal dynamics
    #         sample_solver (`str`, *optional*, defaults to 'unipc'):
    #             Solver used to sample the video.
    #         sampling_steps (`int`, *optional*, defaults to 40):
    #             Number of diffusion sampling steps. Higher values improve quality but slow generation
    #         guide_scale (`float`, *optional*, defaults 5.0):
    #             Classifier-free guidance scale. Controls prompt adherence vs. creativity
    #         n_prompt (`str`, *optional*, defaults to ""):
    #             Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
    #         seed (`int`, *optional*, defaults to -1):
    #             Random seed for noise generation. If -1, use random seed.
    #         offload_model (`bool`, *optional*, defaults to True):
    #             If True, offloads models to CPU during generation to save VRAM

    #     Returns:
    #         torch.Tensor:
    #             Generated video frames tensor. Dimensions: (C, N H, W) where:
    #             - C: Color channels (3 for RGB)
    #             - N: Number of frames (81)
    #             - H: Frame height (from size)
    #             - W: Frame width from size)
    #     """
    #     # preprocess
    #     F = frame_num
    #     target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
    #                     size[1] // self.vae_stride[1],
    #                     size[0] // self.vae_stride[2])

    #     seq_len = math.ceil((target_shape[2] * target_shape[3]) /
    #                         (self.patch_size[1] * self.patch_size[2]) *
    #                         target_shape[1] / self.sp_size) * self.sp_size

    #     if n_prompt == "":
    #         n_prompt = self.sample_neg_prompt
    #     seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    #     seed_g = torch.Generator(device=self.device)
    #     seed_g.manual_seed(seed)

    #     if not self.t5_cpu:
    #         self.text_encoder.model.to(self.device)
    #         context = self.text_encoder([input_prompt], self.device)
    #         context_null = self.text_encoder([n_prompt], self.device)
    #         if offload_model:
    #             self.text_encoder.model.cpu()
    #     else:
    #         context = self.text_encoder([input_prompt], torch.device('cpu'))
    #         context_null = self.text_encoder([n_prompt], torch.device('cpu'))
    #         context = [t.to(self.device) for t in context]
    #         context_null = [t.to(self.device) for t in context_null]

    #     noise = [
    #         torch.randn(
    #             target_shape[0],
    #             target_shape[1],
    #             target_shape[2],
    #             target_shape[3],
    #             dtype=torch.float32,
    #             device=self.device,
    #             generator=seed_g)
    #     ]

    #     @contextmanager
    #     def noop_no_sync():
    #         yield

    #     no_sync = getattr(self.model, 'no_sync', noop_no_sync)

    #     # evaluation mode
    #     with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

    #         if sample_solver == 'unipc':
    #             sample_scheduler = FlowUniPCMultistepScheduler(
    #                 num_train_timesteps=self.num_train_timesteps,
    #                 shift=1,
    #                 use_dynamic_shifting=False)
    #             sample_scheduler.set_timesteps(
    #                 sampling_steps, device=self.device, shift=shift)
    #             timesteps = sample_scheduler.timesteps
    #         elif sample_solver == 'dpm++':
    #             sample_scheduler = FlowDPMSolverMultistepScheduler(
    #                 num_train_timesteps=self.num_train_timesteps,
    #                 shift=1,
    #                 use_dynamic_shifting=False)
    #             sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
    #             timesteps, _ = retrieve_timesteps(
    #                 sample_scheduler,
    #                 device=self.device,
    #                 sigmas=sampling_sigmas)
    #         else:
    #             raise NotImplementedError("Unsupported solver.")

    #         # sample videos
    #         latents = noise

    #         arg_c = {'context': context, 'seq_len': seq_len}
    #         arg_null = {'context': context_null, 'seq_len': seq_len}

    #         for _, t in enumerate(tqdm(timesteps)):
    #             latent_model_input = latents
    #             timestep = [t]

    #             timestep = torch.stack(timestep)

    #             self.model.to(self.device)
    #             noise_pred_cond = self.model(
    #                 latent_model_input, t=timestep, **arg_c)[0]
    #             noise_pred_uncond = self.model(
    #                 latent_model_input, t=timestep, **arg_null)[0]

    #             noise_pred = noise_pred_uncond + guide_scale * (
    #                 noise_pred_cond - noise_pred_uncond)

    #             temp_x0 = sample_scheduler.step(
    #                 noise_pred.unsqueeze(0),
    #                 t,
    #                 latents[0].unsqueeze(0),
    #                 return_dict=False,
    #                 generator=seed_g)[0]
    #             latents = [temp_x0.squeeze(0)]

    #         x0 = latents
    #         if offload_model:
    #             self.model.cpu()
    #             torch.cuda.empty_cache()
    #         if self.rank == 0:
    #             videos = self.vae.decode(x0)

    #     del noise, latents
    #     del sample_scheduler
    #     if offload_model:
    #         gc.collect()
    #         torch.cuda.synchronize()
    #     if dist.is_initialized():
    #         dist.barrier()

    #     return videos[0] if self.rank == 0 else None

    
