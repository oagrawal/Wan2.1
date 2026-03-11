#!/usr/bin/env python3
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
# Copyright 2025 The Huazhong University of Science and Technology VLRLab Authors. All rights reserved.
"""
EasyCache single-video script for Wan2.1 (T2V-1.3B or T2V-14B).

3 modes:
  baseline  – no skipping; records k_t and pred_change over denoising steps.
  easycache – fixed-threshold caching.
  adaptive  – low threshold at start/end of denoising, high in the middle.

Key Wan difference vs HunyuanVideo:
  Wan uses CFG (classifier-free guidance), so every denoising step calls the
  DiT model TWICE: once for the conditional branch (even cnt) and once for the
  unconditional branch (odd cnt).  EasyCache operates on *even* (condition)
  steps only and mirrors the skip decision to the paired odd step.

  With --sample_steps 50, there are 100 total model calls (cnt 0..99).
  ret_steps = 10*2 = 20  →  first 10 denoising steps always compute.
  cutoff_steps = sample_steps*2 - 2  →  last 1 denoising step always computes.
"""

import argparse
import gc
import json
import math
import os
import random
import sys
import warnings
from contextlib import contextmanager
from datetime import datetime
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from tqdm import tqdm

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.fm_solvers import (FlowDPMSolverMultistepScheduler,
                                  get_sampling_sigmas, retrieve_timesteps)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.utils.utils import cache_video, str2bool

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Forward: baseline (no skipping, records k_t and pred_change on even steps)
# ---------------------------------------------------------------------------

def easycache_baseline_forward_wan(self, x, t, context, seq_len,
                                    clip_fea=None, y=None):
    """
    Full model forward every call (no caching).
    On even (condition) steps, records:
      - k_t  = ||v_t - v_{t-2}|| / ||x_t - x_{t-2}||  (transformation rate)
      - pred_change = k_{t-2} * ||x_t - x_{t-2}|| / ||v_{t-2}||  (predicted relative change)
    """
    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None

    raw_input = [u.clone() for u in x]

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    self.is_even = (self.cnt % 2 == 0)

    is_eligible = (self.cnt >= self.ret_steps and self.cnt < self.cutoff_steps)

    # Compute pred_change for ALL even steps that have data, then record to:
    #   pred_change_all_history  — every step (full picture)
    #   pred_change_history      — eligible steps only (where caching decides)
    if self.is_even and self.k is not None and \
            self.previous_raw_input_even is not None and \
            self.previous_raw_output_even is not None:
        raw_input_change = torch.cat([
            (u - v).flatten() for u, v in zip(raw_input, self.previous_raw_input_even)
        ]).abs().mean()
        output_norm = torch.cat([
            u.flatten() for u in self.previous_raw_output_even
        ]).abs().mean()
        if output_norm > 0:
            pred_change = (self.k * (raw_input_change / output_norm)).item()
            if not self.pred_change_all_history:
                self.pred_change_all_start = self.cnt // 2  # record first cond step
            self.pred_change_all_history.append(pred_change)
            if is_eligible:
                self.pred_change_history.append(pred_change)

    # Snapshot current even-step input (used for k next time)
    if self.is_even:
        self._current_raw_input_even = [u.clone() for u in raw_input]

    # --- full model forward ---
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    context_lens = None
    context = self.text_embedding(torch.stack([
        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        for u in context
    ]))
    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
        freqs=self.freqs, context=context, context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)
    x = self.head(x, e)
    output = self.unpatchify(x, grid_sizes)

    # Post-forward: update EasyCache state on even steps
    if self.is_even:
        if self.previous_raw_output_even is not None and \
                self.prev_prev_raw_input_even is not None and \
                self._current_raw_input_even is not None:
            output_change = torch.cat([
                (u - v).flatten() for u, v in zip(output, self.previous_raw_output_even)
            ]).abs().mean()
            input_change = torch.cat([
                (u - v).flatten() for u, v in
                zip(self._current_raw_input_even, self.prev_prev_raw_input_even)
            ]).abs().mean()
            if input_change > 0:
                self.k = output_change / input_change
                self.k_history.append(self.k.item())

        self.prev_prev_raw_input_even = self.previous_raw_input_even
        self.previous_raw_input_even = self._current_raw_input_even
        self.previous_raw_output_even = [u.clone() for u in output]

    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0

    return [u.float() for u in output]


# ---------------------------------------------------------------------------
# Forward: EasyCache with fixed or adaptive thresholds, plus recording
# ---------------------------------------------------------------------------

def easycache_forward_wan(self, x, t, context, seq_len,
                          clip_fea=None, y=None):
    """
    EasyCache forward for Wan2.1 with optional adaptive thresholds.
    Operates on even (condition) steps; mirrors skip to paired odd step.
    Also records k_t and pred_change history for analysis.
    """
    if self.model_type == "i2v":
        assert clip_fea is not None and y is not None

    raw_input = [u.clone() for u in x]

    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    self.is_even = (self.cnt % 2 == 0)

    # Decision made on even steps; odd steps mirror it
    if self.is_even:
        cond_step = self.cnt // 2   # 0-indexed condition step

        if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
            # Forced compute (warmup or final step)
            self.should_calc_current_pair = True
            self.accumulated_error_even = 0.0
        elif self.previous_raw_input_even is not None and \
                self.previous_raw_output_even is not None and \
                self.k is not None:
            # Compute pred_change and decide
            raw_input_change = torch.cat([
                (u - v).flatten() for u, v in zip(raw_input, self.previous_raw_input_even)
            ]).abs().mean()
            output_norm = torch.cat([
                u.flatten() for u in self.previous_raw_output_even
            ]).abs().mean()
            pred_change = (self.k * (raw_input_change / output_norm)).item() \
                if output_norm > 0 else 0.0

            # Record pred_change
            self.pred_change_history.append(pred_change)
            self.accumulated_error_even += pred_change

            # Adaptive threshold selection
            current_thresh = self.thresh
            if getattr(self, "easycache_adaptive", False):
                fs = getattr(self, "first_steps", 8)
                ls = getattr(self, "last_steps", 6)
                n_cond = self.num_steps // 2   # total condition steps
                if cond_step < fs or cond_step >= n_cond - ls - 1:
                    current_thresh = self.thresh_low
                else:
                    current_thresh = self.thresh_high

            if self.accumulated_error_even < current_thresh:
                self.should_calc_current_pair = False
            else:
                self.should_calc_current_pair = True
                self.accumulated_error_even = 0.0
        else:
            self.should_calc_current_pair = True

        # Snapshot even-step input for k computation
        self._current_raw_input_even = [u.clone() for u in raw_input]

    # --- skip path ---
    if self.is_even and not self.should_calc_current_pair and \
            self.cache_even is not None:
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self.cnt = 0
        return [(u + v).float() for u, v in zip(raw_input, self.cache_even)]

    if not self.is_even and not self.should_calc_current_pair and \
            self.cache_odd is not None:
        self.cnt += 1
        if self.cnt >= self.num_steps:
            self.cnt = 0
        return [(u + v).float() for u, v in zip(raw_input, self.cache_odd)]

    # --- full model forward ---
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    context_lens = None
    context = self.text_embedding(torch.stack([
        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
        for u in context
    ]))
    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0, seq_lens=seq_lens, grid_sizes=grid_sizes,
        freqs=self.freqs, context=context, context_lens=context_lens)

    for block in self.blocks:
        x = block(x, **kwargs)
    x = self.head(x, e)
    output = self.unpatchify(x, grid_sizes)

    # Update EasyCache state
    if self.is_even:
        if self.previous_raw_output_even is not None and \
                self.prev_prev_raw_input_even is not None and \
                self._current_raw_input_even is not None:
            output_change = torch.cat([
                (u - v).flatten() for u, v in zip(output, self.previous_raw_output_even)
            ]).abs().mean()
            input_change = torch.cat([
                (u - v).flatten() for u, v in
                zip(self._current_raw_input_even, self.prev_prev_raw_input_even)
            ]).abs().mean()
            if input_change > 0:
                self.k = output_change / input_change
                self.k_history.append(self.k.item())

        self.prev_prev_raw_input_even = self.previous_raw_input_even
        self.previous_raw_input_even = self._current_raw_input_even
        self.previous_raw_output_even = [u.clone() for u in output]
        self.cache_even = [u - v for u, v in zip(output, raw_input)]
    else:
        self.previous_raw_output_odd = [u.clone() for u in output]
        self.cache_odd = [u - v for u, v in zip(output, raw_input)]

    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0

    return [u.float() for u in output]


# ---------------------------------------------------------------------------
# Custom t2v generate (records cost_time, same logic as easycache_generate.py)
# ---------------------------------------------------------------------------

def t2v_generate_wan(self, input_prompt, size=(1280, 720), frame_num=81,
                     shift=5.0, sample_solver="unipc", sampling_steps=50,
                     guide_scale=5.0, n_prompt="", seed=-1, offload_model=True):
    """Standard T2V generate used for all EasyCache modes."""
    F = frame_num
    target_shape = (
        self.vae.model.z_dim,
        (F - 1) // self.vae_stride[0] + 1,
        size[1] // self.vae_stride[1],
        size[0] // self.vae_stride[2],
    )
    seq_len = math.ceil(
        (target_shape[2] * target_shape[3]) /
        (self.patch_size[1] * self.patch_size[2]) *
        target_shape[1] / self.sp_size
    ) * self.sp_size

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt
    seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
    seed_g = torch.Generator(device=self.device)
    seed_g.manual_seed(seed)

    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        context = self.text_encoder([input_prompt], self.device)
        context_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        context = self.text_encoder([input_prompt], torch.device("cpu"))
        context_null = self.text_encoder([n_prompt], torch.device("cpu"))
        context = [t.to(self.device) for t in context]
        context_null = [t.to(self.device) for t in context_null]

    noise = [torch.randn(
        target_shape[0], target_shape[1], target_shape[2], target_shape[3],
        dtype=torch.float32, device=self.device, generator=seed_g)]

    @contextmanager
    def noop_no_sync():
        yield

    no_sync = getattr(self.model, "no_sync", noop_no_sync)

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1, use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif sample_solver == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1, use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=self.device,
                                              sigmas=sampling_sigmas)
        else:
            raise NotImplementedError("Unsupported solver.")

        latents = noise
        arg_c = {"context": context, "seq_len": seq_len}
        arg_null = {"context": context_null, "seq_len": seq_len}

        for _, t in enumerate(tqdm(timesteps)):
            torch.cuda.synchronize()
            step_start = time()
            latent_model_input = latents
            timestep = torch.stack([t])

            self.model.to(self.device)
            noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
            noise_pred_uncond = self.model(latent_model_input, t=timestep, **arg_null)[0]
            noise_pred = noise_pred_uncond + guide_scale * (noise_pred_cond - noise_pred_uncond)

            torch.cuda.synchronize()
            self.cost_time += (time() - step_start)

            temp_x0 = sample_scheduler.step(
                noise_pred.unsqueeze(0), t, latents[0].unsqueeze(0),
                return_dict=False, generator=seed_g)[0]
            latents = [temp_x0.squeeze(0)]

        x0 = latents
        if offload_model:
            self.model.cpu()
            torch.cuda.empty_cache()
        if self.rank == 0:
            videos = self.vae.decode(x0)

    del noise, latents, sample_scheduler
    if offload_model:
        gc.collect()
        torch.cuda.synchronize()
    if dist.is_initialized():
        dist.barrier()

    return videos[0] if self.rank == 0 else None


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------

def configure_model(wan_t2v, mode, sample_steps, thresh=0.025,
                    thresh_low=0.025, thresh_high=0.05,
                    first_steps=8, last_steps=6, ret_steps=10):
    """
    Set up EasyCache on the Wan model.

    IMPORTANT: State is set on the model INSTANCE (not the class) so that
    repeated calls properly reset stale values from previous runs.
    The forward method is still set on the class (required for nn.Module dispatch).
    """
    model = wan_t2v.model
    model_cls = model.__class__

    if mode == "baseline":
        model_cls.forward = easycache_baseline_forward_wan
    else:
        model_cls.forward = easycache_forward_wan

    # Reset all per-run state on the INSTANCE
    model.cnt = 0
    model.num_steps = sample_steps * 2
    model.k = None
    model.is_even = True
    model.should_calc_current_pair = True
    model.accumulated_error_even = 0.0
    model.previous_raw_input_even = None
    model.previous_raw_output_even = None
    model.previous_raw_output_odd = None
    model.prev_prev_raw_input_even = None
    model._current_raw_input_even = None
    model.cache_even = None
    model.cache_odd = None
    model.skip_cond_step = []
    model.skip_uncond_step = []
    model.k_history = []
    model.pred_change_history = []        # eligible steps only
    model.pred_change_all_history = []    # all steps (for full-picture plot)
    model.pred_change_all_start = 0
    model.ret_steps = ret_steps * 2
    model.cutoff_steps = sample_steps * 2 - 2

    if mode == "adaptive":
        model.easycache_adaptive = True
        model.thresh = thresh_low
        model.thresh_low = thresh_low
        model.thresh_high = thresh_high
        model.first_steps = first_steps
        model.last_steps = last_steps
    else:
        model.easycache_adaptive = False
        model.thresh = thresh
        model.thresh_low = thresh
        model.thresh_high = thresh

    # Patch the generate method on the class
    wan_t2v.__class__.generate = t2v_generate_wan
    wan_t2v.__class__.cost_time = 0
    wan_t2v.cost_time = 0


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_plot(values, title, ylabel, save_path, steps, x_start=None, vlines=None):
    """
    vlines: optional list of (x, color, label) to draw vertical reference lines.
    """
    if not values:
        return
    if x_start is None:
        x_start = steps - len(values)
    xs = list(range(x_start, x_start + len(values)))
    plt.figure(figsize=(10, 5))
    plt.plot(xs, values, linewidth=2, marker="o", markersize=4)
    if vlines:
        for x_vline, color, label in vlines:
            plt.axvline(x=x_vline, color=color, linestyle="--", alpha=0.8, label=label)
        plt.legend(fontsize=9)
    plt.xlabel("Condition step (even model calls)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.gca().xaxis.set_major_locator(plt.MultipleLocator(max(1, steps // 10)))
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Wan2.1 EasyCache: baseline profiling and caching modes.")

    # EasyCache experiment args
    p.add_argument("--easycache-mode", type=str, default="baseline",
                   choices=["baseline", "easycache", "adaptive"],
                   help="baseline: profile k_t/pred_change; easycache: fixed threshold; "
                        "adaptive: low at start/end, high in middle.")
    p.add_argument("--easycache-thresh", type=float, default=0.025,
                   help="Fixed threshold (easycache mode).")
    p.add_argument("--easycache-thresh-low", type=float, default=0.025,
                   help="Low threshold (adaptive mode, start/end steps).")
    p.add_argument("--easycache-thresh-high", type=float, default=0.05,
                   help="High threshold (adaptive mode, middle steps).")
    p.add_argument("--easycache-first-steps", type=int, default=12,
                   help="First N condition steps that use low threshold (adaptive). "
                        "With ret_steps=5, covers eligible steps 5-11 (highest pred_change).")
    p.add_argument("--easycache-last-steps", type=int, default=4,
                   help="Last N condition steps that use low threshold (adaptive). "
                        "Covers steps 45-47 where pred_change rises again.")
    p.add_argument("--easycache-ret-steps", type=int, default=5,
                   help="First N condition steps always compute (no skipping). "
                        "Default 5 exposes more volatile eligible steps to adaptive thresholding.")

    # Wan model args
    p.add_argument("--task", type=str, default="t2v-1.3B",
                   choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--size", type=str, default="832*480",
                   choices=list(SIZE_CONFIGS.keys()))
    p.add_argument("--frame_num", type=int, default=None)
    p.add_argument("--ckpt_dir", type=str, default="../Wan2.1-T2V-1.3B")
    p.add_argument("--prompt", type=str,
                   default="Two anthropomorphic cats in comfy boxing gear and bright "
                           "gloves fight intensely on a spotlighted stage.")
    p.add_argument("--save_file", type=str, default=None)
    p.add_argument("--save_dir", type=str, default="./easycache_results_wan")
    p.add_argument("--base_seed", type=int, default=12345)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--sample_solver", type=str, default="unipc",
                   choices=["unipc", "dpm++"])
    p.add_argument("--sample_shift", type=float, default=5.0)
    p.add_argument("--sample_guide_scale", type=float, default=5.0)
    p.add_argument("--offload_model", type=str2bool, default=True)
    p.add_argument("--t5_cpu", action="store_true", default=False)
    return p.parse_args()


def main():
    args = _parse_args()

    if args.frame_num is None:
        args.frame_num = 81
    if args.base_seed < 0:
        args.base_seed = random.randint(0, sys.maxsize)

    # Timestamp-based output folder
    ts = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    prompt_short = args.prompt.replace("/", "").replace(" ", "_")[:50]
    mode_tag = {
        "baseline": "baseline_profile",
        "easycache": f"easycache_thr{args.easycache_thresh}",
        "adaptive": f"easycache_adaptive_l{args.easycache_thresh_low}_h{args.easycache_thresh_high}",
    }[args.easycache_mode]
    folder_name = f"{mode_tag}_{ts}_seed{args.base_seed}_{prompt_short}"
    save_dir = os.path.join(args.save_dir, folder_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n=== Wan2.1 EasyCache ===")
    print(f"Mode       : {args.easycache_mode}")
    if args.easycache_mode == "easycache":
        print(f"Threshold  : {args.easycache_thresh}")
    elif args.easycache_mode == "adaptive":
        print(f"Thresh low : {args.easycache_thresh_low}  (first {args.easycache_first_steps} + last {args.easycache_last_steps} condition steps)")
        print(f"Thresh high: {args.easycache_thresh_high}  (middle steps)")
    print(f"ret_steps  : {args.easycache_ret_steps}  (always compute)")
    print(f"Output dir : {save_dir}")
    print(f"========================\n")

    cfg = WAN_CONFIGS[args.task]
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=args.t5_cpu,
    )

    configure_model(
        wan_t2v,
        mode=args.easycache_mode,
        sample_steps=args.sample_steps,
        thresh=args.easycache_thresh,
        thresh_low=args.easycache_thresh_low,
        thresh_high=args.easycache_thresh_high,
        first_steps=args.easycache_first_steps,
        last_steps=args.easycache_last_steps,
        ret_steps=args.easycache_ret_steps,
    )

    e2e_start = time()
    video = wan_t2v.generate(
        args.prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.sample_shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sample_steps,
        guide_scale=args.sample_guide_scale,
        seed=args.base_seed,
        offload_model=args.offload_model,
    )
    e2e_time = time() - e2e_start
    dit_time = wan_t2v.cost_time
    print(f"End-to-end time: {e2e_time:.2f}s  |  DiT-only time: {dit_time:.2f}s")

    # Save video
    if args.save_file is None:
        args.save_file = os.path.join(save_dir, "video.mp4")
    cache_video(tensor=video[None], save_file=args.save_file,
                fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
    print(f"Video saved to: {args.save_file}")

    # Read back history
    k_history = wan_t2v.model.k_history
    pred_change_history     = wan_t2v.model.pred_change_history      # eligible only
    pred_change_all_history = wan_t2v.model.pred_change_all_history  # all steps
    pred_change_all_start   = wan_t2v.model.pred_change_all_start
    n_cond = args.sample_steps   # total condition steps
    # First eligible condition step = ret_steps (forced warmup ends there).
    # Last eligible condition step = sample_steps - 2  (cutoff_steps // 2 - 1).
    # e.g. ret_steps=10, sample_steps=50 → eligible = steps 10..48 (39 steps).
    eligible_start = args.easycache_ret_steps
    eligible_end   = args.sample_steps - 2   # inclusive

    # k_t plot (all condition steps where k can be computed, including warmup)
    if k_history:
        _save_plot(k_history, "k_t = ||v_t - v_{t-2}|| / ||x_t - x_{t-2}||  (all condition steps)",
                   "k_t", os.path.join(save_dir, "k_t_plot.png"), n_cond)
        with open(os.path.join(save_dir, "k_t.txt"), "w") as f:
            for v in k_history:
                f.write(f"{v}\n")
        print(f"k_t plot + values saved  ({len(k_history)} condition steps)")

    # pred_change plot 1 — ALL steps (full picture with eligible-range markers)
    if pred_change_all_history:
        _save_plot(
            pred_change_all_history,
            f"pred_change  [ALL condition steps, dashed = eligible boundary]\n"
            f"k_{{t-2}} · (||x_t - x_{{t-2}}|| / ||v_{{t-2}}||)",
            "pred_change",
            os.path.join(save_dir, "pred_change_all_plot.png"),
            n_cond,
            x_start=pred_change_all_start,
            vlines=[
                (eligible_start,     "green", f"eligible start (step {eligible_start})"),
                (eligible_end + 1,   "red",   f"eligible end   (step {eligible_end})"),
            ],
        )
        with open(os.path.join(save_dir, "pred_change_all.txt"), "w") as f:
            for v in pred_change_all_history:
                f.write(f"{v}\n")
        print(f"pred_change_all plot + values saved  "
              f"({len(pred_change_all_history)} steps, starting at cond step {pred_change_all_start})")

    # pred_change plot 2 — ELIGIBLE steps only (where caching decisions happen)
    if pred_change_history:
        _save_plot(
            pred_change_history,
            f"pred_change  [eligible steps {eligible_start}–{eligible_end} only]\n"
            f"k_{{t-2}} · (||x_t - x_{{t-2}}|| / ||v_{{t-2}}||)",
            "pred_change",
            os.path.join(save_dir, "pred_change_plot.png"),
            n_cond,
            x_start=eligible_start,
        )
        with open(os.path.join(save_dir, "pred_change.txt"), "w") as f:
            for v in pred_change_history:
                f.write(f"{v}\n")
        print(f"pred_change plot + values saved  "
              f"({len(pred_change_history)} eligible steps, "
              f"condition steps {eligible_start}–{eligible_end})")

    # Diagnostic file
    k_arr = np.array(k_history) if k_history else np.array([])
    with open(os.path.join(save_dir, "diagnostic_info.txt"), "w") as f:
        f.write("Wan2.1 EasyCache Experiment\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mode: {args.easycache_mode}\n")
        f.write(f"Task: {args.task}  Size: {args.size}\n")
        if args.easycache_mode == "easycache":
            f.write(f"Threshold: {args.easycache_thresh}\n")
        elif args.easycache_mode == "adaptive":
            f.write(f"Thresh low/high: {args.easycache_thresh_low} / {args.easycache_thresh_high}\n")
            f.write(f"First/last steps: {args.easycache_first_steps} / {args.easycache_last_steps}\n")
        f.write(f"ret_steps: {args.easycache_ret_steps}\n")
        f.write("\n=== Timing ===\n")
        f.write(f"End-to-end time:   {e2e_time:.2f}s\n")
        f.write(f"DiT-only time:     {dit_time:.2f}s\n")
        f.write(f"Sample steps:      {args.sample_steps}  (model calls: {args.sample_steps * 2})\n")
        if k_arr.size > 0:
            f.write("\n=== k_t Statistics ===\n")
            f.write(f"k_t entries: {len(k_arr)}\n")
            f.write(f"Mean:  {k_arr.mean():.4f}\n")
            f.write(f"Std:   {k_arr.std():.4f}\n")
            f.write(f"Min:   {k_arr.min():.4f}\n")
            f.write(f"Max:   {k_arr.max():.4f}\n")
        f.write("\n=== Prompt ===\n")
        f.write(f"{args.prompt}\n")
        f.write(f"\n=== Seed ===\n{args.base_seed}\n")
    print(f"Diagnostic saved to: {os.path.join(save_dir, 'diagnostic_info.txt')}")


if __name__ == "__main__":
    main()
