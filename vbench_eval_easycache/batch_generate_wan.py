#!/usr/bin/env python3
"""
Batch video generation for EasyCache VBench evaluation — Wan2.1.

Loads the Wan model ONCE then loops over prompts and 4 EasyCache modes.
Saves videos in VBench naming format: {prompt}-{seed}.mp4

4 modes:
  wan_ec_baseline       – no caching (ground truth)
  wan_ec_fixed_0.025    – fixed threshold 0.025
  wan_ec_fixed_0.050    – fixed threshold 0.050
  wan_ec_adaptive       – low=0.025 at first/last N condition steps, high=0.050 in middle

Resume: skips video files that already exist.
GPU split: use --start-idx / --end-idx with disjoint prompt ranges (no race conditions).

IMPORTANT — instance attribute fix:
  Wan's EasyCache state lives on class attributes (self.cnt, self.k, …). When the
  forward uses `self.cnt += 1`, Python creates an *instance* attribute that shadows
  the class attribute on the next reset. configure_easycache_wan() sets all state on
  the MODEL INSTANCE (wan_t2v.model.cnt = 0, …) so resets are correct across runs.
  The `forward` method must still be set on the class (nn.Module dispatch requires it).
"""

import gc
import json
import math
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

# Wan repo root (parent of this file's directory)
WAN_ROOT = str(Path(__file__).resolve().parent.parent)
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import cache_video

# Import our forward functions and generate
from easycache_sample_video_wan import (
    easycache_baseline_forward_wan,
    easycache_forward_wan,
    t2v_generate_wan,
    configure_model,
)

MODES = [
    {"name": "wan_ec_baseline",    "mode": "baseline"},
    {"name": "wan_ec_fixed_0.020", "mode": "easycache", "thresh": 0.020},
    {"name": "wan_ec_fixed_0.025", "mode": "easycache", "thresh": 0.025},
    {"name": "wan_ec_fixed_0.040", "mode": "easycache", "thresh": 0.040},
    {"name": "wan_ec_fixed_0.050", "mode": "easycache", "thresh": 0.050},
    # Adaptive: thresh_low for volatile early (steps < first_steps) and late
    # (steps >= n_cond - last_steps - 1), thresh_high for stable middle.
    # first_steps=12, last_steps=4 with ret_steps=5 protects exactly the
    # eligible steps where pred_change is highest (steps 5-11 and 45-47),
    # giving ~77% skip rate vs ~81% for fixed 0.050 — close speedup, better quality.
    {"name": "wan_ec_adaptive",    "mode": "adaptive",
     "thresh_low": 0.025, "thresh_high": 0.050,
     "first_steps": 12, "last_steps": 4},
    # Adaptive first16: low=0.020, high=0.040 (tighter than 0.025/0.050)
    {"name": "wan_ec_adaptive_16_020040", "mode": "adaptive",
     "thresh_low": 0.020, "thresh_high": 0.040,
     "first_steps": 16, "last_steps": 4},
]


def load_generation_log(log_path):
    if os.path.exists(log_path):
        with open(log_path) as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_generation_log(log_path, data):
    tmp = log_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, log_path)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Wan2.1 EasyCache VBench batch generation")
    p.add_argument("--prompts-file", type=str,
                   default=os.path.join(WAN_ROOT, "vbench_eval", "prompts_subset.json"))
    p.add_argument("--output-dir", type=str,
                   default=os.path.join(WAN_ROOT, "vbench_eval_easycache", "videos"))
    p.add_argument("--ckpt_dir", type=str, required=True,
                   help="Path to Wan2.1 checkpoint (e.g. ../Wan2.1-T2V-1.3B)")
    p.add_argument("--task", type=str, default="t2v-1.3B",
                   choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--size", type=str, default="832*480")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--generation-seed", type=int, default=0)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=-1)
    p.add_argument("--modes", type=str, default="all",
                   help="Comma-separated mode names or 'all'")
    p.add_argument("--offload-model", action="store_true", default=True)
    p.add_argument("--t5-cpu", action="store_true", default=False)
    p.add_argument("--ret-steps", type=int, default=5,
                   help="First N condition steps that always compute (no skipping). "
                        "Default 5 exposes more volatile steps to adaptive thresholding.")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    with open(args.prompts_file) as f:
        all_prompts = json.load(f)
    end_idx = len(all_prompts) if args.end_idx == -1 else args.end_idx
    prompts = all_prompts[args.start_idx:end_idx]

    if args.modes == "all":
        modes = MODES
    else:
        names = {m.strip() for m in args.modes.split(",")}
        modes = [m for m in MODES if m["name"] in names]
        if not modes:
            print(f"ERROR: No valid modes. Available: {[m['name'] for m in MODES]}")
            sys.exit(1)

    seed = args.generation_seed
    output_dir = os.path.abspath(args.output_dir)
    total_videos = len(prompts) * len(modes)

    print("=" * 70)
    print("Wan2.1 EasyCache VBench Batch Generation")
    print("=" * 70)
    print(f"Prompts: [{args.start_idx}, {end_idx}) = {len(prompts)}")
    print(f"Modes: {[m['name'] for m in modes]}")
    print(f"Total videos: {total_videos}")
    print(f"Output: {output_dir}")
    print("=" * 70)

    if args.dry_run:
        for entry in prompts:
            prompt = entry["prompt_en"]
            for m in modes:
                path = os.path.join(output_dir, m["name"], f"{prompt}-{seed}.mp4")
                print(f"  {'EXISTS' if os.path.exists(path) else 'NEW'} {m['name']}/{prompt[:50]}")
        return

    # Load model once
    print("\nLoading Wan2.1 model...")
    cfg = WAN_CONFIGS[args.task]
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=args.t5_cpu,
    )
    print("Model loaded.\n")

    log_filename = f"generation_log_{args.start_idx}-{end_idx}.json"
    log_path = os.path.join(output_dir, log_filename)
    os.makedirs(output_dir, exist_ok=True)
    gen_log = load_generation_log(log_path)

    completed, skipped, failed = 0, 0, 0
    total_gen_time = 0.0

    for prompt_idx, entry in enumerate(prompts):
        prompt = entry["prompt_en"]
        global_idx = args.start_idx + prompt_idx

        for mode_idx, mode in enumerate(modes):
            mode_name = mode["name"]
            video_dir = os.path.join(output_dir, mode_name)
            video_path = os.path.join(video_dir, f"{prompt}-{seed}.mp4")
            run_num = prompt_idx * len(modes) + mode_idx + 1

            if os.path.exists(video_path):
                print(f"[{run_num}/{total_videos}] SKIP: {mode_name} | {prompt[:50]}...")
                skipped += 1
                continue

            # Configure model for this mode (sets instance attributes — no stale state)
            configure_model(
                wan_t2v,
                mode=mode["mode"],
                sample_steps=args.sample_steps,
                thresh=mode.get("thresh", 0.025),
                thresh_low=mode.get("thresh_low", 0.025),
                thresh_high=mode.get("thresh_high", 0.050),
                first_steps=mode.get("first_steps", 12),
                last_steps=mode.get("last_steps", 4),
                ret_steps=args.ret_steps,
            )

            print(f"[{run_num}/{total_videos}] Generating: {mode_name} | {prompt[:50]}...")

            try:
                os.makedirs(video_dir, exist_ok=True)
                t0 = time.time()
                video = wan_t2v.generate(
                    prompt,
                    size=SIZE_CONFIGS[args.size],
                    frame_num=81,
                    shift=5.0,
                    sample_solver="unipc",
                    sampling_steps=args.sample_steps,
                    guide_scale=5.0,
                    seed=seed,
                    offload_model=args.offload_model,
                )
                gen_time = time.time() - t0

                if video is not None:
                    cache_video(
                        tensor=video[None],
                        save_file=video_path,
                        fps=cfg.sample_fps,
                        nrow=1,
                        normalize=True,
                        value_range=(-1, 1),
                    )

                run_key = f"{mode_name}|{prompt}|{seed}"
                gen_log["runs"].append({
                    "prompt": prompt, "seed": seed, "mode": mode_name,
                    "time_seconds": round(gen_time, 2),
                    "dit_time_seconds": round(wan_t2v.cost_time, 2),
                    "video_path": video_path,
                    "prompt_index": global_idx,
                    "timestamp": datetime.now().isoformat(),
                })
                gen_log["completed_keys"].append(run_key)
                save_generation_log(log_path, gen_log)

                completed += 1
                total_gen_time += gen_time
                print(f"  Saved {video_path} (e2e: {gen_time:.1f}s, DiT: {wan_t2v.cost_time:.1f}s)")

            except Exception as e:
                print(f"  FAILED: {e}")
                import traceback; traceback.print_exc()
                failed += 1
                gen_log["runs"].append({
                    "prompt": prompt, "seed": seed, "mode": mode_name,
                    "error": str(e), "prompt_index": global_idx,
                    "timestamp": datetime.now().isoformat(),
                })
                save_generation_log(log_path, gen_log)

    print("\n" + "=" * 70)
    print(f"Completed: {completed}  Skipped: {skipped}  Failed: {failed}")
    if completed:
        print(f"Total time: {total_gen_time:.1f}s ({total_gen_time/3600:.1f}h)")
        print(f"Avg per video: {total_gen_time/completed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
