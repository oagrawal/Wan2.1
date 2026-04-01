#!/usr/bin/env python3
"""
Batch video generation for Wan2.1 DiCache VBench evaluation.

Loads the Wan model ONCE, then loops over prompts and modes.
Saves videos in VBench naming format: {prompt_text}-{seed}.mp4

Supports:
  - Fixed threshold:   --delta 0.10
  - Adaptive schedule: --delta-low 0.05 --delta-high 0.20
                       --stable-start 20 --stable-end 80
                       (cnt-index schedule over num_steps=sample_steps*2 forward calls)
  - True no-caching baseline: --delta 0   (accumulator always > 0 → never skips)

GPU split: use --start-idx / --end-idx with disjoint prompt ranges.
Resume: skips video files that already exist.

IMPORTANT — instance attribute fix (same as EasyCache batch_generate_wan.py):
  DiCache state is set on the MODEL INSTANCE (wan_t2v.model.cnt = 0, …)
  to avoid Python shadowing issues when self.cnt += 1 creates an instance attr.
  The forward method is still patched on the CLASS.
"""

import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Union

import torch

WAN_ROOT = str(Path(__file__).resolve().parent.parent)
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.utils.utils import cache_video

# Import DiCache forward from the existing driver
sys.path.insert(0, os.path.join(WAN_ROOT, "dicache"))
from run_wan_dicache import dicache_forward  # noqa: E402


# ---------------------------------------------------------------------------
# Mode definitions — edit to add/remove modes before running
# ---------------------------------------------------------------------------
MODES = [
    {"name": "wan_dc_baseline",     "delta": 0.0,  "ret_ratio": 0.0},
    {"name": "wan_dc_fixed_0.05",   "delta": 0.05, "ret_ratio": 0.0},
    {"name": "wan_dc_fixed_0.10",   "delta": 0.10, "ret_ratio": 0.0},
    {"name": "wan_dc_fixed_0.15",   "delta": 0.15, "ret_ratio": 0.0},
    {"name": "wan_dc_fixed_0.20",   "delta": 0.20, "ret_ratio": 0.0},
    {"name": "wan_dc_fixed_0.30",   "delta": 0.30, "ret_ratio": 0.0},
    {"name": "wan_dc_fixed_0.40",   "delta": 0.40, "ret_ratio": 0.0},
    # Adaptive (fill in stable_start/stable_end after fixed sweep + VBench):
    # {"name": "wan_dc_adaptive_0.10_0.30",
    #  "delta_low": 0.10, "delta_high": 0.30,
    #  "stable_start": 10, "stable_end": 70,
    #  "ret_ratio": 0.0},
]


# ---------------------------------------------------------------------------
# Model configuration helpers
# ---------------------------------------------------------------------------

def build_delta_schedule(
    num_steps: int,
    delta_low: float,
    delta_high: float,
    stable_start: int,
    stable_end: int,
) -> list:
    """Return per-cnt-index threshold list (length = num_steps)."""
    sched = []
    for i in range(num_steps):
        if stable_start <= i < stable_end:
            sched.append(delta_high)
        else:
            sched.append(delta_low)
    return sched


def configure_dicache(model, mode_cfg: dict, sample_steps: int, probe_depth: int = 1):
    """
    Patch the Wan model instance for DiCache.
    model = wan_t2v.model  (the underlying transformer)
    """
    num_steps = sample_steps * 2  # CFG doubles forward calls

    # Build threshold: scalar or per-step list
    if "delta_low" in mode_cfg:
        delta_eff = build_delta_schedule(
            num_steps,
            delta_low=mode_cfg["delta_low"],
            delta_high=mode_cfg["delta_high"],
            stable_start=mode_cfg["stable_start"],
            stable_end=mode_cfg["stable_end"],
        )
        label = (f"adaptive low={mode_cfg['delta_low']} high={mode_cfg['delta_high']} "
                 f"stable=[{mode_cfg['stable_start']},{mode_cfg['stable_end']})")
    else:
        delta_eff = float(mode_cfg["delta"])
        label = f"fixed delta={delta_eff}"

    # Patch forward on class (nn.Module dispatch requires this)
    model.__class__.forward = dicache_forward

    # State on INSTANCE to avoid class-attribute shadowing on cnt += 1
    model.cnt = 0
    model.probe_depth = probe_depth
    model.num_steps = num_steps
    model.rel_l1_thresh = delta_eff
    model.ret_ratio = float(mode_cfg.get("ret_ratio", 0.0))
    model.accumulated_rel_l1_distance = [0.0, 0.0]
    model.residual_cache = [None, None]
    model.probe_residual_cache = [None, None]
    model.residual_window = [[], []]
    model.probe_residual_window = [[], []]
    model.previous_internal_states = [None, None]
    model.previous_input = [None, None]
    model.previous_output = [None, None]
    model.resume_flag = [False, False]

    return label


# ---------------------------------------------------------------------------
# Generation log helpers (same format as batch_generate_wan.py)
# ---------------------------------------------------------------------------

def load_gen_log(log_path):
    if os.path.exists(log_path):
        with open(log_path) as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_gen_log(log_path, data):
    tmp = log_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp, log_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(
        description="Wan2.1 DiCache VBench batch generation"
    )
    p.add_argument("--prompts-file", type=str,
                   default=os.path.join(WAN_ROOT, "vbench_eval", "prompts_subset.json"))
    p.add_argument("--output-dir", type=str,
                   default=os.path.join(WAN_ROOT, "dicache_results", "videos"))
    p.add_argument("--ckpt_dir", type=str,
                   default="/nfs/oagrawal/wan/Wan2.1-T2V-1.3B",
                   help="Path to Wan2.1 checkpoint directory")
    p.add_argument("--task", type=str, default="t2v-1.3B",
                   choices=list(WAN_CONFIGS.keys()))
    p.add_argument("--size", type=str, default="832*480")
    p.add_argument("--sample-steps", type=int, default=50)
    p.add_argument("--generation-seed", type=int, default=0)
    p.add_argument("--start-idx", type=int, default=0)
    p.add_argument("--end-idx", type=int, default=-1)
    p.add_argument("--modes", type=str, default="all",
                   help="Comma-separated mode names or 'all'")
    p.add_argument("--probe-depth", type=int, default=1)
    p.add_argument("--offload-model", action="store_true", default=True)
    p.add_argument("--t5-cpu", action="store_true", default=False)
    p.add_argument("--dry-run", action="store_true")

    # Adaptive threshold args (override mode-list values if provided)
    p.add_argument("--delta", type=float, default=None,
                   help="Fixed threshold (single mode shortcut)")
    p.add_argument("--mode-name", type=str, default=None,
                   help="Mode name when using --delta shortcut")
    p.add_argument("--delta-low", type=float, default=None)
    p.add_argument("--delta-high", type=float, default=None)
    p.add_argument("--stable-start", type=int, default=None,
                   help="cnt index where stable (HIGH) zone starts")
    p.add_argument("--stable-end", type=int, default=None,
                   help="cnt index where stable zone ends")
    p.add_argument("--ret-ratio", type=float, default=0.0)

    args = p.parse_args()

    with open(args.prompts_file) as f:
        all_prompts = json.load(f)
    end_idx = len(all_prompts) if args.end_idx == -1 else args.end_idx
    prompts = all_prompts[args.start_idx:end_idx]

    # Resolve mode list
    if args.mode_name is not None:
        # Single-mode shortcut from CLI
        mode_cfg = {"delta": args.delta, "ret_ratio": args.ret_ratio}
        if args.delta_low is not None:
            num_steps = args.sample_steps * 2
            mode_cfg = {
                "delta_low": args.delta_low,
                "delta_high": args.delta_high,
                "stable_start": args.stable_start,
                "stable_end": args.stable_end,
                "ret_ratio": args.ret_ratio,
            }
        modes = [{"name": args.mode_name, **mode_cfg}]
    elif args.modes == "all":
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
    print("Wan2.1 DiCache VBench Batch Generation")
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
                print(f"  {'EXISTS' if os.path.exists(path) else 'NEW'} "
                      f"{m['name']}/{prompt[:50]}")
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
    gen_log = load_gen_log(log_path)

    completed, skipped, failed = 0, 0, 0
    total_gen_time = 0.0

    for prompt_idx, entry in enumerate(prompts):
        prompt = entry["prompt_en"]
        global_idx = args.start_idx + prompt_idx

        for mode in modes:
            mode_name = mode["name"]
            video_dir = os.path.join(output_dir, mode_name)
            video_path = os.path.join(video_dir, f"{prompt}-{seed}.mp4")
            run_num = prompt_idx * len(modes) + modes.index(mode) + 1

            if os.path.exists(video_path):
                print(f"[{run_num}/{total_videos}] SKIP: {mode_name} | {prompt[:50]}...")
                skipped += 1
                continue

            # Reconfigure model for this mode
            label = configure_dicache(
                wan_t2v.model,
                mode_cfg=mode,
                sample_steps=args.sample_steps,
                probe_depth=args.probe_depth,
            )

            print(f"[{run_num}/{total_videos}] {mode_name} ({label}) | {prompt[:50]}...")

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
                    "prompt": prompt,
                    "seed": seed,
                    "mode": mode_name,
                    "time_seconds": round(gen_time, 2),
                    "dit_time_seconds": round(getattr(wan_t2v, "cost_time", gen_time), 2),
                    "video_path": video_path,
                    "prompt_index": global_idx,
                    "timestamp": datetime.now().isoformat(),
                })
                gen_log["completed_keys"].append(run_key)
                save_gen_log(log_path, gen_log)

                completed += 1
                total_gen_time += gen_time
                dit_t = getattr(wan_t2v, "cost_time", gen_time)
                print(f"  Saved {video_path} (e2e: {gen_time:.1f}s, DiT: {dit_t:.1f}s)")

            except Exception as exc:
                print(f"  FAILED: {exc}")
                import traceback
                traceback.print_exc()
                failed += 1
                gen_log["runs"].append({
                    "prompt": prompt, "seed": seed, "mode": mode_name,
                    "error": str(exc), "prompt_index": global_idx,
                    "timestamp": datetime.now().isoformat(),
                })
                save_gen_log(log_path, gen_log)

    print("\n" + "=" * 70)
    print(f"Completed: {completed}  Skipped: {skipped}  Failed: {failed}")
    if completed:
        print(f"Total time: {total_gen_time:.1f}s  ({total_gen_time/3600:.1f}h)")
        print(f"Avg per video: {total_gen_time/completed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
