#!/usr/bin/env python3
"""
Batch video generation for VBench evaluation using Wan 2.1 T2V 1.3B + TeaCache.

Generates videos for each prompt and mode by calling teacache_generate.py
(subprocess per video). Saves in VBench naming: {prompt}-{seed}.mp4.
Supports resume (skips existing), --start-idx/--end-idx for GPU splitting.

Modes (same 4 as HunyuanVideo pipeline):
  wan_baseline     — no cache (teacache_thresh 0)
  wan_fixed_0.1    — fixed threshold 0.1
  wan_fixed_0.2    — fixed threshold 0.2
  wan_adaptive     — Wan uses single thresh; we use 0.2 (same as fixed_0.2)

Usage (from Wan2.1 repo root, inside Wan Docker/venv):
  CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_wan.py \\
    --ckpt_dir ../Wan2.1-T2V-1.3B --output-dir vbench_eval/videos \\
    --start-idx 0 --end-idx 17
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Wan repo root = parent of vbench_eval
WAN_ROOT = str(Path(__file__).resolve().parent.parent)

MODES = [
    {"name": "wan_baseline", "teacache_thresh": 0.0},
    {"name": "wan_fixed_0.1", "teacache_thresh": 0.1},
    {"name": "wan_fixed_0.2", "teacache_thresh": 0.2},
    {"name": "wan_adaptive", "teacache_thresh": 0.1, "teacache_thresh_high": 0.2,
     "teacache_adaptive_first_steps": 10, "teacache_adaptive_last_steps": 16},
]


def load_generation_log(log_path):
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            return json.load(f)
    return {"runs": [], "completed_keys": []}


def save_generation_log(log_path, log_data):
    tmp = log_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(log_data, f, indent=2)
    os.replace(tmp, log_path)


def main():
    parser = argparse.ArgumentParser(description="Wan 2.1 VBench batch video generation")
    parser.add_argument("--prompts-file", type=str,
                        default=os.path.join(WAN_ROOT, "vbench_eval", "prompts_subset.json"),
                        help="Path to VBench prompts JSON")
    parser.add_argument("--output-dir", type=str, default=os.path.join(WAN_ROOT, "vbench_eval", "videos"),
                        help="Base output directory for videos (mode subdirs created here)")
    parser.add_argument("--ckpt_dir", type=str, required=True,
                        help="Path to Wan2.1 T2V 1.3B checkpoint (e.g. ../Wan2.1-T2V-1.3B)")
    parser.add_argument("--generation-seed", type=int, default=0)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--modes", type=str, default="all",
                        help="Comma-separated modes or 'all'")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--size", type=str, default="832*480")
    parser.add_argument("--sample-steps", type=int, default=50)
    parser.add_argument("--offload-model", type=str, default="True",
                        help="Set to True on smaller GPUs")
    parser.add_argument("--t5-cpu", action="store_true", help="Keep T5 on CPU (small GPUs)")
    args = parser.parse_args()

    with open(args.prompts_file, "r") as f:
        all_prompts = json.load(f)

    end_idx = len(all_prompts) if args.end_idx == -1 else args.end_idx
    start_idx = args.start_idx
    prompts = all_prompts[start_idx:end_idx]

    if args.modes == "all":
        modes = MODES
    else:
        names = [m.strip() for m in args.modes.split(",")]
        modes = [m for m in MODES if m["name"] in names]
        if not modes:
            print(f"ERROR: No valid modes in '{args.modes}'. Available: {[m['name'] for m in MODES]}")
            sys.exit(1)

    seed = args.generation_seed
    output_dir = os.path.abspath(args.output_dir)
    total_videos = len(prompts) * len(modes)

    # With --start-idx/--end-idx split (e.g. 0-17 and 17-33), each process writes
    # different prompt indices → different filenames → no file overwrites.
    print("=" * 70)
    print("Wan 2.1 VBench Batch Video Generation")
    print("=" * 70)
    print(f"Prompts file:   {args.prompts_file}")
    print(f"Prompt range:   [{start_idx}, {end_idx}) = {len(prompts)} prompts (no overlap with other GPU)")
    print(f"Seed:            {seed}")
    print(f"Modes:           {[m['name'] for m in modes]}")
    print(f"Total videos:    {total_videos}")
    print(f"Output dir:      {output_dir}")
    print(f"ckpt_dir:       {args.ckpt_dir}")
    print("  (Delta TEMNI plots disabled for batch; only .mp4 saved.)")
    print("=" * 70)

    if args.dry_run:
        print("\n[DRY RUN] Would generate:")
        for entry in prompts:
            prompt = entry["prompt_en"]
            for mode in modes:
                fn = f"{prompt}-{seed}.mp4"
                path = os.path.join(output_dir, mode["name"], fn)
                st = "EXISTS" if os.path.exists(path) else "NEW"
                print(f"  [{st}] {mode['name']}/{fn}")
        existing = sum(1 for e in prompts for m in modes
                       if os.path.exists(os.path.join(output_dir, m["name"], f"{e['prompt_en']}-{seed}.mp4")))
        print(f"\nAlready exist: {existing}, to generate: {total_videos - existing}")
        return

    log_filename = f"generation_log_{start_idx}-{end_idx}.json"
    log_path = os.path.join(output_dir, log_filename)
    gen_log = load_generation_log(log_path)
    print(f"Log file: {log_path}\n")

    teacache_script = os.path.join(WAN_ROOT, "teacache_generate.py")
    if not os.path.exists(teacache_script):
        print(f"ERROR: {teacache_script} not found. Run from Wan2.1 repo root.")
        sys.exit(1)

    completed = 0
    skipped = 0
    failed = 0
    total_gen_time = 0.0

    for prompt_idx, entry in enumerate(prompts):
        prompt = entry["prompt_en"]
        global_idx = start_idx + prompt_idx

        for mode_idx, mode in enumerate(modes):
            mode_name = mode["name"]
            thresh = mode["teacache_thresh"]
            video_filename = f"{prompt}-{seed}.mp4"
            video_dir = os.path.join(output_dir, mode_name)
            video_path = os.path.join(video_dir, video_filename)
            run_num = prompt_idx * len(modes) + mode_idx + 1
            run_key = f"{mode_name}|{prompt}|{seed}"

            if os.path.exists(video_path):
                print(f"[{run_num}/{total_videos}] SKIP (exists): {mode_name} | {prompt[:50]}...")
                skipped += 1
                continue

            print(f"[{run_num}/{total_videos}] Generating: {mode_name} | {prompt[:50]}...")
            os.makedirs(video_dir, exist_ok=True)

            cmd = [
                sys.executable,
                teacache_script,
                "--task", "t2v-1.3B",
                "--size", args.size,
                "--ckpt_dir", args.ckpt_dir,
                "--teacache_thresh", str(thresh),
                "--prompt", prompt,
                "--save_file", video_path,
                "--base_seed", str(seed),
                "--sample_steps", str(args.sample_steps),
                "--offload_model", args.offload_model,
                "--no_delta_temni_plot",  # no plot/txt per video in batch
            ]
            if mode.get("teacache_thresh_high") is not None:
                cmd.extend(["--teacache_thresh_high", str(mode["teacache_thresh_high"])])
                cmd.extend(["--teacache_adaptive_first_steps", str(mode.get("teacache_adaptive_first_steps", 10))])
                cmd.extend(["--teacache_adaptive_last_steps", str(mode.get("teacache_adaptive_last_steps", 16))])
            if args.t5_cpu:
                cmd.append("--t5_cpu")

            try:
                t0 = time.time()
                result = subprocess.run(cmd, cwd=WAN_ROOT, capture_output=True, text=True, timeout=3600)
                gen_time = time.time() - t0

                if result.returncode != 0:
                    print(f"  FAILED (exit {result.returncode}): {result.stderr[:500] if result.stderr else result.stdout[-500:]}")
                    failed += 1
                    gen_log["runs"].append({
                        "prompt": prompt, "seed": seed, "mode": mode_name,
                        "error": result.stderr or result.stdout, "prompt_index": global_idx,
                        "timestamp": datetime.now().isoformat(),
                    })
                    save_generation_log(log_path, gen_log)
                    continue

                # One line per video: index, mode, prompt snippet, end-to-end time, path
                prompt_short = (prompt[:48] + "..") if len(prompt) > 50 else prompt
                print(f"  [{run_num}/{total_videos}] {mode_name:18} | {gen_time:6.1f}s | {prompt_short}")
                print(f"      -> {video_path}")
                completed += 1
                total_gen_time += gen_time
                gen_log["runs"].append({
                    "prompt": prompt, "seed": seed, "mode": mode_name,
                    "time_seconds": round(gen_time, 2), "video_path": video_path,
                    "timestamp": datetime.now().isoformat(), "prompt_index": global_idx,
                })
                gen_log["completed_keys"].append(run_key)
                save_generation_log(log_path, gen_log)

            except subprocess.TimeoutExpired:
                print("  FAILED: timeout (3600s)")
                failed += 1
                gen_log["runs"].append({
                    "prompt": prompt, "seed": seed, "mode": mode_name,
                    "error": "timeout", "prompt_index": global_idx,
                    "timestamp": datetime.now().isoformat(),
                })
                save_generation_log(log_path, gen_log)
            except Exception as e:
                print(f"  FAILED: {e}")
                failed += 1
                gen_log["runs"].append({
                    "prompt": prompt, "seed": seed, "mode": mode_name,
                    "error": str(e), "prompt_index": global_idx,
                    "timestamp": datetime.now().isoformat(),
                })
                save_generation_log(log_path, gen_log)

    # Structured final summary
    print("\n" + "=" * 70)
    print("BATCH GENERATION COMPLETE")
    print("=" * 70)
    print(f"  Completed:     {completed}")
    print(f"  Skipped:      {skipped} (already existed)")
    print(f"  Failed:       {failed}")
    if completed:
        print(f"  Total time:    {total_gen_time:.1f}s  ({total_gen_time/3600:.1f}h)")
        print(f"  Avg per video: {total_gen_time/completed:.1f}s")
    print(f"  Log file:     {log_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
