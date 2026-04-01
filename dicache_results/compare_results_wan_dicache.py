#!/usr/bin/env python3
"""
Aggregate VBench scores + timing for Wan2.1 DiCache experiments.

Flexible version: accepts --modes and --baseline CLI args so any set of
DiCache modes can be compared without editing the script.

Usage (from Wan2.1 repo root):
  python3 dicache_results/compare_results_wan_dicache.py \
    --scores-dir  dicache_results/vbench_scores \
    --gen-log-dir dicache_results/videos \
    --modes wan_dc_baseline,wan_dc_fixed_0.05,wan_dc_fixed_0.10,wan_dc_fixed_0.15,wan_dc_fixed_0.20 \
    --baseline wan_dc_baseline \
    --output-csv  dicache_results/results/fixed_sweep_comparison.csv \
    --output-json dicache_results/results/fixed_sweep_comparison.json
"""

import argparse
import csv
import glob
import json
import os

# VBench weighting/normalization (identical to compare_results_wan.py)
QUALITY_LIST = [
    "subject consistency", "background consistency", "temporal flickering",
    "motion smoothness", "aesthetic quality", "imaging quality", "dynamic degree",
]
SEMANTIC_LIST = [
    "object class", "multiple objects", "human action", "color",
    "spatial relationship", "scene", "appearance style", "temporal style",
    "overall consistency",
]
QUALITY_WEIGHT = 4
SEMANTIC_WEIGHT = 1

NORMALIZE_DIC = {
    "subject consistency":    {"Min": 0.1462, "Max": 1.0},
    "background consistency": {"Min": 0.2615, "Max": 1.0},
    "temporal flickering":    {"Min": 0.6293, "Max": 1.0},
    "motion smoothness":      {"Min": 0.706,  "Max": 0.9975},
    "dynamic degree":         {"Min": 0.0,    "Max": 1.0},
    "aesthetic quality":      {"Min": 0.0,    "Max": 1.0},
    "imaging quality":        {"Min": 0.0,    "Max": 1.0},
    "object class":           {"Min": 0.0,    "Max": 1.0},
    "multiple objects":       {"Min": 0.0,    "Max": 1.0},
    "human action":           {"Min": 0.0,    "Max": 1.0},
    "color":                  {"Min": 0.0,    "Max": 1.0},
    "spatial relationship":   {"Min": 0.0,    "Max": 1.0},
    "scene":                  {"Min": 0.0,    "Max": 0.8222},
    "appearance style":       {"Min": 0.0009, "Max": 0.2855},
    "temporal style":         {"Min": 0.0,    "Max": 0.364},
    "overall consistency":    {"Min": 0.0,    "Max": 0.364},
}
DIM_WEIGHT = {d: 1 for d in QUALITY_LIST + SEMANTIC_LIST}
DIM_WEIGHT["dynamic degree"] = 0.5


def load_vbench_scores(score_dir):
    scores = {}
    if not os.path.exists(score_dir):
        return scores
    for fname in os.listdir(score_dir):
        if not fname.endswith("_eval_results.json"):
            continue
        with open(os.path.join(score_dir, fname)) as f:
            d = json.load(f)
        for k, v in d.items():
            scores[k] = v[0] if isinstance(v, list) else v
    return scores


def compute_vbench_aggregate(raw):
    scaled = {}
    for k, v in raw.items():
        dim = k.replace("_", " ")
        if dim in NORMALIZE_DIC:
            n = NORMALIZE_DIC[dim]
            s = (float(v) - n["Min"]) / (n["Max"] - n["Min"])
            scaled[dim] = s * DIM_WEIGHT.get(dim, 1)
    q = [scaled[d] for d in QUALITY_LIST if d in scaled]
    s = [scaled[d] for d in SEMANTIC_LIST if d in scaled]
    qs = sum(q) / sum(DIM_WEIGHT[d] for d in QUALITY_LIST if d in scaled) if q else None
    ss = sum(s) / sum(DIM_WEIGHT[d] for d in SEMANTIC_LIST if d in scaled) if s else None
    total = (qs * QUALITY_WEIGHT + ss * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT) \
        if qs is not None and ss is not None else None
    return {"quality_score": qs, "semantic_score": ss, "total_score": total}


def load_timing(log_dir):
    """
    Reads generation_log_*.json files written by batch_generate_wan_dicache.py.
    Returns {mode_name: {"avg_time": float, "num_videos": int}}.
    """
    timing = {}
    for p in glob.glob(os.path.join(log_dir, "generation_log_*.json")):
        with open(p) as f:
            data = json.load(f)
        for run in data.get("runs", []):
            if "time_seconds" not in run:
                continue
            mode = run["mode"]
            timing.setdefault(mode, []).append(run["time_seconds"])
    return {mode: {"avg_time": sum(t) / len(t), "num_videos": len(t)}
            for mode, t in timing.items()}


def main():
    p = argparse.ArgumentParser(
        description="Aggregate Wan2.1 DiCache VBench + timing results"
    )
    p.add_argument("--scores-dir", default="dicache_results/vbench_scores",
                   help="Directory containing {mode}/..._eval_results.json files")
    p.add_argument("--gen-log-dir", default="dicache_results/videos",
                   help="Directory with generation_log_*.json timing files")
    p.add_argument("--fidelity-dir", default="dicache_results/fidelity_metrics",
                   help="Directory with fidelity results (optional)")
    p.add_argument("--modes", default=None,
                   help="Comma-separated mode names to include. "
                        "Defaults to all subdirs found in --scores-dir.")
    p.add_argument("--baseline", default="wan_dc_baseline",
                   help="Mode name to use as speedup denominator")
    p.add_argument("--output-json", default="dicache_results/results/comparison.json")
    p.add_argument("--output-csv",  default="dicache_results/results/comparison.csv")
    args = p.parse_args()

    # Resolve mode list
    if args.modes:
        mode_list = [m.strip() for m in args.modes.split(",")]
    else:
        if os.path.isdir(args.scores_dir):
            mode_list = sorted(os.listdir(args.scores_dir))
        else:
            print(f"ERROR: --scores-dir {args.scores_dir!r} does not exist and --modes not provided.")
            raise SystemExit(1)

    timing = load_timing(args.gen_log_dir)
    baseline_time = timing.get(args.baseline, {}).get("avg_time")

    # Fidelity (optional)
    fidelity = {}
    fid_path = os.path.join(args.fidelity_dir, "all_fidelity_results.json")
    if os.path.exists(fid_path):
        with open(fid_path) as f:
            fidelity = json.load(f)

    print("=" * 88)
    print(f"Wan2.1 DiCache — Evaluation Results (baseline={args.baseline})")
    print("=" * 88)
    print(f"{'Mode':<32} {'Speedup':>8} {'Latency':>9} {'VBench':>10} "
          f"{'PSNR':>8} {'SSIM':>7} {'LPIPS':>7}")
    print("-" * 88)

    rows = []
    for mode in mode_list:
        raw  = load_vbench_scores(os.path.join(args.scores_dir, mode))
        agg  = compute_vbench_aggregate(raw)
        t    = timing.get(mode, {})
        fid  = fidelity.get(mode, {})
        speedup = baseline_time / t["avg_time"] if baseline_time and t.get("avg_time") else None

        vbench_str = f"{agg['total_score'] * 100:.4f}%" if agg["total_score"] else "—"
        row = {
            "mode":    mode,
            "speedup": f"{speedup:.2f}x" if speedup else "—",
            "latency": f"{t['avg_time']:.0f}s" if t.get("avg_time") else "—",
            "vbench":  vbench_str,
            "psnr":    f"{fid['psnr']['mean']:.2f}"  if fid and "psnr"  in fid else "—",
            "ssim":    f"{fid['ssim']['mean']:.4f}"  if fid and "ssim"  in fid else "—",
            "lpips":   f"{fid['lpips']['mean']:.4f}" if fid and "lpips" in fid else "—",
        }
        rows.append(row)
        print(f"{row['mode']:<32} {row['speedup']:>8} {row['latency']:>9} "
              f"{row['vbench']:>10} {row['psnr']:>8} {row['ssim']:>7} {row['lpips']:>7}")

    print("=" * 88)

    out_dir = os.path.dirname(os.path.abspath(args.output_json))
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump({"baseline": args.baseline, "modes": mode_list,
                   "rows": rows, "timing": timing, "fidelity": fidelity}, f, indent=2)
    print(f"\nSaved JSON: {args.output_json}")

    fieldnames = ["mode", "speedup", "latency", "vbench", "psnr", "ssim", "lpips"]
    with open(args.output_csv, "w", newline="") as cf:
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"Saved CSV:  {args.output_csv}")


if __name__ == "__main__":
    main()
