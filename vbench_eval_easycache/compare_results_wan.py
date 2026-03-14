#!/usr/bin/env python3
"""
Aggregate VBench scores + fidelity + timing for Wan2.1 EasyCache experiment.
Prints a summary table and writes all_comparison_results.csv.

Usage (from Wan2.1 repo root, no GPU needed):
  python3 vbench_eval_easycache/compare_results_wan.py
"""

import argparse
import csv
import glob
import json
import os

# VBench weighting/normalization (same as HunyuanVideo compare_results.py)
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

ALL_MODES = [
    "wan_ec_baseline",
    "wan_ec_fixed_0.020",
    "wan_ec_fixed_0.025",
    "wan_ec_fixed_0.040",
    "wan_ec_fixed_0.050",
    "wan_ec_adaptive",
    "wan_ec_adaptive_16_020040",
]

MODE_LABELS = {
    "wan_ec_baseline":            "Wan baseline",
    "wan_ec_fixed_0.020":         "Wan EasyCache 0.020",
    "wan_ec_fixed_0.025":         "Wan EasyCache 0.025",
    "wan_ec_fixed_0.040":         "Wan EasyCache 0.040",
    "wan_ec_fixed_0.050":         "Wan EasyCache 0.050",
    "wan_ec_adaptive":            "Wan EasyCache adaptive",
    "wan_ec_adaptive_16_020040":  "Wan EasyCache adaptive 16 (0.02/0.04)",
}


def load_vbench_scores(score_dir):
    scores = {}
    if not os.path.exists(score_dir):
        return scores
    for f in os.listdir(score_dir):
        if not f.endswith("_eval_results.json"):
            continue
        with open(os.path.join(score_dir, f)) as fp:
            d = json.load(fp)
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
    qs = sum(q) / len(q) if q else None
    ss = sum(s) / len(s) if s else None
    total = (qs * QUALITY_WEIGHT + ss * SEMANTIC_WEIGHT) / (QUALITY_WEIGHT + SEMANTIC_WEIGHT) \
        if qs is not None and ss is not None else None
    return {"quality_score": qs, "semantic_score": ss, "total_score": total}


def load_timing(log_dir):
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


def load_fidelity(fidelity_dir):
    p = os.path.join(fidelity_dir, "all_fidelity_results.json")
    if os.path.exists(p):
        with open(p) as f:
            return json.load(f)
    result = {}
    for fp in glob.glob(os.path.join(fidelity_dir, "*_vs_wan_ec_baseline.json")):
        with open(fp) as f:
            d = json.load(f)
        mode = d.get("mode")
        if mode:
            result[mode] = d
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores-dir",  default="vbench_eval_easycache/vbench_scores")
    p.add_argument("--fidelity-dir", default="vbench_eval_easycache/fidelity_metrics")
    p.add_argument("--gen-log-dir",  default="vbench_eval_easycache/videos")
    p.add_argument("--output-json",  default="vbench_eval_easycache/all_comparison_results.json")
    p.add_argument("--output-csv",   default="vbench_eval_easycache/all_comparison_results.csv")
    args = p.parse_args()

    timing   = load_timing(args.gen_log_dir)
    fidelity = load_fidelity(args.fidelity_dir)
    baseline_time = timing.get("wan_ec_baseline", {}).get("avg_time")

    print("=" * 80)
    print("Wan2.1 EasyCache Evaluation Results")
    print("=" * 80)
    print(f"{'Mode':<24} {'Speedup':>8} {'Latency':>8} {'VBench%':>10} "
          f"{'PSNR':>8} {'SSIM':>7} {'LPIPS':>7}")
    print("-" * 80)

    rows = []
    for mode in ALL_MODES:
        raw    = load_vbench_scores(os.path.join(args.scores_dir, mode))
        agg    = compute_vbench_aggregate(raw)
        t      = timing.get(mode, {})
        fid    = fidelity.get(mode, {})
        speedup = baseline_time / t["avg_time"] if baseline_time and t else None

        row = {
            "mode":    MODE_LABELS.get(mode, mode),
            "speedup": f"{speedup:.2f}x" if speedup else "—",
            "latency": f"{t['avg_time']:.0f}s"         if t else "—",
            "vbench":  f"{agg['total_score'] * 100:.2f}%" if agg["total_score"] else "—",
            "psnr":    f"{fid['psnr']['mean']:.2f}"     if fid and "psnr"  in fid else "—",
            "ssim":    f"{fid['ssim']['mean']:.4f}"     if fid and "ssim"  in fid else "—",
            "lpips":   f"{fid['lpips']['mean']:.4f}"    if fid and "lpips" in fid else "—",
        }
        rows.append(row)
        print(f"{row['mode']:<24} {row['speedup']:>8} {row['latency']:>8} "
              f"{row['vbench']:>10} {row['psnr']:>8} {row['ssim']:>7} {row['lpips']:>7}")

    print("=" * 80)

    out_dir = os.path.dirname(args.output_json) or "."
    os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w") as f:
        json.dump({"modes": ALL_MODES, "rows": rows,
                   "timing": timing, "fidelity": fidelity}, f, indent=2)
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
