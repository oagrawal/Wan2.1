#!/bin/bash
set -e
HV=/nfs/oagrawal/HunyuanVideo
VID=/nfs/oagrawal/wan/Wan2.1/dicache_results/videos
OUT=/nfs/oagrawal/wan/Wan2.1/dicache_results/vbench_scores
PROMPTS=$HV/vbench_eval/prompts_subset.json
PY=$HV/vbench_eval_easycache/run_vbench_eval.py

echo "=== VBench wan_dc_fixed_0.20 (GPU1) ==="
CUDA_VISIBLE_DEVICES=1 python3 "$PY" --video-dir "$VID" --save-dir "$OUT" --full-info "$PROMPTS" --modes wan_dc_fixed_0.20

echo "=== GPU1 VBench done ==="
