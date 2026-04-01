#!/bin/bash
# GPU0: baseline, fixed_0.10, fixed_0.20 on prompts 0-16
set -e
cd /nfs/oagrawal/wan/Wan2.1

echo "=== [GPU0] baseline (0-16) ==="
python3 dicache_results/batch_generate_wan_dicache.py   --prompts-file vbench_eval/prompts_subset.json   --output-dir dicache_results/videos   --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B   --sample-steps 50   --generation-seed 0   --ret-ratio 0.0   --probe-depth 1   --start-idx 0   --end-idx 17   --mode-name wan_dc_baseline   --delta 0.0

echo "=== [GPU0] fixed_0.10 (0-16) ==="
python3 dicache_results/batch_generate_wan_dicache.py   --prompts-file vbench_eval/prompts_subset.json   --output-dir dicache_results/videos   --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B   --sample-steps 50   --generation-seed 0   --ret-ratio 0.0   --probe-depth 1   --start-idx 0   --end-idx 17   --mode-name wan_dc_fixed_0.10   --delta 0.10

echo "=== [GPU0] fixed_0.20 (0-16) ==="
python3 dicache_results/batch_generate_wan_dicache.py   --prompts-file vbench_eval/prompts_subset.json   --output-dir dicache_results/videos   --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B   --sample-steps 50   --generation-seed 0   --ret-ratio 0.0   --probe-depth 1   --start-idx 0   --end-idx 17   --mode-name wan_dc_fixed_0.20   --delta 0.20

echo "=== [GPU0] Done ==="
