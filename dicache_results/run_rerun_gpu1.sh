#!/bin/bash
# GPU 1: fixed_0.10 then adaptive_0.10_0.30, prompts 17-32
set -e
cd /nfs/oagrawal/wan/Wan2.1

echo "=== [GPU1] wan_dc_fixed_0.10 (prompts 17-32) ==="
python3 dicache_results/batch_generate_wan_dicache.py   --prompts-file vbench_eval/prompts_subset.json   --output-dir dicache_results/videos   --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B   --sample-steps 50   --generation-seed 0   --ret-ratio 0.0   --probe-depth 1   --start-idx 17   --end-idx 33   --mode-name wan_dc_fixed_0.10   --delta 0.10

echo "=== [GPU1] wan_dc_adaptive_0.10_0.30 (prompts 17-32) ==="
python3 dicache_results/batch_generate_wan_dicache.py   --prompts-file vbench_eval/prompts_subset.json   --output-dir dicache_results/videos   --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B   --sample-steps 50   --generation-seed 0   --ret-ratio 0.0   --probe-depth 1   --start-idx 17   --end-idx 33   --mode-name wan_dc_adaptive_0.10_0.30   --delta-low 0.10   --delta-high 0.30   --stable-start 7   --stable-end 85

echo "=== [GPU1] Done ==="
