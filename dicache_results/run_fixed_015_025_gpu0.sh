#!/bin/bash
# GPU 0: fixed_0.15 then fixed_0.25, prompts 0-16
set -e
cd /nfs/oagrawal/wan/Wan2.1

echo "=== [GPU0] wan_dc_fixed_0.15 (prompts 0-16) ==="
python3 dicache_results/batch_generate_wan_dicache.py   --prompts-file vbench_eval/prompts_subset.json   --output-dir dicache_results/videos   --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B   --sample-steps 50   --generation-seed 0   --ret-ratio 0.0   --probe-depth 1   --start-idx 0   --end-idx 17   --mode-name wan_dc_fixed_0.15   --delta 0.15

echo "=== [GPU0] wan_dc_fixed_0.25 (prompts 0-16) ==="
python3 dicache_results/batch_generate_wan_dicache.py   --prompts-file vbench_eval/prompts_subset.json   --output-dir dicache_results/videos   --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B   --sample-steps 50   --generation-seed 0   --ret-ratio 0.0   --probe-depth 1   --start-idx 0   --end-idx 17   --mode-name wan_dc_fixed_0.25   --delta 0.25

echo "=== [GPU0] Done ==="
