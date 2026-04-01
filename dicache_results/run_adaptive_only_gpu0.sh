#!/bin/bash
# GPU 0: adaptive_0.20_0.30 then adaptive_0.10_0.30, prompts 0-16
set -e
cd /nfs/oagrawal/wan/Wan2.1

echo "=== [GPU0] wan_dc_adaptive_0.20_0.30 (prompts 0-16) ==="
python3 dicache_results/batch_generate_wan_dicache.py   --prompts-file vbench_eval/prompts_subset.json   --output-dir dicache_results/videos   --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B   --sample-steps 50   --generation-seed 0   --ret-ratio 0.0   --probe-depth 1   --start-idx 0   --end-idx 17   --mode-name wan_dc_adaptive_0.20_0.30   --delta-low 0.20   --delta-high 0.30   --stable-start 7   --stable-end 85

echo "=== [GPU0] wan_dc_adaptive_0.10_0.30 (prompts 0-16) ==="
python3 dicache_results/batch_generate_wan_dicache.py   --prompts-file vbench_eval/prompts_subset.json   --output-dir dicache_results/videos   --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B   --sample-steps 50   --generation-seed 0   --ret-ratio 0.0   --probe-depth 1   --start-idx 0   --end-idx 17   --mode-name wan_dc_adaptive_0.10_0.30   --delta-low 0.10   --delta-high 0.30   --stable-start 7   --stable-end 85

echo "=== [GPU0] Done ==="
