#!/bin/bash
# GPU 2: wan_dc_adaptive_0.20_0.30 (prompts 17-24)
# LOW=0.20 for steps [0-6, 85-99]; HIGH=0.30 for steps [7-84]

cd /nfs/oagrawal/wan/Wan2.1

python3 dicache_results/batch_generate_wan_dicache.py \
  --prompts-file vbench_eval/prompts_subset.json \
  --output-dir dicache_results/videos \
  --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B \
  --sample-steps 50 \
  --generation-seed 0 \
  --ret-ratio 0.0 \
  --probe-depth 1 \
  --start-idx 17 \
  --end-idx 25 \
  --mode-name wan_dc_adaptive_0.20_0.30 \
  --delta-low 0.20 \
  --delta-high 0.30 \
  --stable-start 7 \
  --stable-end 85
