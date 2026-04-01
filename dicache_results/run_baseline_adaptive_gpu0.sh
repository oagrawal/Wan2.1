#!/bin/bash
# GPU 0: wan_dc_baseline (prompts 0-8)

cd /nfs/oagrawal/wan/Wan2.1

python3 dicache_results/batch_generate_wan_dicache.py \
  --prompts-file vbench_eval/prompts_subset.json \
  --output-dir dicache_results/videos \
  --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B \
  --sample-steps 50 \
  --generation-seed 0 \
  --ret-ratio 0.0 \
  --probe-depth 1 \
  --start-idx 0 \
  --end-idx 9 \
  --mode-name wan_dc_baseline \
  --delta 0.0
