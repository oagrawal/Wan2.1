#!/usr/bin/env bash
# Usage: run_fixed_sweep_one_gpu.sh <GPU_ID> <start-idx> <end-idx>
# Runs wan_dc_baseline + wan_dc_fixed_{0.10,0.20,0.30,0.40} for the prompt slice.
set -euo pipefail
GPU_ID="${1:?need GPU id}"
START="${2:?need start-idx}"
END="${3:?need end-idx}"
LOG="/nfs/oagrawal/wan/Wan2.1/dicache_results/logs/gpu${GPU_ID}_fixed_sweep.log"

cd /nfs/oagrawal/wan/Wan2.1
export CUDA_VISIBLE_DEVICES="${GPU_ID}"

exec > >(tee -a "${LOG}") 2>&1

echo "======================================================================"
echo "Wan2.1 DiCache fixed sweep | GPU=${GPU_ID} | prompts [${START}, ${END})"
echo "======================================================================"

MODES="wan_dc_baseline,wan_dc_fixed_0.10,wan_dc_fixed_0.20,wan_dc_fixed_0.30,wan_dc_fixed_0.40"

python3 dicache_results/batch_generate_wan_dicache.py \
  --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B \
  --prompts-file vbench_eval/prompts_subset.json \
  --output-dir dicache_results/videos \
  --generation-seed 0 \
  --start-idx "${START}" \
  --end-idx "${END}" \
  --sample-steps 50 \
  --modes "${MODES}"

echo "======================================================================"
echo "Done GPU=${GPU_ID} prompts [${START}, ${END})"
echo "======================================================================"
