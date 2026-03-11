# Wan2.1 EasyCache — Run Instructions

Run all commands from the **Wan2.1 repo root** inside the **`wan21` Docker container**.
See `INSTRUCTIONS_WAN.md` for container setup details (same container works here).

---

## Key difference: Wan uses paired (CFG) model calls

With Classifier-Free Guidance, every denoising step calls the DiT model **twice**:
once for the conditional branch (even model-call index) and once for the unconditional
branch (odd model-call index). EasyCache makes the skip decision on **even** steps only
and mirrors it to the paired odd step.

With `--sample_steps 50` there are **100 total model calls**:
- `ret_steps = 10` → first 10 denoising steps (20 model calls) always compute.
- `cutoff_steps` → last 1 denoising step (2 model calls) always compute.

---

## New files

| File | Purpose |
|------|---------|
| `easycache_sample_video_wan.py` | Single-video script: baseline profiling + fixed + adaptive EasyCache |
| `vbench_eval_easycache/batch_generate_wan.py` | In-process batch generation (loads model once) |
| `vbench_eval_easycache/compare_results_wan.py` | Aggregate results → CSV |

---

## Part A: Baseline profiling (understanding k_t and pred_change shape)

### What to look for

Running the model with `--easycache-mode baseline` records two metrics on each
**condition step** (even model call):

- **k_t** = `||v_t − v_{t−2}|| / ||x_t − x_{t−2}||` — transformation rate
  (how sensitive the model output is to input changes at this step).
- **pred_change** = `k_{t−2} · (||x_t − x_{t−2}|| / ||v_{t−2}||)` — predicted
  relative output change (what EasyCache accumulates to decide whether to skip).

A high pred_change means this step is "dangerous to skip". The plot shape tells you
where to use a low (conservative) vs high (aggressive) threshold.

### Step 1: Set up the Wan container

```bash
# On host (one-time)
docker run -it --gpus all --name wan21 \
  -v /nfs/oagrawal/wan:/workspace/wan \
  pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel bash

# Re-enter later
docker start -ai wan21
```

Inside the container, install dependencies:
```bash
cd /workspace/wan/Wan2.1
pip install --upgrade pip
pip install "torch>=2.4.0" "torchvision>=0.19.0"
pip install packaging wheel
pip install flash-attn --no-build-isolation
pip install -r requirements.txt
pip install matplotlib portalocker
```

### Step 2: Run the baseline profiling

```bash
cd /workspace/wan/Wan2.1
mkdir -p easycache_results_wan

python3 easycache_sample_video_wan.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --easycache-mode baseline \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 12345 \
  --save_dir ./easycache_results_wan
```

**Outputs** (in `easycache_results_wan/baseline_profile_<timestamp>_<prompt>/`):
- `video.mp4`
- `k_t_plot.png` + `k_t.txt`
- `pred_change_plot.png` + `pred_change.txt`
- `diagnostic_info.txt` (timing, k_t stats)

Use these plots to decide adaptive threshold boundaries, analogous to
the U-shaped delta-TEMNI plot used for TeaCache.

---

## Part B: Single-video EasyCache runs (test before full VBench)

### Mode 1 — Baseline (no caching):
```bash
python3 easycache_sample_video_wan.py \
  --task t2v-1.3B --size 832*480 \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --easycache-mode baseline \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 12345 --save_dir ./easycache_results_wan
```

### Mode 2 — Fixed low threshold (0.025):
```bash
python3 easycache_sample_video_wan.py \
  --task t2v-1.3B --size 832*480 \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --easycache-mode easycache --easycache-thresh 0.025 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 12345 --save_dir ./easycache_results_wan
```

### Mode 3 — Fixed high threshold (0.050):
```bash
python3 easycache_sample_video_wan.py \
  --task t2v-1.3B --size 832*480 \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --easycache-mode easycache --easycache-thresh 0.050 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 12345 --save_dir ./easycache_results_wan
```

### Mode 4 — Adaptive (0.025 at start/end, 0.050 in middle):
```bash
python3 easycache_sample_video_wan.py \
  --task t2v-1.3B --size 832*480 \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --easycache-mode adaptive \
  --easycache-thresh-low 0.025 --easycache-thresh-high 0.050 \
  --easycache-first-steps 8 --easycache-last-steps 6 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --base_seed 12345 --save_dir ./easycache_results_wan
```

**Threshold tuning** — adjust these after viewing the baseline pred_change plot:
- If pred_change is consistently small in the middle, raise `--easycache-thresh-high`.
- If pred_change is large early/late, keep `--easycache-thresh-low` conservative.

---

## Part C: Full VBench evaluation (4 modes × 33 prompts)

Uses `vbench_eval/prompts_subset.json` (same 33 prompts as HunyuanVideo).

### Folder layout

```
Wan2.1/vbench_eval_easycache/
├── videos/
│   ├── wan_ec_baseline/       # {prompt}-{seed}.mp4
│   ├── wan_ec_fixed_0.025/
│   ├── wan_ec_fixed_0.050/
│   └── wan_ec_adaptive/
├── vbench_scores/             # written by Step 2
├── fidelity_metrics/          # written by Step 3
├── all_comparison_results.json
└── all_comparison_results.csv
```

### 4 EasyCache modes

| Mode name | Description |
|-----------|-------------|
| `wan_ec_baseline`    | No caching (ground truth) |
| `wan_ec_fixed_0.025` | Fixed threshold 0.025 |
| `wan_ec_fixed_0.050` | Fixed threshold 0.050 |
| `wan_ec_adaptive`    | Low 0.025 (first 8 + last 6 condition steps), high 0.050 (middle) |

---

### Step 1: Generate videos (Wan container, split across GPUs)

The batch script loads the model **once** and loops over all (prompt, mode) pairs —
much faster than the subprocess-per-video approach used in TeaCache.

**Important:** The instance-attribute bug that affected HunyuanVideo is already fixed
in `batch_generate_wan.py`. All EasyCache state is reset on the model **instance**
(not the class) between runs so modes are correctly isolated.

**Quick smoke test first** (1 prompt, 2 modes, ~3-4 min):
```bash
cd /workspace/wan/Wan2.1
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval_easycache/batch_generate_wan.py \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --output-dir /tmp/wan_ec_test \
  --modes wan_ec_baseline,wan_ec_fixed_0.025 \
  --start-idx 0 --end-idx 1

# Verify different MD5s (confirms modes are actually different)
md5sum /tmp/wan_ec_test/wan_ec_baseline/*.mp4 \
       /tmp/wan_ec_test/wan_ec_fixed_0.025/*.mp4
```

**Full run — split across 4 GPUs** (4 terminals, disjoint prompt ranges):

Create 4 tmux sessions:
```bash
# Terminal 1
tmux new -s wan_ec0
docker start -ai wan21
cd /workspace/wan/Wan2.1

# Terminal 2
tmux new -s wan_ec1
docker exec -it wan21 bash
cd /workspace/wan/Wan2.1

# Terminal 3
tmux new -s wan_ec2
docker exec -it wan21 bash
cd /workspace/wan/Wan2.1

# Terminal 4
tmux new -s wan_ec3
docker exec -it wan21 bash
cd /workspace/wan/Wan2.1
```

**Terminal 1 (GPU 0, prompts 0–8):**
```bash
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval_easycache/batch_generate_wan.py \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --start-idx 0 --end-idx 9
```

**Terminal 2 (GPU 1, prompts 9–17):**
```bash
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval_easycache/batch_generate_wan.py \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --start-idx 9 --end-idx 18
```

**Terminal 3 (GPU 2, prompts 18–26):**
```bash
CUDA_VISIBLE_DEVICES=2 python3 vbench_eval_easycache/batch_generate_wan.py \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --start-idx 18 --end-idx 27
```

**Terminal 4 (GPU 3, prompts 27–32):**
```bash
CUDA_VISIBLE_DEVICES=3 python3 vbench_eval_easycache/batch_generate_wan.py \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --start-idx 27 --end-idx 33
```

On smaller GPUs, add `--offload-model --t5-cpu`.

**Resume:** Re-run the same command; existing `.mp4` files are skipped.

**Reattach after disconnect:**
```bash
tmux attach -t wan_ec0   # etc.
```

**Verify completion:**
```bash
for mode in wan_ec_baseline wan_ec_fixed_0.025 wan_ec_fixed_0.050 wan_ec_adaptive; do
  count=$(ls vbench_eval_easycache/videos/$mode/ 2>/dev/null | wc -l)
  echo "$mode: $count videos"
done
```

**Check timing (sanity check — cached modes should be faster than baseline):**
```bash
python3 - << 'PY'
import json, glob
logs = glob.glob("vbench_eval_easycache/videos/generation_log_*.json")
by_mode = {}
for lf in logs:
    for r in json.load(open(lf))["runs"]:
        by_mode.setdefault(r["mode"], []).append(r["time_seconds"])
for mode, times in sorted(by_mode.items()):
    print(f"{mode}: {len(times)} videos, avg {sum(times)/len(times):.0f}s")
PY
```

---

### Step 2: Run VBench evaluation

VBench evaluation reuses the HunyuanVideo VBench scripts (same container).
The `run_vbench_eval.py` in `HunyuanVideo/vbench_eval_easycache/` accepts
`--video-dir` and `--save-dir` arguments that can point to any location.

**Prerequisites (inside HunyuanVideo container or hunyuanvideo_eval_wan container):**
```bash
apt-get update && apt-get install -y libgl1
pip install transformers==4.33.2
pip install vbench
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

If the `hunyuanvideo` container doesn't mount `/nfs/oagrawal/wan`, create a new eval container:
```bash
docker run -it --gpus all --init --net=host --uts=host --ipc=host \
  --name hv_eval_wan --security-opt=seccomp=unconfined \
  --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged \
  -v /nfs/oagrawal:/nfs/oagrawal \
  hunyuanvideo/hunyuanvideo:cuda_11 bash
```

**Run VBench — split across 4 GPUs (one mode per GPU):**

Set paths:
```bash
HV_ROOT=/nfs/oagrawal/HunyuanVideo
WAN_EC=/nfs/oagrawal/wan/Wan2.1/vbench_eval_easycache
```

```bash
# GPU 0 — baseline
CUDA_VISIBLE_DEVICES=0 python3 $HV_ROOT/vbench_eval_easycache/run_vbench_eval.py \
  --video-dir $WAN_EC/videos \
  --save-dir $WAN_EC/vbench_scores \
  --full-info $HV_ROOT/vbench_eval/prompts_subset.json \
  --modes wan_ec_baseline

# GPU 1 — fixed 0.025
CUDA_VISIBLE_DEVICES=1 python3 $HV_ROOT/vbench_eval_easycache/run_vbench_eval.py \
  --video-dir $WAN_EC/videos \
  --save-dir $WAN_EC/vbench_scores \
  --full-info $HV_ROOT/vbench_eval/prompts_subset.json \
  --modes wan_ec_fixed_0.025

# GPU 2 — fixed 0.050
CUDA_VISIBLE_DEVICES=2 python3 $HV_ROOT/vbench_eval_easycache/run_vbench_eval.py \
  --video-dir $WAN_EC/videos \
  --save-dir $WAN_EC/vbench_scores \
  --full-info $HV_ROOT/vbench_eval/prompts_subset.json \
  --modes wan_ec_fixed_0.050

# GPU 3 — adaptive
CUDA_VISIBLE_DEVICES=3 python3 $HV_ROOT/vbench_eval_easycache/run_vbench_eval.py \
  --video-dir $WAN_EC/videos \
  --save-dir $WAN_EC/vbench_scores \
  --full-info $HV_ROOT/vbench_eval/prompts_subset.json \
  --modes wan_ec_adaptive
```

**Verify (11 eval_results per mode):**
```bash
python3 - << 'PY'
import glob
WAN_EC = "/nfs/oagrawal/wan/Wan2.1/vbench_eval_easycache"
for mode in ["wan_ec_baseline","wan_ec_fixed_0.025","wan_ec_fixed_0.050","wan_ec_adaptive"]:
    n = len(glob.glob(f"{WAN_EC}/vbench_scores/{mode}/*_eval_results.json"))
    print(f"{mode}: {n}/11 eval_results.json")
PY
```

Note: `dynamic_degree` requires `libgl1` (installed above). Missing `*-1.mp4 … *-4.mp4`
warnings are normal — VBench expects 5 seeds per prompt; we only generate seed 0.

---

### Step 3: Run fidelity metrics (PSNR / SSIM / LPIPS)

Uses the HunyuanVideo fidelity script (it accepts `--video-dir` / `--baseline` flags):
```bash
# Inside HunyuanVideo container (with lpips installed)
pip install lpips

HV_ROOT=/nfs/oagrawal/HunyuanVideo
WAN_EC=/nfs/oagrawal/wan/Wan2.1/vbench_eval_easycache

CUDA_VISIBLE_DEVICES=0 python3 $HV_ROOT/vbench_eval_easycache/run_fidelity_metrics.py \
  --video-dir $WAN_EC/videos \
  --baseline wan_ec_baseline \
  --modes wan_ec_fixed_0.025,wan_ec_fixed_0.050,wan_ec_adaptive \
  --save-dir $WAN_EC/fidelity_metrics
```

---

### Step 4: Compare results and write CSV

No GPU needed. Run from the Wan2.1 repo root (host or any container):
```bash
cd /nfs/oagrawal/wan/Wan2.1
python3 vbench_eval_easycache/compare_results_wan.py
```

Outputs:
- `vbench_eval_easycache/all_comparison_results.json`
- `vbench_eval_easycache/all_comparison_results.csv` — columns: mode, speedup, latency, vbench, psnr, ssim, lpips

---

## Quick reference

| Step | Container | Command |
|------|-----------|---------|
| Baseline profile | `wan21` | `python3 easycache_sample_video_wan.py --easycache-mode baseline …` |
| Batch generation | `wan21`, tmux | `batch_generate_wan.py --ckpt_dir … --start-idx … --end-idx …` |
| VBench eval | `hunyuanvideo` / `hv_eval_wan` | `run_vbench_eval.py --video-dir $WAN_EC/videos …` |
| Fidelity | `hunyuanvideo` / `hv_eval_wan` | `run_fidelity_metrics.py --video-dir … --baseline wan_ec_baseline …` |
| Compare | Host or any container | `compare_results_wan.py` |

## Transformers versions

| Task | Version |
|------|---------|
| Video generation (Wan) | ≥ 2.4.0 (PyTorch; no specific `transformers` pin) |
| VBench evaluation | `transformers==4.33.2` |
| Fidelity / compare | Either |

## Troubleshooting

**`ModuleNotFoundError: No module named 'easycache_sample_video_wan'`**
Run from the Wan2.1 repo root, or add it to `PYTHONPATH`:
```bash
PYTHONPATH=/workspace/wan/Wan2.1 python3 vbench_eval_easycache/batch_generate_wan.py …
```

**`libGL.so.1: cannot open shared object file` (dynamic_degree)**
```bash
apt-get update && apt-get install -y libgl1
```

**All cached modes produce identical videos (speedup 1.00x)**
This is the instance-attribute bug — it's already fixed in `batch_generate_wan.py`
via `configure_model()`. Verify by running the smoke test (different MD5s = working).

**Permission denied when deleting VBench scores on NFS**
Files are owned by `root` from inside Docker. Delete from inside the container:
```bash
rm -rf /nfs/oagrawal/wan/Wan2.1/vbench_eval_easycache/vbench_scores
```
