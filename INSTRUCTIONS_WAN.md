# Wan2.1 + TeaCache — Run Instructions

This document describes how to run **Wan2.1** with the adaptive TeaCache integration: generation and **delta TEMNI** plotting (no-cache baseline).

**Important:** Wan2.1 uses a **different Python environment** than HunyuanVideo (different `transformers`, `torch`, `flash_attn`, etc.). Use a **separate container or venv** for Wan2.1; do **not** run Wan2.1 inside the HunyuanVideo `hv` container.

---

## Getting started (Wan2.1 environment)

You need a environment with **CUDA**, **PyTorch ≥ 2.4**, and Wan2.1’s dependencies. Two options:

### Option A: Docker container for Wan2.1 (recommended)

Use the **devel** PyTorch image (includes nvcc so flash_attn can build). From the **host**:

1. **Create and enter the container** (mount path so Wan2.1 and checkpoints are visible):

   ```bash
   docker run -it --gpus all --name wan21 \
     -v /nfs/oagrawal/wan:/workspace/wan \
     pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel bash
   ```

   Adjust `-v` if your Wan2.1 path differs (e.g. `-v /path/to/wan:/workspace/wan`). Inside the container, Wan2.1 will be at `/workspace/wan/Wan2.1`.

2. **Inside the container** — install in this order so flash_attn’s build sees torch:

   ```bash
   cd /workspace/wan/Wan2.1

   pip install --upgrade pip
   pip install "torch>=2.4.0" "torchvision>=0.19.0"
   pip install packaging wheel
   pip install flash-attn --no-build-isolation
   pip install -r requirements.txt
   pip install matplotlib
   ```

3. **Checkpoints:** put Wan2.1 checkpoints under the same mount as the repo. If the checkpoint is a **sibling** of `Wan2.1` (e.g. `wan/Wan2.1-T2V-1.3B` next to `wan/Wan2.1`), from `cd /workspace/wan/Wan2.1` use `--ckpt_dir ../Wan2.1-T2V-1.3B`. If the checkpoint is **inside** the repo, use `./Wan2.1-T2V-1.3B`.

**Enter the container later:**

```bash
docker start -ai wan21
```

**Extra shell:**

```bash
docker exec -it wan21 bash
```

### Option B: Native / venv on the host

If you already have CUDA and PyTorch ≥ 2.4 on the host:

```bash
cd /path/to/Wan2.1
python3 -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install --upgrade pip
# Install torch first so flash_attn's build can import it
pip install "torch>=2.4.0" "torchvision>=0.19.0"
pip install -r requirements.txt
pip install matplotlib
```

If `flash_attn` still fails when installing from `requirements.txt`, install it after torch:

```bash
pip install flash-attn --no-build-isolation
```

Then run all `teacache_generate.py` commands from this venv.

**Installing flash_attn:** The build imports `torch` to detect CUDA. If you see `ModuleNotFoundError: No module named 'torch'`, install torch first, then flash_attn:

```bash
pip install "torch>=2.4.0" "torchvision>=0.19.0"
pip install packaging wheel
pip install flash-attn --no-build-isolation
```

Then install the rest: `pip install -r requirements.txt` (pip will skip already-installed packages).

**"FlashAttention is only supported on CUDA 11.7 and above"** — The build uses **nvcc**. If the host’s `nvcc -V` is &lt; 11.7 (e.g. 11.5), use **Option A (Docker)** with the **devel** image: `pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel` (includes nvcc 12.1). On a cluster with modules you can instead `module load cuda/12` then `pip install flash-attn --no-build-isolation`.

---

## About the HunyuanVideo (hv) container

The **hunyuanvideo** image (`hv` = `docker start -ai hunyuanvideo`) is for **HunyuanVideo** only (see `HunyuanVideo/vbench_eval/INSTRUCTIONS.md` and `HunyuanVideo/research.txt`). Its Python stack (e.g. transformers version) does not match Wan2.1. Keep HunyuanVideo and Wan2.1 in **separate** containers or environments.

---

## Quick start: delta TEMNI experiment (1.3B, no cache)

1. **Create and enter the Wan2.1 container** (from host; adjust mount path if needed):

   ```bash
   docker run -it --gpus all --name wan21 \
     -v /nfs/oagrawal/wan:/workspace/wan \
     pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel bash
   ```

2. **Inside the container** — install dependencies in this order:

   ```bash
   cd /workspace/wan/Wan2.1
   pip install --upgrade pip
   pip install "torch>=2.4.0" "torchvision>=0.19.0"
   pip install packaging wheel
   pip install flash-attn --no-build-isolation
   pip install -r requirements.txt
   pip install matplotlib
   ```

3. **Run the baseline (delta TEMNI plot, no TeaCache):**

   ```bash
   mkdir -p wan_results
   python teacache_generate.py \
     --task t2v-1.3B \
     --size 832*480 \
     --ckpt_dir ../Wan2.1-T2V-1.3B \
     --teacache_thresh 0 \
     --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
     --save_file ./wan_results/cats_boxing_1.3B_baseline.mp4
   ```

   `--ckpt_dir ../Wan2.1-T2V-1.3B` assumes the checkpoint is a **sibling** of the repo (e.g. `wan/Wan2.1-T2V-1.3B` when the repo is `wan/Wan2.1`). Adjust if yours is elsewhere. Outputs: video plus `cats_boxing_1.3B_baseline_delta_TEMNI_plot.png` and `..._delta_TEMNI.txt` in `wan_results/`.

---

## Reference prompt

All example commands below use this prompt (same as in HunyuanVideo experiments):

```text
Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.
```

---

## 1. Delta TEMNI plot (no TeaCache)

To see **delta TEMNI** over steps **without** any caching (baseline behavior):

- Use **`--teacache_thresh 0`**.
- Every forward is computed; the script still records delta TEMNI and writes a plot + a `.txt` of values.

**T2V-14B (720P)** — single GPU:

```bash
cd /workspace/wan/Wan2.1

python teacache_generate.py \
  --task t2v-14B \
  --size 1280*720 \
  --ckpt_dir ../Wan2.1-T2V-14B \
  --teacache_thresh 0 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file ./wan_results/cats_boxing_baseline.mp4
```

**T2V-1.3B (480P)** — use `--offload_model True --t5_cpu` only on smaller GPUs (e.g. RTX 4090 24GB); not needed on A100 80GB:

```bash
python teacache_generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --teacache_thresh 0 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file ./wan_results/cats_boxing_1.3B_baseline.mp4
```

**Outputs when `--teacache_thresh 0`:**

- Video: path given by `--save_file`.
- **Delta TEMNI plot**: `{save_file_basename}_delta_TEMNI_plot.png` (same directory as the video).
- **Delta TEMNI values**: `{save_file_basename}_delta_TEMNI.txt` (one value per line).

Use the plot to see whether delta TEMNI is higher at the beginning, middle, or end of sampling; that guides adaptive threshold choices.

---

## 2. TeaCache enabled (with caching)

- **`--teacache_thresh 0.1`**: ~2× speedup, stricter (better quality, less skip).
- **`--teacache_thresh 0.2`**: ~3× speedup, more aggressive.

Same prompt as above.

**T2V-14B with TeaCache (e.g. 0.2):**

```bash
python teacache_generate.py \
  --task t2v-14B \
  --size 1280*720 \
  --ckpt_dir ../Wan2.1-T2V-14B \
  --teacache_thresh 0.2 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file ./wan_results/cats_boxing_teacache_0.2.mp4
```

**T2V-1.3B with TeaCache** (add `--offload_model True --t5_cpu` only if needed on smaller GPUs):

```bash
python teacache_generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --teacache_thresh 0.2 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file ./wan_results/cats_boxing_1.3B_teacache_0.2.mp4
```

Delta TEMNI is **not** recorded/plotted when `teacache_thresh > 0` (only when `teacache_thresh == 0`).

---

## 3. Optional arguments

| Argument | Description | Example |
|----------|-------------|--------|
| `--ckpt_dir` | Path to Wan2.1 checkpoint dir (from repo: sibling `../Wan2.1-T2V-1.3B` or in-repo `./Wan2.1-T2V-1.3B`) | `../Wan2.1-T2V-1.3B` |
| `--save_file` | Output video (and plot base name) | `./wan_results/out.mp4` |
| `--teacache_thresh` | `0` = no cache + plot delta TEMNI; `0.1` / `0.2` = TeaCache | `0`, `0.1`, `0.2` |
| `--sample_steps` | Diffusion steps (default 50 T2V, 40 I2V) | `50` |
| `--base_seed` | Random seed | `42` |
| `--offload_model` | Offload to CPU to reduce VRAM | `True` |
| `--t5_cpu` | Keep T5 on CPU | use with 1.3B on small GPUs |

If `--save_file` is omitted, a default name is generated (task, size, prompt snippet, timestamp).

---

## 4. VBench Evaluation Pipeline (same setup as HunyuanVideo)

This section replicates the experiment in **HunyuanVideo/vbench_eval/INSTRUCTIONS.md** using the **Wan 2.1 1.3B T2V** model: 33 VBench prompts, 4 modes (baseline + 3 TeaCache), VBench 16-dimension evaluation, fidelity metrics (PSNR/SSIM/LPIPS), and **3 output CSVs** (VBench table, fidelity table, summary table).

**Environments:** Video generation runs in the **Wan2.1** container (`wan21`). VBench evaluation and fidelity metrics run in the **HunyuanVideo** container (they use `vbench` and `transformers==4.33.2`). The compare step (3 CSVs) runs with no GPU and can be run from the host or either container as long as paths are visible.

**Paths below:** `WAN_ROOT` = Wan2.1 repo root (e.g. `/workspace/wan/Wan2.1` in Docker or `/nfs/oagrawal/wan/Wan2.1` on host). `HV_ROOT` = HunyuanVideo repo root (e.g. `/nfs/oagrawal/HunyuanVideo`). Adjust if your layout differs.

### 4.1 Modes and folder structure

| Mode | Description |
|------|-------------|
| `wan_baseline` | No caching (teacache_thresh 0) |
| `wan_fixed_0.1` | Fixed threshold 0.1 |
| `wan_fixed_0.2` | Fixed threshold 0.2 |
| `wan_adaptive` | First 10 and last 16 forward steps (as on delta TEMNI plot) use low threshold (0.1); middle uses high (0.2) |

```
Wan2.1/vbench_eval/
├── prompts_subset.json       # 33 VBench prompts (same as HunyuanVideo)
├── batch_generate_wan.py     # Batch video generation for Wan
├── videos/                   # Generated videos
│   ├── wan_baseline/         #   {prompt}-{seed}.mp4
│   ├── wan_fixed_0.1/
│   ├── wan_fixed_0.2/
│   └── wan_adaptive/
├── vbench_scores/             # Filled by Step 2 (run from HunyuanVideo)
│   ├── wan_baseline/
│   └── ...
├── fidelity_metrics/          # Filled by Step 3
├── all_comparison_results.json
├── vbench_scores_table.csv    # Table 1
├── fidelity_table.csv        # Table 2
└── summary_table.csv         # Table 3
```

### 4.2 Tmux (SSH-safe runs)

Long jobs (generation, VBench eval) should run inside **tmux** so they survive SSH disconnects.

- **Create session:** `tmux new -s <name>`
- **Reattach after disconnect:** `tmux attach -t <name>`
- **Extra shell in same container:** `docker exec -it wan21 bash` or `docker exec -it hunyuanvideo bash`

---

### Step 1: Generate videos (Wan2.1 container, tmux recommended)

Use the **Wan2.1** Docker container. Generation is slow (~132 videos = 33 prompts × 4 modes); split across 2 GPUs with `--start-idx` / `--end-idx`.

**Tmux + two terminals (recommended):**

```bash
# Terminal 1
tmux new -s wan_gen0
docker start -ai wan21
# then run GPU 0 command below

# Terminal 2 (new SSH or new terminal)
tmux new -s wan_gen1
docker exec -it wan21 bash
# then run GPU 1 command below
```

**Reattach after disconnect:**

```bash
tmux attach -t wan_gen0
tmux attach -t wan_gen1
```

**Dry run (no generation):**

```bash
cd /workspace/wan/Wan2.1
python3 vbench_eval/batch_generate_wan.py --ckpt_dir ../Wan2.1-T2V-1.3B --dry-run
```

**Single GPU (all 33 prompts):**

```bash
cd /workspace/wan/Wan2.1
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_wan.py \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --output-dir vbench_eval/videos
```

**Split across 2 GPUs (recommended):**

- **No race conditions:** GPU 0 generates prompts 0–16, GPU 1 generates 17–32. Each (prompt, mode) pair has a unique filename (`{prompt}-{seed}.mp4`), so the two processes never write the same file.
- **Per-video timing:** Each generated video prints one line with end-to-end time (e.g. `[1/68] wan_baseline | 123.4s | prompt...`) and the save path. The same timing is **saved to disk** in `vbench_eval/videos/generation_log_{start}-{end}.json` (each run has `time_seconds`, `video_path`, `prompt`, `mode`, `timestamp`). The final summary shows completed/skipped/failed counts and total/avg time. Step 4 (compare_results) uses these log files for the latency/speedup columns in the CSVs.
- **No plot output in batch:** Delta TEMNI plots are disabled during batch runs; only the `.mp4` file is saved per video.

**GPU 0 — Terminal 1** (prompts 0–16):

```bash
cd /workspace/wan/Wan2.1
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/batch_generate_wan.py \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --output-dir vbench_eval/videos \
  --start-idx 0 --end-idx 17
```

**GPU 1 — Terminal 2** (prompts 17–32):

```bash
cd /workspace/wan/Wan2.1
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/batch_generate_wan.py \
  --ckpt_dir ../Wan2.1-T2V-1.3B \
  --output-dir vbench_eval/videos \
  --start-idx 17 --end-idx 33
```

On smaller GPUs, add: `--offload_model True` and optionally `--t5_cpu`.

**Resume after Ctrl+C or restart:** Re-run the **exact same** command. The script skips any video for which the `.mp4` already exists; the generation log (`generation_log_{start}-{end}.json`) is only appended when a new video completes, so you do not lose or duplicate timing data. No shared in-memory state is used—everything is file-based, so interrupting and resuming is safe.

---

### Step 2: Run VBench evaluation (HunyuanVideo container, tmux recommended)

**Requires:** A HunyuanVideo environment/container that can see **both**:

- `HV_ROOT` (e.g. `/nfs/oagrawal/HunyuanVideo`)
- `WAN_VBENCH` (e.g. `/nfs/oagrawal/wan/Wan2.1/vbench_eval`)

If your existing `hunyuanvideo` container only mounts the HunyuanVideo repo to `/workspace`, it **will not** be able to see `/nfs/oagrawal/wan/...`. In that case, create a separate eval container that mounts `/nfs/oagrawal` explicitly.

#### 2.0 Recommended: create an eval container that can see `/nfs/oagrawal`

From the **host** (one-time):

```bash
docker run -it --gpus all --init --net=host --uts=host --ipc=host \
  --name hunyuanvideo_eval_wan --security-opt=seccomp=unconfined \
  --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged \
  -v /nfs/oagrawal:/nfs/oagrawal \
  hunyuanvideo/hunyuanvideo:cuda_11 bash
```

Re-enter later:

```bash
docker start -ai hunyuanvideo_eval_wan
```

Extra shell:

```bash
docker exec -it hunyuanvideo_eval_wan bash
```

Visibility check (inside container):

```bash
python3 - <<'PY'
import os
p="/nfs/oagrawal/wan/Wan2.1/vbench_eval/videos"
print("exists:", os.path.isdir(p), "path:", p)
PY
```

Switch transformers for VBench:

```bash
# Inside HunyuanVideo container (eval)
pip install transformers==4.33.2
```

VBench package must be importable (fixes `ModuleNotFoundError: No module named 'vbench'`):

```bash
cd $HV_ROOT
pip install -e ./VBench
python3 -c "from vbench import VBench; print('vbench import ok')"
```

OpenCV system dependency (fixes `ImportError: libGL.so.1: cannot open shared object file`):

```bash
apt-get update
apt-get install -y libgl1 libglib2.0-0
python3 -c "import cv2; print('cv2 ok', cv2.__version__)"
```

Detectron2 is required for these 4 dimensions: `multiple_objects`, `spatial_relationship`, `object_class`, `color`:

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

**Tmux:**

```bash
# Terminal 1
tmux new -s vbench0
docker start -ai hunyuanvideo_eval_wan
pip install transformers==4.33.2

# Terminal 2
tmux new -s vbench1
docker exec -it hunyuanvideo_eval_wan bash
pip install transformers==4.33.2
```

**Full run split across 2 GPUs** (replace `HV_ROOT` and `WAN_VBENCH` with your paths; e.g. `HV_ROOT=/path/to/HunyuanVideo`, `WAN_VBENCH=/path/to/wan/Wan2.1/vbench_eval`):

**GPU 0 — modes wan_baseline, wan_fixed_0.1:**

```bash
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $WAN_VBENCH/videos \
  --save-dir $WAN_VBENCH/vbench_scores \
  --full-info $WAN_VBENCH/prompts_subset.json \
  --modes wan_baseline,wan_fixed_0.1
```

**GPU 1 — modes wan_fixed_0.2, wan_adaptive:**

```bash
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=1 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $WAN_VBENCH/videos \
  --save-dir $WAN_VBENCH/vbench_scores \
  --full-info $WAN_VBENCH/prompts_subset.json \
  --modes wan_fixed_0.2,wan_adaptive
```

Example with absolute paths:

```bash
WAN_VBENCH=/nfs/oagrawal/wan/Wan2.1/vbench_eval
cd /nfs/oagrawal/HunyuanVideo
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_vbench_eval.py \
  --video-dir $WAN_VBENCH/videos --save-dir $WAN_VBENCH/vbench_scores \
  --full-info $WAN_VBENCH/prompts_subset.json \
  --modes wan_baseline,wan_fixed_0.1
```

**Note on warnings about missing `...-1.mp4` to `...-4.mp4`:** VBench expects up to 5 seeds per prompt. If you only generated seed index `0`, VBench prints warnings for missing seed indices `1..4`. This is normal; it evaluates using whatever seeds exist.

**Verify completion:** each mode should have `16/16` files named `*_eval_results.json`:

```bash
WAN_VBENCH=/nfs/oagrawal/wan/Wan2.1/vbench_eval
for m in wan_baseline wan_fixed_0.1 wan_fixed_0.2 wan_adaptive; do
  python3 - <<PY
import glob
m="$m"
base="$WAN_VBENCH/vbench_scores/"+m
print(m, len(glob.glob(base+"/*_eval_results.json")), "/16")
PY
done
```

**If you need to wipe and rerun VBench scores:** When running in Docker as `root`, files on NFS may be owned by `root` so deleting from the host can fail with `Permission denied`. Safest is to delete from inside the eval container:

```bash
rm -rf $WAN_VBENCH/vbench_scores
mkdir -p $WAN_VBENCH/vbench_scores
chmod -R a+rwX $WAN_VBENCH/vbench_scores
```

---

### Step 3: Run fidelity metrics (HunyuanVideo container, tmux recommended)

Compares cached modes to `wan_baseline` (PSNR / SSIM / LPIPS). Single GPU is enough.

```bash
# Inside HunyuanVideo container
pip install lpips
cd $HV_ROOT
CUDA_VISIBLE_DEVICES=0 python3 vbench_eval/run_fidelity_metrics.py \
  --video-dir $WAN_VBENCH/videos \
  --baseline wan_baseline \
  --modes wan_fixed_0.1,wan_fixed_0.2,wan_adaptive \
  --save-dir $WAN_VBENCH/fidelity_metrics
```

---

### Step 4: Compare results and get the 3 CSVs

Aggregates VBench scores, fidelity, and generation timing; prints the same **3 tables** as HunyuanVideo and writes **3 CSVs** in the same layout. No GPU; run from host or any env where `HV_ROOT` and Wan’s `vbench_eval` are visible.

```bash
cd $HV_ROOT
python3 vbench_eval/compare_results.py \
  --scores-dir $WAN_VBENCH/vbench_scores \
  --fidelity-dir $WAN_VBENCH/fidelity_metrics \
  --gen-log-dir $WAN_VBENCH/videos \
  --output-json $WAN_VBENCH/all_comparison_results.json \
  --modes wan_baseline,wan_fixed_0.1,wan_fixed_0.2,wan_adaptive
```

If you see `IndentationError` in `HunyuanVideo/vbench_eval/compare_results.py`, update/fix the file and rerun (the script should write `all_comparison_results.json` plus the three CSVs).

**Output files** (same as in HunyuanVideo INSTRUCTIONS.md):

| File | Description |
|------|-------------|
| `vbench_scores_table.csv` | VBench scores — all 16 dimensions per mode + Quality/Semantic/Total score + Latency |
| `fidelity_table.csv` | Fidelity — PSNR, SSIM, LPIPS, Latency per mode |
| `summary_table.csv` | Compact — Speedup, Latency, VBench Total, PSNR, SSIM, LPIPS |

All three are written under the directory of `--output-json` (e.g. `$WAN_VBENCH/vbench_scores_table.csv`, `$WAN_VBENCH/fidelity_table.csv`, `$WAN_VBENCH/summary_table.csv`).

**Tables printed to stdout:** Table 1 = VBench (quality + semantic dims + latency). Table 2 = Fidelity vs baseline. Table 3 = Compact summary (one line per mode).

---

### VBench pipeline quick reference

| Step | Where | Command / note |
|------|--------|-----------------|
| 1. Generate | Wan container, tmux | `batch_generate_wan.py` with `--ckpt_dir`, `--output-dir`, optional `--start-idx` / `--end-idx` |
| 2. VBench | HunyuanVideo container, tmux | `run_vbench_eval.py` with `--video-dir` / `--save-dir` / `--full-info` pointing to Wan’s `vbench_eval`, `--modes wan_*` |
| 3. Fidelity | HunyuanVideo container | `run_fidelity_metrics.py` with `--video-dir`, `--baseline wan_baseline`, `--save-dir` |
| 4. Compare | Any (no GPU) | `compare_results.py` with `--scores-dir`, `--fidelity-dir`, `--gen-log-dir`, `--output-json`, `--modes wan_baseline,wan_fixed_0.1,wan_fixed_0.2,wan_adaptive` |

---

## 5. Quick reference

- **Environment**: Use the Wan2.1 Docker container (`wan21`) or a dedicated venv — not the HunyuanVideo `hv` container.
- **Working dir**: From the Wan2.1 repo root (e.g. `cd /workspace/wan/Wan2.1` in Docker, or your local path).
- **Delta TEMNI (no cache)**: `--teacache_thresh 0`; plot and `.txt` are written next to the video.
- **TeaCache**: `--teacache_thresh 0.1` or `0.2`.

All example commands use the prompt:

**"Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."**
