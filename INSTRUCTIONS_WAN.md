# Wan2.1 + TeaCache — Run Instructions

This document describes how to run **Wan2.1** with the adaptive TeaCache integration: generation, **delta TEMNI** plotting (no-cache baseline), and container setup using the same **hv** (HunyuanVideo) container as in the HunyuanVideo vbench eval.

---

## Container setup (same as HunyuanVideo)

Use the **hunyuanvideo** Docker image and the same workflow as in `HunyuanVideo/vbench_eval/INSTRUCTIONS.md` and `HunyuanVideo/research.txt`.

### Create container (first time)

From a directory that will be your workspace (e.g. a parent folder containing both HunyuanVideo and `wan/Wan2.1` so both are available under the mount):

```bash
# Create and enter the container (mounts current directory at /workspace)
docker run -it --gpus all --init --net=host --uts=host --ipc=host \
  --name hunyuanvideo \
  --security-opt=seccomp=unconfined \
  --ulimit=stack=67108864 --ulimit=memlock=-1 --privileged \
  -v $(pwd):/workspace \
  hunyuanvideo/hunyuanvideo:cuda_11 bash
```

If you don't have the image:

```bash
docker pull hunyuanvideo/hunyuanvideo:cuda_11
```

### Enter existing container

```bash
# Start and attach (use only one terminal this way)
hv
```

Where **`hv`** is the alias:

```bash
alias hv='docker start -ai hunyuanvideo'
```

### Extra shells (e.g. second GPU or tmux)

You can only run `hv` once. For more shells, use:

```bash
docker exec -it hunyuanvideo bash
```

### Optional: tmux for long runs / SSH

So that runs survive SSH disconnects:

```bash
# Terminal 1
tmux new -s gpu0
hv
# run your command here

# Terminal 2 (new SSH connection)
tmux new -s gpu1
docker exec -it hunyuanvideo bash
# run your command here
```

Reattach after disconnect:

```bash
tmux attach -t gpu0
tmux attach -t gpu1
```

### Recreate container from scratch

```bash
docker rm -f hunyuanvideo
# then run the docker run ... command above again (or use hv-create if you defined it)
```

### Permissions

If you hit EACCES inside the container:

```bash
chmod -R 777 ./
```

---

## Wan2.1 setup inside the container

From inside the container (`hv` or `docker exec -it hunyuanvideo bash`):

1. **Go to Wan2.1** (path depends on your mount; adjust if your repo is elsewhere):

   ```bash
   cd /workspace/wan/Wan2.1
   # or e.g. cd /workspace/Wan2.1
   ```

2. **Install deps** (if not already done):

   ```bash
   pip install -r requirements.txt
   ```

3. **Checkpoints**: ensure Wan2.1 checkpoints are available (e.g. under `/workspace/` or the same mount). Example layout:

   - `/workspace/wan/Wan2.1/Wan2.1-T2V-14B`
   - `/workspace/wan/Wan2.1/Wan2.1-T2V-1.3B`

   Set `--ckpt_dir` to the correct path in the commands below.

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
  --ckpt_dir ./Wan2.1-T2V-14B \
  --teacache_thresh 0 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file ./wan_results/cats_boxing_baseline.mp4
```

**T2V-1.3B (480P)** — with offload to avoid OOM:

```bash
python teacache_generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --teacache_thresh 0 \
  --offload_model True \
  --t5_cpu \
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
  --ckpt_dir ./Wan2.1-T2V-14B \
  --teacache_thresh 0.2 \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file ./wan_results/cats_boxing_teacache_0.2.mp4
```

**T2V-1.3B with TeaCache + offload:**

```bash
python teacache_generate.py \
  --task t2v-1.3B \
  --size 832*480 \
  --ckpt_dir ./Wan2.1-T2V-1.3B \
  --teacache_thresh 0.2 \
  --offload_model True \
  --t5_cpu \
  --prompt "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage." \
  --save_file ./wan_results/cats_boxing_1.3B_teacache_0.2.mp4
```

Delta TEMNI is **not** recorded/plotted when `teacache_thresh > 0` (only when `teacache_thresh == 0`).

---

## 3. Optional arguments

| Argument | Description | Example |
|----------|-------------|--------|
| `--ckpt_dir` | Path to Wan2.1 checkpoint dir | `./Wan2.1-T2V-14B` |
| `--save_file` | Output video (and plot base name) | `./wan_results/out.mp4` |
| `--teacache_thresh` | `0` = no cache + plot delta TEMNI; `0.1` / `0.2` = TeaCache | `0`, `0.1`, `0.2` |
| `--sample_steps` | Diffusion steps (default 50 T2V, 40 I2V) | `50` |
| `--base_seed` | Random seed | `42` |
| `--offload_model` | Offload to CPU to reduce VRAM | `True` |
| `--t5_cpu` | Keep T5 on CPU | use with 1.3B on small GPUs |

If `--save_file` is omitted, a default name is generated (task, size, prompt snippet, timestamp).

---

## 4. Quick reference: running inside the hv container

- **Enter container**: `hv` (or `docker exec -it hunyuanvideo bash` for extra shells).
- **Wan2.1 dir**: e.g. `cd /workspace/wan/Wan2.1` (adjust to your mount).
- **Delta TEMNI (no cache)**: add `--teacache_thresh 0` and set `--save_file`; plot and `.txt` are written next to the video.
- **TeaCache**: use `--teacache_thresh 0.1` or `0.2` with the same prompt and `--save_file` as needed.

All commands in this file use the prompt:

**"Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."**
