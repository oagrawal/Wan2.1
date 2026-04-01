"""
Calibration run for Wan2.1 DiCache.

Runs a single prompt with NO skipping (rel_l1_thresh=0, ret_ratio=0.0) and
records per-forward-call delta_y (relative L1 probe change) and accumulated
error for both CFG slots (cnt%2 == 0 and 1).

Outputs:
  --calibration-out  <path>.json   per-step data
  A matching <path>.png            plot (delta_y + accumulated per CFG slot)

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 dicache/run_wan_dicache_calibrate.py \
    --ckpt_dir /nfs/oagrawal/wan/Wan2.1-T2V-1.3B \
    --prompt "A dog runs across a green field." \
    --calibration-out dicache_results/calibration/wan_dicache_probe_curve.json \
    --save_file dicache_results/calibration/calib_video.mp4
"""
import argparse
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

import torch
import torch.cuda.amp as amp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

WAN_ROOT = str(Path(__file__).resolve().parent.parent)
if WAN_ROOT not in sys.path:
    sys.path.insert(0, WAN_ROOT)

import wan
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS
from wan.modules.model import sinusoidal_embedding_1d
from wan.utils.utils import cache_video
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Calibration-instrumented forward
# ---------------------------------------------------------------------------

def dicache_calibrate_forward(
    self,
    x,
    t,
    context,
    seq_len,
    clip_fea=None,
    y=None,
):
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack(
        [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    assert seq_lens.max() <= seq_len
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    context_lens = None
    context = self.text_embedding(
        torch.stack([
            torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
            for u in context
        ]))

    if clip_fea is not None:
        context_clip = self.img_emb(clip_fea)
        context = torch.concat([context_clip, context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=context_lens,
    )

    slot = self.cnt % 2
    ori_x = x.clone()

    # Probe always runs (ret_ratio=0.0 for calibration)
    if self.cnt >= int(self.num_steps * self.ret_ratio):
        test_x = x.clone()
        for blk in self.blocks[: self.probe_depth]:
            test_x = blk(test_x, **kwargs)

        # Guard for the very first eligible call per slot (previous states = None)
        if self.previous_internal_states[slot] is not None:
            delta_y = (
                (test_x - self.previous_internal_states[slot]).abs().mean()
                / (self.previous_internal_states[slot].abs().mean() + 1e-8)
            ).item()
            self.accumulated_rel_l1_distance[slot] += delta_y
        else:
            delta_y = 0.0

        # Record calibration data
        self._calib_cnt.append(self.cnt)
        self._calib_slot.append(slot)
        self._calib_delta_y.append(delta_y)
        self._calib_accumulated.append(self.accumulated_rel_l1_distance[slot])

        # Full run (threshold = 0 so we never skip)
        self.previous_internal_states[slot] = test_x
        # resume from probe output
        x = test_x
        for blk in self.blocks[self.probe_depth:]:
            x = blk(x, **kwargs)
        residual_x = x - ori_x
        self.residual_cache[slot] = residual_x
        self.previous_input[slot] = ori_x
        self.previous_output[slot] = x
    else:
        # Retention prefix: full run without probe recording
        for ind, blk in enumerate(self.blocks):
            x = blk(x, **kwargs)
            if ind == self.probe_depth - 1:
                self.previous_internal_states[slot] = x
        residual_x = x - ori_x
        self.residual_cache[slot] = residual_x
        self.previous_input[slot] = ori_x
        self.previous_output[slot] = x

    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)

    self.cnt += 1
    if self.cnt >= self.num_steps:
        self.cnt = 0
        self.accumulated_rel_l1_distance = [0.0, 0.0]
        self.residual_cache = [None, None]
        self.previous_internal_states = [None, None]
        self.previous_input = [None, None]
        self.previous_output = [None, None]

    return [u.float() for u in x]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_probe_curve(calib_data, save_path):
    cnt = calib_data["cnt"]
    delta_y = calib_data["delta_y"]
    accumulated = calib_data["accumulated"]
    slot = calib_data["slot"]

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Wan2.1 DiCache — Probe Calibration Curve", fontsize=14)

    for s, label, color in [(0, "CFG slot 0 (uncond)", "steelblue"),
                             (1, "CFG slot 1 (cond)",  "darkorange")]:
        idxs = [i for i, sl in enumerate(slot) if sl == s]
        xs = [cnt[i] for i in idxs]
        dy = [delta_y[i] for i in idxs]
        ac = [accumulated[i] for i in idxs]
        if not xs:
            continue
        axes[0].plot(xs, dy, marker='o', markersize=3, linewidth=1,
                     label=label, color=color)
        axes[1].plot(xs, ac, marker='o', markersize=3, linewidth=1,
                     label=label, color=color)

    axes[0].set_ylabel("delta_y (relative L1 per step)")
    axes[0].set_title("Per-step probe change (delta_y)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_ylabel("Accumulated error (E_t)")
    axes[1].set_title("Accumulated error (resets never fire with thresh=0)")
    axes[1].set_xlabel("Forward call index (cnt, 0..num_steps-1)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"Plot saved to {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Wan2.1 DiCache calibration run")
    p.add_argument("--ckpt_dir", type=str,
                   default="/nfs/oagrawal/wan/Wan2.1-T2V-1.3B")
    p.add_argument("--task", type=str, default="t2v-1.3B")
    p.add_argument("--size", type=str, default="832*480")
    p.add_argument("--frame_num", type=int, default=81)
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--prompt", type=str,
                   default="Two anthropomorphic cats in comfy boxing gear "
                           "and bright gloves fight intensely on a spotlighted stage.")
    p.add_argument("--probe_depth", type=int, default=1,
                   help="Number of shallow blocks to use as probe")
    p.add_argument("--ret_ratio", type=float, default=0.0,
                   help="Retention prefix ratio (0.0 = probe from step 0)")
    p.add_argument("--save_file", type=str,
                   default="dicache_results/calibration/calib_video.mp4")
    p.add_argument("--calibration-out", type=str,
                   default="dicache_results/calibration/wan_dicache_probe_curve.json",
                   help="Path to write calibration JSON (and matching .png)")
    return p.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    logging.info(f"Calibration run — probe_depth={args.probe_depth}, "
                 f"ret_ratio={args.ret_ratio}, sample_steps={args.sample_steps}")
    logging.info(f"Prompt: {args.prompt}")

    cfg = WAN_CONFIGS[args.task]

    logging.info("Loading Wan2.1 model...")
    wan_t2v = wan.WanT2V(
        config=cfg,
        checkpoint_dir=args.ckpt_dir,
        device_id=0,
        rank=0,
        t5_cpu=False,
    )
    logging.info("Model loaded.")

    num_steps = args.sample_steps * 2  # CFG doubles forward calls

    # Set forward + state (instance attributes to avoid cross-run contamination)
    wan_t2v.model.__class__.forward = dicache_calibrate_forward
    wan_t2v.model.cnt = 0
    wan_t2v.model.probe_depth = args.probe_depth
    wan_t2v.model.num_steps = num_steps
    wan_t2v.model.ret_ratio = args.ret_ratio
    wan_t2v.model.rel_l1_thresh = 0.0  # never skip
    wan_t2v.model.accumulated_rel_l1_distance = [0.0, 0.0]
    wan_t2v.model.residual_cache = [None, None]
    wan_t2v.model.previous_internal_states = [None, None]
    wan_t2v.model.previous_input = [None, None]
    wan_t2v.model.previous_output = [None, None]

    # Calibration buffers
    wan_t2v.model._calib_cnt = []
    wan_t2v.model._calib_slot = []
    wan_t2v.model._calib_delta_y = []
    wan_t2v.model._calib_accumulated = []

    logging.info("Generating calibration video (no skipping)...")
    video = wan_t2v.generate(
        args.prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=args.sample_steps,
        guide_scale=5.0,
        seed=args.base_seed,
        offload_model=True,
    )

    # Save video
    os.makedirs(os.path.dirname(os.path.abspath(args.save_file)), exist_ok=True)
    cache_video(
        tensor=video[None],
        save_file=args.save_file,
        fps=cfg.sample_fps,
        nrow=1,
        normalize=True,
        value_range=(-1, 1),
    )
    logging.info(f"Video saved to {args.save_file}")

    # Save calibration data
    calib_data = {
        "prompt": args.prompt,
        "sample_steps": args.sample_steps,
        "num_steps": num_steps,
        "probe_depth": args.probe_depth,
        "ret_ratio": args.ret_ratio,
        "timestamp": datetime.now().isoformat(),
        "cnt": wan_t2v.model._calib_cnt,
        "slot": wan_t2v.model._calib_slot,
        "delta_y": wan_t2v.model._calib_delta_y,
        "accumulated": wan_t2v.model._calib_accumulated,
    }

    calib_json = args.calibration_out
    os.makedirs(os.path.dirname(os.path.abspath(calib_json)), exist_ok=True)
    with open(calib_json, "w") as f:
        json.dump(calib_data, f, indent=2)
    logging.info(f"Calibration JSON saved to {calib_json}")

    calib_png = calib_json.replace(".json", ".png")
    plot_probe_curve(calib_data, calib_png)

    # Print quick summary
    dy = calib_data["delta_y"]
    logging.info(f"Recorded {len(dy)} probe steps")
    if dy:
        logging.info(f"delta_y — min={min(dy):.4f}  max={max(dy):.4f}  "
                     f"mean={sum(dy)/len(dy):.4f}")


if __name__ == "__main__":
    main()
