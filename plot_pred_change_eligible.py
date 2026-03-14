#!/usr/bin/env python3
"""Plot pred_change for eligible steps (zoomed: steps 5-48, excludes first 5 ret_steps and last cutoff)."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use pred_change_all.txt (all steps) then slice to steps 5-48 (EasyCache eligible with ret_steps=5)
DIR = Path("/nfs/oagrawal/wan/Wan2.1/easycache_results_wan/baseline_profile_2026-03-11-15:41:29_seed12345_Two_anthropomorphic_cats_in_comfy_boxing_gear_and_")
all_data = np.loadtxt(DIR / "pred_change_all.txt")
# pred_change_all starts at cond step 2 (first pred_change needs 2 prior evens)
pred_change_all_start = 2
# Eligible for fixed/adaptive: steps 5-48 (exclude first 5 ret_steps and last cutoff step)
eligible_start = 5
eligible_end = 48
start_idx = eligible_start - pred_change_all_start
end_idx = eligible_end - pred_change_all_start + 1
data = all_data[start_idx:end_idx]
n = len(data)

x = np.arange(eligible_start, eligible_start + n)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, data, "o-", color="#2563eb", linewidth=1.5, markersize=4)
ax.set_xlabel("Condition step", fontsize=11)
ax.set_ylabel("pred_change", fontsize=11)
ax.set_title("pred_change (zoomed): eligible steps 5–48 for fixed/adaptive EasyCache\n"
             "First 5 ret_steps + last step = mandatory full diffusion (excluded)", fontsize=12)
ax.grid(True, alpha=0.3)
ax.set_xlim(eligible_start - 0.5, eligible_end + 0.5)
fig.tight_layout()
out = Path("/nfs/oagrawal/wan/Wan2.1/pred_change_eligible_zoomed.png")
fig.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
plt.close()
