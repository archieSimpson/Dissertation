

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from sheep_sim.core.utils import circadian_factor

OUTPUT_DIR = Path("outputs/circadian")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

T            = 1920
REST_THRESH  = 0.28
PHASE_SHIFT  = -0.25
AMPLITUDE    = 0.80

steps  = np.arange(0, T + 1)
values = np.array([circadian_factor(int(s), T, PHASE_SHIFT, AMPLITUDE) for s in steps])
below  = values < REST_THRESH

fig, ax = plt.subplots(figsize=(11, 5), dpi=180)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")


in_rest = False
rest_start = 0
for i, b in enumerate(below):
    if b and not in_rest:
        rest_start = steps[i]; in_rest = True
    elif not b and in_rest:
        ax.axvspan(rest_start, steps[i], alpha=0.09, color="#534AB7", label="_nolegend_")
        in_rest = False
if in_rest:
    ax.axvspan(rest_start, steps[-1], alpha=0.09, color="#534AB7", label="_nolegend_")


ax.axhline(REST_THRESH, color="#534AB7", linewidth=1.2,
           linestyle="--", zorder=3, label=f"Rest threshold ({REST_THRESH})")


ax.plot(steps, values, color="#0F6E56", linewidth=2,
        label="Circadian activity c(t)", zorder=4)



ax.annotate(f"Dawn start\nc(0) = {values[0]:.2f}",
    xy=(0, values[0]), xytext=(60, values[0] + 0.12),
    fontsize=8, color="#854F0B", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#854F0B", lw=0.8))


morn_trough_idx = int(np.argmin(values[100:500])) + 100
ax.annotate(f"Morning trough\nc = {values[morn_trough_idx]:.2f}",
    xy=(steps[morn_trough_idx], values[morn_trough_idx]),
    xytext=(steps[morn_trough_idx] + 60, values[morn_trough_idx] + 0.18),
    fontsize=8, color="#444441",
    arrowprops=dict(arrowstyle="->", color="#444441", lw=0.8))


peak1_idx = int(np.argmax(values[400:650])) + 400
ax.annotate(f"First peak\nc = {values[peak1_idx]:.2f}",
    xy=(steps[peak1_idx], values[peak1_idx]),
    xytext=(steps[peak1_idx] - 180, values[peak1_idx] + 0.06),
    fontsize=8, color="#854F0B", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#854F0B", lw=0.8))


peak2_idx = int(np.argmax(values[550:700])) + 550
ax.annotate(f"Dusk peak (max)\nc = {values[peak2_idx]:.2f}",
    xy=(steps[peak2_idx], values[peak2_idx]),
    xytext=(steps[peak2_idx] + 50, values[peak2_idx] - 0.14),
    fontsize=8, color="#993C1D", fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="#993C1D", lw=0.8))


aft_trough_idx = int(np.argmin(values[700:1100])) + 700
ax.annotate(f"Afternoon trough\nc = {values[aft_trough_idx]:.2f}",
    xy=(steps[aft_trough_idx], values[aft_trough_idx]),
    xytext=(steps[aft_trough_idx] - 220, values[aft_trough_idx] + 0.18),
    fontsize=8, color="#444441",
    arrowprops=dict(arrowstyle="->", color="#444441", lw=0.8))


ax.text(20, REST_THRESH + 0.03, "Rest threshold", fontsize=8,
        color="#534AB7", fontweight="bold")


ax.set_xlim(0, T)
ax.set_ylim(0, 1.15)
ax.set_xlabel("Simulation step (30 s per step)",
              fontsize=9)
ax.set_ylabel("Activity factor  c(t) One full active day", fontsize=9)
ax.tick_params(labelsize=8)
ax.set_xticks(range(0, T + 1, 100))
ax.set_xticklabels([str(s) if s % 200 == 0 else "" for s in range(0, T + 1, 100)])
ax.grid(axis="y", color="#e8e6e0", linewidth=0.4, zorder=0)
ax.grid(axis="x", color="#e8e6e0", linewidth=0.4, zorder=0)
ax.set_axisbelow(True)


rest_patch = mpatches.Patch(color="#534AB7", alpha=0.15,
                             label="Rest period (c below threshold)")
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles + [rest_patch], labels + ["Rest period (c below threshold)"],
          fontsize=8, loc="upper right", framealpha=0.9)


ax.set_title(
    "Bimodal circadian activity factor c(t)",
    fontsize=9, loc="left", pad=6, color="#444")

plt.tight_layout()

png_path = OUTPUT_DIR / "figure2x_circadian_curve.png"
pdf_path = OUTPUT_DIR / "figure2x_circadian_curve.pdf"
plt.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="white")
plt.savefig(pdf_path,          bbox_inches="tight", facecolor="white")
plt.close()

print(f"Saved: {png_path}")
print(f"       {pdf_path}")
print(f"\nDawn start (step 0):  c = {values[0]:.4f}")
print(f"First peak (step {steps[peak1_idx]}):  c = {values[peak1_idx]:.4f}")
print(f"Dusk peak  (step {steps[peak2_idx]}):  c = {values[peak2_idx]:.4f}")
print(f"Dusk/dawn ratio: {values[peak2_idx]/values[peak1_idx]:.3f}")
