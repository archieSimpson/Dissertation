from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sheep_sim.core.config import FieldConfig
from sheep_sim.environment.field import build_environment

POSITIONS_CSV = Path("outputs/iso_terrain/positions.csv")
LANDSCAPE_SEED = 7
W, H = 220.0, 140.0
COLS, ROWS = 88, 56

OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PDF = OUT_DIR / "terrain_verification.pdf"



positions = pd.read_csv(POSITIONS_CSV)
print(f"Loaded {len(positions):,} position rows from {POSITIONS_CSV}")


field_cfg = FieldConfig()
env = build_environment(np.random.default_rng(LANDSCAPE_SEED), field_cfg)
terrain = env.terrain
print(f"Terrain shape: {terrain.shape}   min/max: {terrain.min():.3f} / {terrain.max():.3f}")


heatmap, _, _ = np.histogram2d(
    positions["x"], positions["y"],
    bins=[COLS, ROWS],
    range=[[0, W], [0, H]],
)
heatmap = heatmap.T


fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(
    heatmap,
    origin="lower",
    extent=[0, W, 0, H],
    aspect="auto",
    cmap="YlGn",
)

xs_t = np.linspace(0, W, terrain.shape[1])
ys_t = np.linspace(0, H, terrain.shape[0])
ax.contour(
    xs_t, ys_t, terrain,
    levels=[0.2, 0.4, 0.6, 0.8],
    colors="black",
    linewidths=0.8,
    alpha=0.6,
)

ax.set_xlabel("Field X (m)")
ax.set_ylabel("Field Y (m)")
ax.set_title("Visit frequency overlaid on terrain — iso_terrain on abundant")
plt.colorbar(im, ax=ax, label="Visits per cell")

plt.tight_layout()
plt.savefig(OUT_PDF, dpi=200, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUT_PDF}")




hill_mask = terrain > 0.6
flat_mask = terrain < 0.2
edge_margin_cells = 4
inner_mask = np.zeros_like(terrain, dtype=bool)
inner_mask[edge_margin_cells:-edge_margin_cells, edge_margin_cells:-edge_margin_cells] = True

hill_avg = heatmap[hill_mask & inner_mask].mean() if (hill_mask & inner_mask).any() else float("nan")
flat_avg = heatmap[flat_mask & inner_mask].mean() if (flat_mask & inner_mask).any() else float("nan")


edge_avg   = heatmap[~inner_mask].mean()
centre_avg = heatmap[inner_mask].mean()

print("\nVerification diagnostics")
print(f"  hill summit cells (terrain > 0.6, interior) : mean visits = {hill_avg:.2f}")
print(f"  flat cells        (terrain < 0.2, interior) : mean visits = {flat_avg:.2f}")
if hill_avg < flat_avg:
    print(f"  → PASS: hill summits less visited than flats (ratio {hill_avg / max(flat_avg, 1e-9):.2f})")
else:
    print(f"  → FAIL: hill summits NOT less visited than flats (ratio {hill_avg / max(flat_avg, 1e-9):.2f})")
print()
print(f"  edge cells   (outer 10 m strip) : mean visits = {edge_avg:.2f}")
print(f"  centre cells (inner field)      : mean visits = {centre_avg:.2f}")
if edge_avg > centre_avg * 1.5:
    print(f"  → FAIL: edges dominate (ratio edge/centre = {edge_avg / max(centre_avg, 1e-9):.2f})")
else:
    print(f"  → PASS: edges do NOT dominate (ratio edge/centre = {edge_avg / max(centre_avg, 1e-9):.2f})")
