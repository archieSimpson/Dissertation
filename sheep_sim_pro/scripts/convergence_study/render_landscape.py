"""Render the world produced by landscape_seed = 7.

Produces a 1x3 figure: terrain elevation, abundant food landscape, scarce food
landscape. Terrain is identical across scenarios (it depends only on
landscape_seed); food differs because each scenario's NDVI generator is
parameterised differently in scenarios.py.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

LANDSCAPE_SEED = 7
OUT = HERE / "outputs" / "landscape_7.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

cfg_a = build_scenario_config("abundant", steps=1, seed=1, landscape_seed=LANDSCAPE_SEED)
cfg_s = build_scenario_config("scarce",   steps=1, seed=1, landscape_seed=LANDSCAPE_SEED)

sim_a = SheepSimulation(cfg_a)
sim_s = SheepSimulation(cfg_s)

terrain = sim_a.environment.terrain
W, H = cfg_a.field.width, cfg_a.field.height

fig, axes = plt.subplots(1, 3, figsize=(20, 5.6), dpi=160)

ax = axes[0]
im = ax.imshow(terrain, origin="lower", extent=[0, W, 0, H],
               aspect="auto", cmap="terrain")
plt.colorbar(im, ax=ax, label="Elevation (sum of 4 Gaussian hills)")
ax.set_title("Terrain  (landscape_seed = 7)", fontsize=11)
ax.set_xlabel("Field X (m)"); ax.set_ylabel("Field Y (m)")

for ax, sim, label in [
    (axes[1], sim_a, "Abundant"),
    (axes[2], sim_s, "Scarce"),
]:
    food_img = np.clip(
        0.80 * sim.food.biomass + 0.15 * sim.food.ndvi, 0.0, None,
    )
    im = ax.imshow(food_img, origin="lower", extent=[0, W, 0, H],
                   aspect="auto", cmap="YlGn",
                   vmin=0.0, vmax=float(food_img.max()))
    plt.colorbar(im, ax=ax, label="Vegetation quality")
    ax.set_title(f"{label} food landscape  (landscape_seed = 7)", fontsize=11)
    ax.set_xlabel("Field X (m)"); ax.set_ylabel("Field Y (m)")

fig.suptitle("Landscape produced by seed 7 (used across all 18 behaviour seeds)",
             fontsize=12, y=1.02)
fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}")
