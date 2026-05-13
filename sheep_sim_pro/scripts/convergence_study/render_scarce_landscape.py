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
SCENARIO       = "scarce"
OUT            = HERE / "outputs" / f"landscape_{SCENARIO}_seed{LANDSCAPE_SEED:02d}.png"

cfg = build_scenario_config(SCENARIO, steps=1, seed=1, landscape_seed=LANDSCAPE_SEED)
sim = SheepSimulation(cfg)
food = sim.food
W, H = cfg.field.width, cfg.field.height





field_img = np.clip(
    0.78 * food.biomass + 0.12 * food.ndvi - 0.18 * (1.0 - food.health),
    0.0, None,
)

fig, ax = plt.subplots(figsize=(11, 7), dpi=160)
im = ax.imshow(
    field_img, origin="lower", extent=[0, W, 0, H],
    aspect="auto", cmap="YlGn",
    vmin=float(np.min(field_img)),
    vmax=float(np.max(field_img) + 1e-9),
    alpha=0.95,
)
plt.colorbar(im, ax=ax, label="Vegetation quality (relative — autoscaled)")

ax.set_title(
    f"{SCENARIO} — initial food landscape (landscape_seed = {LANDSCAPE_SEED})",
    fontsize=11,
)
ax.set_xlabel("Field X (m)")
ax.set_ylabel("Field Y (m)")

fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {OUT}")
print(f"  NDVI mean : {float(food.ndvi.mean()):.3f}")
print(f"  NDVI max  : {float(food.ndvi.max()):.3f}")
print(f"  biomass mean : {float(food.biomass.mean()):.3f}")
print(f"  biomass max  : {float(food.biomass.max()):.3f}")
print(f"  field area : {food.biomass.size} cells "
      f"({W:.0f} m x {H:.0f} m, "
      f"{cfg.field.cell_width:.1f} x {cfg.field.cell_height:.1f} m per cell)")
