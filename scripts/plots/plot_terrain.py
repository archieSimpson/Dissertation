import sys
import numpy as np
import matplotlib.pyplot as plt

from sheep_sim.core.config import SimulationConfig
from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation


SEED = 7
cfg = build_scenario_config('abundant', 5760, SEED)
sim = SheepSimulation(cfg)

W = cfg.field.width
H = cfg.field.height

fig, ax = plt.subplots(figsize=(10, 6))

im = ax.imshow(
    sim.environment.terrain,
    origin='lower',
    extent=[0, W, 0, H],
    cmap='terrain',
    aspect='auto',
)

ax.set_title('Terrain elevation surface (seed {})'.format(SEED), fontsize=13)
ax.set_xlabel('Field x-coordinate (m)', fontsize=11)
ax.set_ylabel('Field y-coordinate (m)', fontsize=11)

cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Elevation (raw Gaussian sum)', fontsize=10)

plt.tight_layout()
plt.savefig('terrain_elevation.png', dpi=150, bbox_inches='tight')
print('Saved to terrain_elevation.png')