"""Per-agent memory-map density over time for the isolated-memory
configuration of the scarce scenario.

All features off except memory. The memory map is the only spatial-cognition
input the sheep have, and with no foraging FSM and no circadian forcing they
mostly drift around their spawn, accumulating memory in cells they visit.

Picks three agents by final-step density (highest, median, lowest) and
renders their density-over-time curves as a 3-panel figure.

Density definition: fraction of memory-map cells with value > 0.01 (treats
a cell as "remembered" once it's above the decay floor noise).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sheep_sim.core.config import FeatureConfig
from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

SCENARIO         = "scarce"
LANDSCAPE        = 7
SEED             = 42
STEPS            = 1920
THRESHOLD        = 0.01
SECONDS_PER_STEP = 30
START_HOUR       = 6

OUT_DIR = Path("outputs/iso_memory")
OUT_DIR.mkdir(parents=True, exist_ok=True)


cfg = build_scenario_config(SCENARIO, steps=STEPS, seed=SEED, landscape_seed=LANDSCAPE)
features = FeatureConfig(
    foraging=False, social=False, circadian=False,
    memory=True, personality=False, terrain=False,
)
cfg = replace(cfg, features=features)

print(f"Running iso_memory {SCENARIO} (landscape={LANDSCAPE}, seed={SEED}, {STEPS} steps) ...")
sim = SheepSimulation(cfg)
n_sheep = len(sim.flock)
densities = np.zeros((n_sheep, STEPS), dtype=float)

for step in range(STEPS):
    sim.step(step)
    for sheep in sim.flock:
        densities[sheep.sheep_id, step] = (sheep.memory_map > THRESHOLD).mean() * 100.0

AGENT_ID = 30
hours = START_HOUR + np.arange(STEPS) * SECONDS_PER_STEP / 3600.0
hour_ticks = list(range(START_HOUR, 23, 2))

fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
ax.plot(hours, densities[AGENT_ID], color="#1565C0", linewidth=1.8)
ax.set_xticks(hour_ticks)
ax.set_xticklabels([f"{h:02d}:00" for h in hour_ticks], fontsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.set_xlabel("Time of day", fontsize=16)
ax.set_ylabel("Memory density (%)", fontsize=16)
ax.grid(True, alpha=0.3)
ax.set_xlim(START_HOUR, START_HOUR + STEPS * SECONDS_PER_STEP / 3600.0)
ax.set_ylim(0, max(0.5, densities[AGENT_ID].max() * 1.1))

fig.tight_layout()
out = OUT_DIR / f"memory_density_agent{AGENT_ID}_scarce_L07.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")
print(f"Agent {AGENT_ID}: final density = {densities[AGENT_ID, -1]:.2f}%   "
      f"peak across day = {densities[AGENT_ID].max():.2f}%")
