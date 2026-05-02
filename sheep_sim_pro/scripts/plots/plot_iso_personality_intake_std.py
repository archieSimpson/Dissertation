"""Inter-agent cumulative-path-length std over time for base vs iso_personality.

Metric: column `path_length_std` from metrics.csv — at each step it's the
standard deviation of `path_length` across the 40 sheep. Path length is
each agent's cumulative distance travelled, so the column is monotonically
non-decreasing and the gap between agents widens whenever per-agent
variation (movement_vigor x turning_bias) produces different
trajectories.

  base            : all features off — every agent identical except for
                    spawn position + heading RNG
  iso_personality : only personality on (movement_vigor, turning_bias
                    vary per sheep; OU drift and Markov bias are tracked
                    but unused because social / foraging are off)

Scenario: abundant on landscape 7, seed 42, 1920 steps.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sheep_sim.core.config import FeatureConfig
from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

SCENARIO         = "abundant"
LANDSCAPE        = 7
SEED             = 42
STEPS            = 1920
SECONDS_PER_STEP = 30
START_HOUR       = 6

OUT_DIR = Path("outputs/iso_personality")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "base": FeatureConfig(
        foraging=False, social=False, circadian=False,
        memory=False, personality=False, terrain=False,
    ),
    "iso_personality": FeatureConfig(
        foraging=False, social=False, circadian=False,
        memory=False, personality=True, terrain=False,
    ),
}


def run_config(features: FeatureConfig) -> pd.DataFrame:
    cfg = build_scenario_config(SCENARIO, steps=STEPS, seed=SEED, landscape_seed=LANDSCAPE)
    cfg = replace(cfg, features=features)
    sim = SheepSimulation(cfg)
    for step in range(STEPS):
        sim.step(step)
        sim.metrics.record(step, sim.flock, SCENARIO, sim.food)
    return sim.metrics.group_dataframe()


print(f"Running base            {SCENARIO} ...")
base = run_config(CONFIGS["base"])
print(f"Running iso_personality {SCENARIO} ...")
isop = run_config(CONFIGS["iso_personality"])

hours = START_HOUR + base["step"].to_numpy() * SECONDS_PER_STEP / 3600.0

C_BASE = "#999999"
C_ISO  = "#26A69A"

fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
ax.plot(hours, base["path_length_std"], color=C_BASE, linewidth=2.0, label="base")
ax.plot(hours, isop["path_length_std"], color=C_ISO,  linewidth=2.0, label="iso_personality")

hour_ticks = list(range(START_HOUR, 23, 2))
ax.set_xticks(hour_ticks)
ax.set_xticklabels([str(h) for h in hour_ticks], fontsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.set_xlim(START_HOUR, START_HOUR + STEPS * SECONDS_PER_STEP / 3600.0)
ax.set_ylim(bottom=0)
ax.set_xlabel("time of day (h)", fontsize=16)
ax.set_ylabel("variation between agents (path length std)", fontsize=16)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=14, framealpha=0.92)

fig.tight_layout()
out = OUT_DIR / "path_length_std_base_vs_iso_personality_abundant_L07.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

print(f"\nFinal (step {STEPS - 1}) path_length_std:")
print(f"  base            : {float(base['path_length_std'].iloc[-1]):.3f} m")
print(f"  iso_personality : {float(isop['path_length_std'].iloc[-1]):.3f} m")
print(f"  ratio iso/base  : "
      f"{float(isop['path_length_std'].iloc[-1]) / max(float(base['path_length_std'].iloc[-1]), 1e-9):.2f}x")
