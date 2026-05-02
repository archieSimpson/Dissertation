"""Nearest-neighbour distance and cluster count over time for the base and
iso_social configurations on abundant (landscape 7, seed 42, 1920 steps).

  base       : all features disabled (correlated random walk, boundary only)
  iso_social : only social force enabled (Reynolds 3-zone + REGROUPING +
               departure contagion); foraging, circadian, memory,
               personality, terrain all off

NND on the primary y-axis (left), cluster count on the secondary y-axis
(right, twin). Solid lines = NND, dashed lines = clusters; colour
distinguishes base (blue) from iso_social (green).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.lines as mlines
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
SMOOTH_WIN       = 30

OUT_DIR = Path("outputs/iso_social")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = {
    "base": FeatureConfig(
        foraging=False, social=False, circadian=False,
        memory=False, personality=False, terrain=False,
    ),
    "iso_social": FeatureConfig(
        foraging=False, social=True, circadian=False,
        memory=False, personality=False, terrain=False,
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


print(f"Running base  (no features) {SCENARIO} ...")
base = run_config(CONFIGS["base"])
print(f"Running iso_social         {SCENARIO} ...")
isos = run_config(CONFIGS["iso_social"])

hours = START_HOUR + base["step"].to_numpy() * SECONDS_PER_STEP / 3600.0
base_nnd = base["mean_nearest_neighbour_distance"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
isos_nnd = isos["mean_nearest_neighbour_distance"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
base_clu = base["number_of_clusters"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()
isos_clu = isos["number_of_clusters"].rolling(SMOOTH_WIN, center=True, min_periods=1).mean()

C_BASE = "#1565C0"
C_ISO  = "#2E7D32"

fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
ax2 = ax.twinx()

ax.plot(hours, base_nnd, color=C_BASE, linestyle="-",  linewidth=2.0)
ax.plot(hours, isos_nnd, color=C_ISO,  linestyle="-",  linewidth=2.0)
ax2.plot(hours, isos_clu, color=C_ISO,  linestyle="--", linewidth=1.8)

hour_ticks = list(range(START_HOUR, 23, 2))
ax.set_xticks(hour_ticks)
ax.set_xticklabels([f"{h:02d}:00" for h in hour_ticks], fontsize=12)
ax.tick_params(axis="y", labelsize=12)
ax2.tick_params(axis="y", labelsize=12)
ax.set_xlim(START_HOUR, START_HOUR + STEPS * SECONDS_PER_STEP / 3600.0)
ax.set_ylim(bottom=0)
ax2.set_ylim(bottom=0)

ax.set_xlabel("Time of day", fontsize=16)
ax.set_ylabel("Mean NND (m)", fontsize=16)
ax2.set_ylabel("Number of clusters", fontsize=16)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = OUT_DIR / "nnd_clusters_base_vs_iso_social_abundant_L07.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

print(f"\nMean over day:")
print(f"  base       : NND = {base['mean_nearest_neighbour_distance'].mean():5.2f} m   "
      f"clusters = {base['number_of_clusters'].mean():4.2f}")
print(f"  iso_social : NND = {isos['mean_nearest_neighbour_distance'].mean():5.2f} m   "
      f"clusters = {isos['number_of_clusters'].mean():4.2f}")
