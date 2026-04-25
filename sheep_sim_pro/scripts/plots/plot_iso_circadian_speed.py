"""Average agent speed over time for the isolated-circadian configuration.

All features off except circadian. The mean speed curve reflects only the
bimodal active_factor (no food gradient, no social cohesion, no memory,
no personality variation, no terrain). This produces the cleanest possible
test of the circadian module: speed should peak when active_factor is high
and dip to ~0 during the RESTING window.

Scenario: abundant on landscape_seed = 7, seed = 42, 1920 steps.
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
SMOOTH_WIN       = 30

OUT_DIR = Path("outputs/iso_circadian")
OUT_DIR.mkdir(parents=True, exist_ok=True)

cfg = build_scenario_config(SCENARIO, steps=STEPS, seed=SEED, landscape_seed=LANDSCAPE)
features = FeatureConfig(
    foraging=False, social=False, circadian=True,
    memory=False, personality=False, terrain=False,
)
cfg = replace(cfg, features=features)

print(f"Running iso_circadian {SCENARIO} (landscape={LANDSCAPE}, seed={SEED}, {STEPS} steps) ...")
sim = SheepSimulation(cfg)

mean_speed = np.zeros(STEPS, dtype=float)
for step in range(STEPS):
    sim.step(step)
    speeds = np.array([s.speed() for s in sim.flock])
    mean_speed[step] = speeds.mean()

smoothed = pd.Series(mean_speed).rolling(SMOOTH_WIN, center=True, min_periods=1).mean().to_numpy()

steps_arr = np.arange(STEPS)
hours     = START_HOUR + steps_arr * SECONDS_PER_STEP / 3600.0

fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
ax.plot(hours, mean_speed, color="#999", linewidth=0.8, alpha=0.45)
ax.plot(hours, smoothed,   color="#0F6E56", linewidth=2.0)

ax.set_xlim(START_HOUR, START_HOUR + STEPS * SECONDS_PER_STEP / 3600.0)
ax.set_ylim(bottom=0)
hour_ticks = list(range(START_HOUR, 23, 2))
ax.set_xticks(hour_ticks)
ax.set_xticklabels([f"{h:02d}:00" for h in hour_ticks], fontsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.set_xlabel("Time of day", fontsize=16)
ax.set_ylabel("Mean flock speed (m / step)", fontsize=16)
ax.grid(True, alpha=0.3)

fig.tight_layout()
out = OUT_DIR / "speed_iso_circadian_abundant_L07.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

morning_idx = (steps_arr < 720)
evening_idx = (steps_arr >= 960)
print("\nKey statistics:")
print(f"  Mean speed (whole day) : {smoothed.mean():.3f} m/step")
print(f"  Peak speed (morning, before 12:00) : {smoothed[morning_idx].max():.3f} m/step at step {int(smoothed[morning_idx].argmax())}")
print(f"  Peak speed (evening,  after 14:00) : {smoothed[evening_idx].max():.3f} m/step at step {int(smoothed[evening_idx].argmax()) + 960}")
print(f"  Min speed (midday)     : {smoothed[(steps_arr >= 720) & (steps_arr < 960)].min():.3f} m/step")
