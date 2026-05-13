from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from sheep_sim.core.config import FeatureConfig
from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

SCENARIO         = "scarce"
LANDSCAPE        = 7
SEED             = 42
STEPS            = 1920
N_SHEEP          = 40
SECONDS_PER_STEP = 30
START_HOUR       = 6

STATES = ["grazing", "walking", "travelling", "regrouping", "resting"]
STATE_LABELS = {
    "grazing":    "Grazing",
    "walking":    "Walking",
    "travelling": "Travelling",
    "regrouping": "Regrouping",
    "resting":    "Resting",
}
STATE_COLOURS = {
    "grazing":    "#4CAF50",
    "walking":    "#26A69A",
    "travelling": "#1E88E5",
    "regrouping": "#FB8C00",
    "resting":    "#757575",
}

OUT_DIR = Path("outputs/iso_foraging")
OUT_DIR.mkdir(parents=True, exist_ok=True)


cfg = build_scenario_config(SCENARIO, steps=STEPS, seed=SEED, landscape_seed=LANDSCAPE)
features = FeatureConfig(
    foraging=True, social=False, circadian=False,
    memory=False, personality=False, terrain=False,
)
cfg = replace(cfg, features=features)

print(f"Running iso_foraging {SCENARIO} (landscape={LANDSCAPE}, seed={SEED}, {STEPS} steps) ...")
sim = SheepSimulation(cfg)

counts = np.zeros((len(STATES), STEPS), dtype=float)
for step in range(STEPS):
    sim.step(step)
    for i, name in enumerate(STATES):
        counts[i, step] = sum(1 for s in sim.flock if s.state.value == name)


fig, ax = plt.subplots(figsize=(8, 5), dpi=160)
steps_arr = np.arange(STEPS)
hours = START_HOUR + steps_arr * SECONDS_PER_STEP / 3600.0
fractions_pct = counts / N_SHEEP * 100.0

ax.stackplot(
    hours, fractions_pct,
    labels=[STATE_LABELS[s] for s in STATES],
    colors=[STATE_COLOURS[s] for s in STATES],
    alpha=0.92,
)
ax.set_ylim(0, 100)
ax.set_xlim(START_HOUR, START_HOUR + STEPS * SECONDS_PER_STEP / 3600.0)
hour_ticks = list(range(START_HOUR, 23, 2))
ax.set_xticks(hour_ticks)
ax.set_xticklabels([f"{h:02d}:00" for h in hour_ticks], fontsize=12)
ax.tick_params(axis="y", labelsize=12)
ax.set_ylabel("% of flock", fontsize=16)
ax.set_xlabel("Time of day", fontsize=16)
ax.grid(True, axis="y", alpha=0.3)

handles = [mpatches.Patch(color=STATE_COLOURS[s], label=STATE_LABELS[s]) for s in STATES]
ax.legend(handles=handles, loc="upper right", fontsize=14, framealpha=0.92)

fig.tight_layout()
out = OUT_DIR / "state_composition_iso_foraging_scarce_L07.png"
fig.savefig(out, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {out}")

mean_budgets = counts.mean(axis=1) / N_SHEEP * 100
print("\nMean state-time budgets across the simulated day:")
for s, pct in zip(STATES, mean_budgets):
    print(f"  {s:10s}  {pct:5.2f}%")
