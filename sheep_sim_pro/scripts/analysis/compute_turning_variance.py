"""Turning-angle variance per behavioural state for abundant and scarce
scenarios across landscape seeds 7, 23, 41.

Method (per seed): for each agent, compute consecutive-step heading angles
from (x, y) positions, take their finite differences (wrapped to [-pi, pi])
to get turning angles, partition by `state`, and compute the sample
variance of turning angles in each state. Then aggregate across N=9 seeds.

Hypothesis: var(GRAZING turning angles) > var(TRAVELLING turning angles)
            because grazing is area-restricted search (high turning variability)
            and travelling is directed (low, consistent turning).

Caches:
  outputs/abundant_seed{42..50}/positions.csv  -- L=7 only
  outputs/scarce_seed{42..50}/positions.csv    -- L=7 only
Other (scenario, landscape, seed) combinations are run in-process.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

STEPS    = 1920
SEEDS    = [42, 43, 44, 45, 46, 47, 48, 49, 50]
SCENARIOS  = ["abundant", "scarce"]
LANDSCAPES = [7, 23, 41]


def turning_angle_variance(pos: pd.DataFrame, state: str) -> float | None:
    """Sample variance of signed turning angles in the given state."""
    p = pos[pos["state"] == state].sort_values(["sheep_id", "step"])
    if len(p) < 3:
        return None
    d = p.groupby("sheep_id")[["x", "y"]].diff()
    headings = np.arctan2(d["y"], d["x"])
    turning  = headings.groupby(p["sheep_id"]).diff()
    turning  = ((turning + math.pi) % (2 * math.pi)) - math.pi
    turning  = turning.dropna()
    if len(turning) < 10:
        return None
    return float(turning.var(ddof=1))


def get_positions(scenario: str, landscape: int, seed: int) -> pd.DataFrame:
    cached = Path(f"outputs/{scenario}_seed{seed}/positions.csv")
    if landscape == 7 and cached.exists():
        return pd.read_csv(cached)
    cfg = build_scenario_config(scenario, steps=STEPS, seed=seed, landscape_seed=landscape)
    sim = SheepSimulation(cfg)
    for step in range(STEPS):
        sim.step(step)
        sim.metrics.record(step, sim.flock, scenario, sim.food)
    return sim.metrics.position_dataframe()


rows = []
for scenario in SCENARIOS:
    for L in LANDSCAPES:
        print(f"\n=== {scenario}  L={L} ===")
        grazings: list[float] = []
        travellings: list[float] = []
        n_graz_steps_total = 0
        n_trav_steps_total = 0
        for s in SEEDS:
            t0 = time.perf_counter()
            pos = get_positions(scenario, L, s)
            v_graze = turning_angle_variance(pos, "grazing")
            v_trav  = turning_angle_variance(pos, "travelling")
            n_graz = int((pos["state"] == "grazing").sum())
            n_trav = int((pos["state"] == "travelling").sum())
            n_graz_steps_total += n_graz
            n_trav_steps_total += n_trav
            if v_graze is not None: grazings.append(v_graze)
            if v_trav  is not None: travellings.append(v_trav)
            elapsed = time.perf_counter() - t0
            graze_s = f"{v_graze:.4f}" if v_graze is not None else "  n/a"
            trav_s  = f"{v_trav:.4f}"  if v_trav  is not None else "  n/a"
            print(f"  seed {s}: graze var={graze_s:>7s} (n={n_graz:5d}) "
                  f"trav var={trav_s:>7s} (n={n_trav:5d})  ({elapsed:.1f}s)")
        rows.append({
            "scenario":              scenario,
            "landscape":             L,
            "n_seeds":               len(SEEDS),
            "grazing_var_mean":      float(np.mean(grazings)) if grazings else None,
            "grazing_var_sd":        float(np.std(grazings, ddof=1)) if len(grazings) > 1 else 0.0,
            "grazing_n_seeds":       len(grazings),
            "grazing_total_samples": n_graz_steps_total,
            "travelling_var_mean":   float(np.mean(travellings)) if travellings else None,
            "travelling_var_sd":     float(np.std(travellings, ddof=1)) if len(travellings) > 1 else 0.0,
            "travelling_n_seeds":    len(travellings),
            "travelling_total_samples": n_trav_steps_total,
        })


print("\n" + "=" * 90)
print("Summary  (sample variance of signed turning angles, rad^2)")
print("=" * 90)
header = f"{'scenario':10s} {'L':>4s}   {'GRAZING var':>22s}   {'TRAVELLING var':>22s}   {'graze > trav?':14s}"
print(header)
print("-" * 90)
for r in rows:
    g  = f"{r['grazing_var_mean']:.4f} ± {r['grazing_var_sd']:.4f}" if r["grazing_var_mean"] is not None else "—"
    tv = f"{r['travelling_var_mean']:.4f} ± {r['travelling_var_sd']:.4f}" if r["travelling_var_mean"] is not None else "— (no data)"
    if r["grazing_var_mean"] is not None and r["travelling_var_mean"] is not None:
        verdict = "YES" if r["grazing_var_mean"] > r["travelling_var_mean"] else "NO"
        ratio = f" ({r['grazing_var_mean'] / r['travelling_var_mean']:.2f}x)"
    else:
        verdict = "—"; ratio = ""
    print(f"{r['scenario']:10s} {r['landscape']:>4d}   {g:>22s}   {tv:>22s}   {verdict}{ratio}")

pd.DataFrame(rows).to_csv("turning_variance_results.csv", index=False)
print("\nSaved: turning_variance_results.csv")
