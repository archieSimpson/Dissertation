"""Re-run §4.3.2 uniform_low metrics with ABUNDANT parameters applied.

Scenario name stays 'uniform_low' (so food.py still produces NDVI = 0.1
everywhere), but cfg.food, cfg.transitions, cfg.flock, cfg.field are
overridden with the abundant frozen parameter set after the default
'uniform_low → scarce' inheritance is applied.

Two runs: full features + no-personality ablation. Same seed/landscape
as the original §4.3.2 table.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from sheep_sim.core.config import FeatureConfig
from sheep_sim.scenarios import build_scenario_config, _apply_abundant_params
from sheep_sim.simulation import SheepSimulation

SCENARIO  = "uniform_low"
LANDSCAPE = 7
SEED      = 42
STEPS     = 1920
N_SHEEP   = 40


def run(features: FeatureConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = build_scenario_config(SCENARIO, steps=STEPS, seed=SEED, landscape_seed=LANDSCAPE)
    cfg = _apply_abundant_params(cfg)
    cfg = replace(cfg, features=features)
    sim = SheepSimulation(cfg)
    for step in range(STEPS):
        sim.step(step)
        sim.metrics.record(step, sim.flock, SCENARIO, sim.food)
    return sim.metrics.position_dataframe(), sim.metrics.group_dataframe()


def state_pct(pos: pd.DataFrame) -> dict[str, float]:
    n = len(pos)
    return {s: 100.0 * (pos["state"] == s).sum() / n
            for s in ("grazing", "walking", "travelling", "regrouping", "resting")}


def daily_distance(pos: pd.DataFrame) -> float:
    return float(pos.groupby("sheep_id")["path_length"].max().mean())


print("Running uniform_low + ABUNDANT params (full features) ...")
pos_full, m_full = run(FeatureConfig())
print("Running uniform_low + ABUNDANT params (no personality) ...")
pos_no,   m_no   = run(FeatureConfig(personality=False))

sp_full = state_pct(pos_full)
sp_no   = state_pct(pos_no)
dd_full = daily_distance(pos_full)
dd_no   = daily_distance(pos_no)
cc_full = float(m_full["number_of_clusters"].mean())
cc_no   = float(m_no["number_of_clusters"].mean())

print("\n=== §4.3.2 (re-run) — uniform_low landscape, ABUNDANT parameters ===\n")
print(f"{'State':<11} {'Full':>10} {'No personality':>18}")
for s in ("grazing", "walking", "travelling", "regrouping", "resting"):
    print(f"{s.capitalize():<11} {sp_full[s]:>9.1f}% {sp_no[s]:>17.1f}%")
print()
print(f"{'Mean daily distance':<25} {dd_full:>8.0f} m {dd_no:>15.0f} m")
print(f"{'Mean cluster count':<25} {cc_full:>10.2f} {cc_no:>17.2f}")
print()
print(f"REGROUPING delta : full {sp_full['regrouping']:.1f}%  ->  "
      f"no personality {sp_no['regrouping']:.1f}%  "
      f"(delta {sp_no['regrouping'] - sp_full['regrouping']:+.1f}%)")


cfg = build_scenario_config(SCENARIO, steps=1, seed=1, landscape_seed=LANDSCAPE)
cfg = _apply_abundant_params(cfg)
print(f"\nKey abundant params now active on uniform_low:")
print(f"  local_food_enter_graze = {cfg.transitions.local_food_enter_graze}")
print(f"  total_food_scale       = {cfg.food.total_food_scale}")
print(f"  max_biomass_per_cell   = {cfg.food.max_biomass_per_cell}")
print(f"  initial_biomass_frac   = {cfg.food.initial_biomass_fraction}")
expected_biomass = 0.1 * cfg.food.max_biomass_per_cell * cfg.food.total_food_scale * cfg.food.initial_biomass_fraction
print(f"  expected per-cell biomass = 0.1 * {cfg.food.max_biomass_per_cell} * "
      f"{cfg.food.total_food_scale} * {cfg.food.initial_biomass_fraction} = {expected_biomass:.4f}")
