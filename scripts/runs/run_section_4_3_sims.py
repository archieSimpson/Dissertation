from __future__ import annotations
import time
from dataclasses import replace
from pathlib import Path

from sheep_sim.core.config import FeatureConfig
from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

STEPS     = 1920
LANDSCAPE = 7
ALL_ON  = FeatureConfig()
NO_PERS = FeatureConfig(personality=False)
NO_SOC  = FeatureConfig(social=False)

RUNS = [

    ("corridors",        43, ALL_ON,  "corridors_seed43"),
    ("corridors",        44, ALL_ON,  "corridors_seed44"),
    ("corridors",        45, ALL_ON,  "corridors_seed45"),
    ("corridors",        46, ALL_ON,  "corridors_seed46"),
    ("corridors",        47, ALL_ON,  "corridors_seed47"),
    ("corridors",        48, ALL_ON,  "corridors_seed48"),
    ("corridors",        49, ALL_ON,  "corridors_seed49"),
    ("corridors",        50, ALL_ON,  "corridors_seed50"),
    ("uniform_low",      42, NO_PERS, "uniform_low_seed42_nopersonality"),
    ("radial_increase",  42, NO_SOC,  "radial_increase_seed42_nosocial"),
    ("corridors",        42, NO_SOC,  "corridors_seed42_nosocial"),
]

for scenario, seed, features, dirname in RUNS:
    out_dir = Path("outputs") / dirname
    pos_path = out_dir / "positions.csv"
    if pos_path.exists():
        print(f"  {dirname:40s}  cached, skipping")
        continue
    t0 = time.perf_counter()
    print(f"  {dirname:40s}  running ...", end="", flush=True)
    cfg = build_scenario_config(scenario, steps=STEPS, seed=seed, landscape_seed=LANDSCAPE)
    cfg = replace(cfg, features=features)
    sim = SheepSimulation(cfg)
    for step in range(STEPS):
        sim.step(step)
        sim.metrics.record(step, sim.flock, scenario, sim.food)
    out_dir.mkdir(parents=True, exist_ok=True)
    sim.metrics.position_dataframe().to_csv(pos_path, index=False)
    sim.metrics.group_dataframe().to_csv(out_dir / "metrics.csv", index=False)
    sim.metrics.field_dataframe().to_csv(out_dir / "field_health.csv", index=False)
    print(f" done ({time.perf_counter() - t0:.1f}s)")

print("\nAll §4.3 sims done.")
