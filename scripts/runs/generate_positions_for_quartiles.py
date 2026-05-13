from __future__ import annotations
import time
from pathlib import Path

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

SCENARIO       = "scarce"
LANDSCAPE_SEED = 7
STEPS          = 1920
SEEDS          = [42, 43, 44, 45, 46, 47, 48, 49, 50]

for s in SEEDS:
    out_dir = Path("outputs") / f"{SCENARIO}_seed{s}"
    pos_path = out_dir / "positions.csv"
    if pos_path.exists():
        print(f"  seed {s}: already exists, skipping")
        continue
    t0 = time.perf_counter()
    print(f"  seed {s}: running ...", end="", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = build_scenario_config(SCENARIO, steps=STEPS, seed=s, landscape_seed=LANDSCAPE_SEED)
    sim = SheepSimulation(cfg)
    for step in range(STEPS):
        sim.step(step)
        sim.metrics.record(step, sim.flock, SCENARIO, sim.food)
    sim.metrics.position_dataframe().to_csv(pos_path, index=False)
    print(f" done ({time.perf_counter() - t0:.1f}s) -> {pos_path}")

print("\nAll positions generated.")
