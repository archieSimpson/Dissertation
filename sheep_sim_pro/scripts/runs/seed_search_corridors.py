"""Seed search for an even left/right split in the corridors scenario.

Runs corridors on landscape 7 for a range of behaviour seeds; counts the
final agent positions in the left corridor (x < 30) vs right corridor
(x > 190); reports the seed closest to 20/20.
"""
from __future__ import annotations
import time
import numpy as np
from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

LANDSCAPE = 7
STEPS     = 1920
SEEDS     = list(range(1, 21))
LEFT_X    = 30.0
RIGHT_X   = 190.0

results = []
for s in SEEDS:
    t0 = time.perf_counter()
    cfg = build_scenario_config("corridors", steps=STEPS, seed=s, landscape_seed=LANDSCAPE)
    sim = SheepSimulation(cfg)
    for step in range(STEPS):
        sim.step(step)
    positions = np.array([sh.position for sh in sim.flock])
    xs = positions[:, 0]
    left   = int((xs < LEFT_X).sum())
    right  = int((xs > RIGHT_X).sum())
    middle = 40 - left - right
    imbalance = abs(left - right)
    elapsed = time.perf_counter() - t0
    results.append({
        "seed": s, "left": left, "right": right, "middle": middle,
        "imbalance": imbalance, "elapsed": elapsed,
    })
    print(f"  seed {s:2d}: left={left:2d}  right={right:2d}  mid={middle:2d}  "
          f"|L-R|={imbalance:2d}   ({elapsed:.1f}s)")

print("\n=== ranked by closest-to-even ===")
results.sort(key=lambda r: r["imbalance"])
for r in results[:5]:
    print(f"  seed {r['seed']:2d}: left={r['left']:2d}  right={r['right']:2d}  "
          f"mid={r['middle']:2d}   |L-R|={r['imbalance']:2d}")
best = results[0]
print(f"\nBest seed: {best['seed']}  ->  {best['left']} left / {best['right']} right / {best['middle']} middle")
