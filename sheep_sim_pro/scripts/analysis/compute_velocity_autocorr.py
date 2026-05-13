from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

SCENARIO = "scarce"
STEPS    = 1920
SEEDS    = [42, 43, 44, 45, 46, 47, 48, 49, 50]
LANDSCAPES = [7, 23, 41]


def autocorr_for_state(pos: pd.DataFrame, state: str) -> float | None:
    sub = pos[pos["state"] == state].sort_values(["sheep_id", "step"])
    vx_segs: list[float] = []
    for _, grp in sub.groupby("sheep_id"):
        vx = grp["vx"].to_numpy()
        if len(vx) > 5:
            vx_segs.extend(vx[:30].tolist())
    if len(vx_segs) <= 20:
        return None
    vx_arr = np.array(vx_segs)
    r = float(np.corrcoef(vx_arr[:-1], vx_arr[1:])[0, 1])
    return r


def get_positions_for_seed(landscape: int, seed: int) -> pd.DataFrame:
    cached = Path(f"outputs/{SCENARIO}_seed{seed}/positions.csv")


    if landscape == 7 and cached.exists():
        return pd.read_csv(cached)
    cfg = build_scenario_config(SCENARIO, steps=STEPS, seed=seed, landscape_seed=landscape)
    sim = SheepSimulation(cfg)
    for step in range(STEPS):
        sim.step(step)
        sim.metrics.record(step, sim.flock, SCENARIO, sim.food)
    return sim.metrics.position_dataframe()


rows = []
for L in LANDSCAPES:
    print(f"\n=== landscape_seed = {L} ===")
    walks: list[float] = []
    travs: list[float] = []
    for s in SEEDS:
        t0 = time.perf_counter()
        pos = get_positions_for_seed(L, s)
        r_walk = autocorr_for_state(pos, "walking")
        r_trav = autocorr_for_state(pos, "travelling")
        elapsed = time.perf_counter() - t0
        if r_walk is not None: walks.append(r_walk)
        if r_trav is not None: travs.append(r_trav)
        walk_s = f"{r_walk:.3f}" if r_walk is not None else "  n/a"
        trav_s = f"{r_trav:.3f}" if r_trav is not None else "  n/a"
        print(f"  seed {s}: walk r={walk_s:>6}   trav r={trav_s:>6}   ({elapsed:.1f}s)")
    rows.append({
        "landscape":            L,
        "n_walking_seeds":      len(walks),
        "walking_mean":         float(np.mean(walks)) if walks else None,
        "walking_std":          float(np.std(walks, ddof=1)) if len(walks) > 1 else 0.0,
        "n_travelling_seeds":   len(travs),
        "travelling_mean":      float(np.mean(travs)) if travs else None,
        "travelling_std":       float(np.std(travs, ddof=1)) if len(travs) > 1 else 0.0,
    })

print("\n=== Summary ===")
for r in rows:
    walk = f"{r['walking_mean']:.3f} ± {r['walking_std']:.3f}" if r["walking_mean"] is not None else "—"
    trav = f"{r['travelling_mean']:.3f} ± {r['travelling_std']:.3f}" if r["travelling_mean"] is not None else "—"
    print(f"  L{r['landscape']:02d}: walking n={r['n_walking_seeds']:2d}  r = {walk:>20s}    "
          f"travelling n={r['n_travelling_seeds']:2d}  r = {trav:>20s}")
