from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from sheep_sim.core.config import FeatureConfig
from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

SCENARIO         = "abundant"
STEPS            = 1920


RUNS = [
    (23, 1),
    (41, 2),
    (53, 3),
]
W, H             = 220.0, 140.0
COLS, ROWS       = 88, 56

OUT_DIR = Path("figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES = FeatureConfig(
    foraging=False, social=False, circadian=False,
    memory=False, personality=False, terrain=True,
)


def run_and_collect(landscape_seed: int, behaviour_seed: int) -> tuple[np.ndarray, np.ndarray]:
    cfg = build_scenario_config(
        SCENARIO, steps=STEPS, seed=behaviour_seed, landscape_seed=landscape_seed,
    )
    cfg = replace(cfg, features=FEATURES)
    sim = SheepSimulation(cfg)
    xs = np.empty(STEPS * len(sim.flock))
    ys = np.empty(STEPS * len(sim.flock))
    idx = 0
    for step in range(STEPS):
        sim.step(step)
        for sh in sim.flock:
            xs[idx] = sh.position[0]
            ys[idx] = sh.position[1]
            idx += 1
    heatmap, _, _ = np.histogram2d(xs, ys, bins=[COLS, ROWS], range=[[0, W], [0, H]])
    return heatmap.T, sim.environment.terrain


def render(landscape_seed: int, behaviour_seed: int, heatmap: np.ndarray, terrain: np.ndarray) -> Path:
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(
        heatmap, origin="lower", extent=[0, W, 0, H],
        aspect="auto", cmap="YlGn",
    )

    xs_t = np.linspace(0, W, terrain.shape[1])
    ys_t = np.linspace(0, H, terrain.shape[0])
    ax.contour(
        xs_t, ys_t, terrain,
        levels=[0.2, 0.4, 0.6, 0.8],
        colors="black", linewidths=0.8, alpha=0.6,
    )


    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Visits per cell", fontsize=16)
    cbar.ax.tick_params(labelsize=12)

    plt.tight_layout(pad=0.4)
    out = OUT_DIR / f"terrain_verification_L{landscape_seed:02d}_seed{behaviour_seed:02d}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    return out


def diagnostics(landscape_seed: int, heatmap: np.ndarray, terrain: np.ndarray) -> None:
    hill_mask = terrain > 0.6
    flat_mask = terrain < 0.2
    edge = 4
    inner = np.zeros_like(terrain, dtype=bool)
    inner[edge:-edge, edge:-edge] = True

    hill_avg = heatmap[hill_mask & inner].mean() if (hill_mask & inner).any() else float("nan")
    flat_avg = heatmap[flat_mask & inner].mean() if (flat_mask & inner).any() else float("nan")
    edge_avg = heatmap[~inner].mean()
    centre_avg = heatmap[inner].mean()

    print(f"  hill (>0.6): {hill_avg:6.2f}   flat (<0.2): {flat_avg:6.2f}   "
          f"ratio hill/flat = {hill_avg / max(flat_avg, 1e-9):.2f}  "
          f"{'PASS' if hill_avg < flat_avg else 'FAIL'}")
    print(f"  edge: {edge_avg:6.2f}   centre: {centre_avg:6.2f}   "
          f"ratio edge/centre = {edge_avg / max(centre_avg, 1e-9):.2f}  "
          f"{'PASS' if edge_avg < centre_avg * 1.5 else 'FAIL'}")


for L, S in RUNS:
    print(f"\n=== landscape_seed = {L}, behaviour_seed = {S} ===")
    heatmap, terrain = run_and_collect(L, S)
    out = render(L, S, heatmap, terrain)
    print(f"  terrain min/max: {terrain.min():.3f} / {terrain.max():.3f}")
    diagnostics(L, heatmap, terrain)
    print(f"  saved: {out}")
