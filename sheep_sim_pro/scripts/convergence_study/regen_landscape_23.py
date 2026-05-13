from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

SCENARIO       = "abundant"
LANDSCAPE_SEED = 23
N_SEEDS        = 7
STEPS          = 1920
GRID_COLS      = 88
GRID_ROWS      = 56

OUT_DIR   = HERE / "outputs" / "two_landscapes" / SCENARIO
SEEDS_DIR = OUT_DIR / "seeds" / f"landscape_{LANDSCAPE_SEED:02d}"


def main() -> None:
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)


    cleared = 0
    for seed in range(1, N_SEEDS + 1):
        for fname in ("visit_grid.npy", "graze_grid.npy"):
            p = SEEDS_DIR / f"seed_{seed:02d}" / fname
            if p.exists():
                p.unlink()
                cleared += 1
    print(f"Cleared {cleared} stale cache files for landscape {LANDSCAPE_SEED}.\n")

    visit_count = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    graze_count = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)

    for seed in range(1, N_SEEDS + 1):
        seed_dir = SEEDS_DIR / f"seed_{seed:02d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        print(f"  seed {seed}: simulating ", end="", flush=True)

        cfg = build_scenario_config(
            SCENARIO, steps=STEPS, seed=seed, landscape_seed=LANDSCAPE_SEED,
        )
        sim = SheepSimulation(cfg)
        field_w, field_h = cfg.field.width, cfg.field.height
        visit_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
        graze_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)

        for step in range(STEPS):
            sim.step(step)
            positions = np.asarray([s.position for s in sim.flock])
            intakes   = np.asarray([s.last_food_intake for s in sim.flock])
            col_idx = np.clip(
                (positions[:, 0] / field_w * GRID_COLS).astype(int), 0, GRID_COLS - 1,
            )
            row_idx = np.clip(
                (positions[:, 1] / field_h * GRID_ROWS).astype(int), 0, GRID_ROWS - 1,
            )
            visit_grid[row_idx, col_idx] = True
            graze_mask = intakes > 0
            if graze_mask.any():
                graze_grid[row_idx[graze_mask], col_idx[graze_mask]] = True
            if step % 320 == 0:
                print(".", end="", flush=True)
        print(" done")

        np.save(seed_dir / "visit_grid.npy", visit_grid)
        np.save(seed_dir / "graze_grid.npy", graze_grid)
        visit_count += visit_grid.astype(int)
        graze_count += graze_grid.astype(int)


    cfg = build_scenario_config(
        SCENARIO, steps=1, seed=1, landscape_seed=LANDSCAPE_SEED,
    )
    sim = SheepSimulation(cfg)
    ndvi = sim.food.ndvi
    W, H = cfg.field.width, cfg.field.height

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=160)

    for ax, label, cnt, cmap in [
        (axes[0], "movement (any visit)",   visit_count, "YlOrRd"),
        (axes[1], "grazing (intake > 0)",   graze_count, "OrRd"),
    ]:
        ax.imshow(ndvi, origin="lower", extent=[0, W, 0, H],
                  aspect="auto", cmap="YlGn", alpha=0.45)
        masked = np.where(cnt > 0, cnt, np.nan)
        im = ax.imshow(masked, origin="lower", extent=[0, W, 0, H],
                       aspect="auto", cmap=cmap, alpha=0.85,
                       vmin=1, vmax=N_SEEDS)
        cbar = plt.colorbar(im, ax=ax, label=f"Seeds visiting cell — {label}")
        cbar.ax.axhline(1.5, color="black", linewidth=1.5)
        ax.set_title(
            f"L{LANDSCAPE_SEED} — {label}, N = {N_SEEDS}",
            fontsize=11,
        )
        ax.set_xlabel("Field X (m)"); ax.set_ylabel("Field Y (m)")

    fig.suptitle(
        f"abundant — converged per-cell visit counts "
        f"(landscape_seed = {LANDSCAPE_SEED}, N = {N_SEEDS})",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()

    out_path = OUT_DIR / f"heatmap_landscape_{LANDSCAPE_SEED:02d}_n{N_SEEDS}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    cov_pct = float((visit_count >= 2).sum()) / visit_count.size * 100
    grz_pct = float((graze_count >= 2).sum()) / graze_count.size * 100
    print(f"\nSaved: {out_path}")
    print(f"Coverage envelope (>=2 of {N_SEEDS}): {cov_pct:.1f}% of field")
    print(f"Grazing  envelope (>=2 of {N_SEEDS}): {grz_pct:.1f}% of field "
          f"({100 * grz_pct / cov_pct:.0f}% of coverage)")


if __name__ == "__main__":
    main()
