"""Three per-cell visit/grazing heatmaps in identical render_heatmap_n9 style.

Produces (all N = 9, landscape_seed = 7, continuous YlOrRd, vmin=1, vmax=9):

  outputs/heatmap_scarce_n9.png             — scarce coverage (any visit)
  outputs/grazing_heatmap_scarce_n9.png     — scarce grazing  (intake > 0)
  outputs/grazing_heatmap_abundant_n9.png   — abundant grazing (intake > 0)

Coverage grids are loaded from convergence_study/outputs/seeds/<scenario>/.
Grazing grids are loaded from
convergence_study/outputs/two_landscapes/<scenario>/seeds/landscape_07/.
Missing grazing grids are simulated and cached on first run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

N_SEEDS        = 9
LANDSCAPE_SEED = 7
STEPS          = 1920
GRID_COLS      = 88
GRID_ROWS      = 56
OUT_DIR        = HERE / "outputs"


def ensure_graze_grid(scenario: str, seed: int) -> np.ndarray:
    """Return graze_grid for (scenario, LANDSCAPE_SEED, seed). Run+cache if missing."""
    seed_dir = OUT_DIR / "two_landscapes" / scenario / "seeds" / f"landscape_{LANDSCAPE_SEED:02d}" / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    graze_path = seed_dir / "graze_grid.npy"
    visit_path = seed_dir / "visit_grid.npy"

    if graze_path.exists():
        return np.load(graze_path)

    print(f"  {scenario:9s} seed {seed}: generating graze_grid.npy ", end="", flush=True)
    cfg = build_scenario_config(
        scenario, steps=STEPS, seed=seed, landscape_seed=LANDSCAPE_SEED,
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

    np.save(graze_path, graze_grid)
    if not visit_path.exists():
        np.save(visit_path, visit_grid)
    return graze_grid


def load_visit_grid(scenario: str, seed: int) -> np.ndarray:
    """Load visit_grid from the run_convergence cache."""
    return np.load(OUT_DIR / "seeds" / scenario / f"seed_{seed:02d}" / "visit_grid.npy")


def render_heatmap(
    count_grid: np.ndarray,
    scenario: str,
    kind: str,
    out_path: Path,
) -> None:
    """Single-panel per-cell count heatmap matching render_heatmap_n9.py exactly."""
    cfg = build_scenario_config(scenario, steps=1, seed=1, landscape_seed=LANDSCAPE_SEED)
    sim = SheepSimulation(cfg)
    ndvi = sim.food.ndvi
    W, H = cfg.field.width, cfg.field.height

    fig, ax = plt.subplots(figsize=(11, 7), dpi=160)
    ax.imshow(ndvi, origin="lower", extent=[0, W, 0, H],
              aspect="auto", cmap="YlGn", alpha=0.45)

    masked = np.where(count_grid > 0, count_grid, np.nan)
    im = ax.imshow(masked, origin="lower", extent=[0, W, 0, H],
                   aspect="auto", cmap="YlOrRd", alpha=0.85,
                   vmin=1, vmax=N_SEEDS)
    label = "Number of seeds visiting cell" if kind == "visit" else "Number of seeds grazing in cell"
    cbar = plt.colorbar(im, ax=ax, label=label)
    cbar.ax.axhline(1.5, color="black", linewidth=1.5)

    title_kind = "visit count" if kind == "visit" else "grazing count"
    ax.set_title(
        f"{scenario} — per-cell {title_kind} across converged seed set "
        f"(N = {N_SEEDS}, landscape_seed = {LANDSCAPE_SEED})",
        fontsize=11,
    )
    ax.set_xlabel("Field X (m)")
    ax.set_ylabel("Field Y (m)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    env_cells = int((count_grid >= 2).sum())
    total     = count_grid.size
    visited   = int((count_grid > 0).sum())
    print(f"Saved: {out_path}")
    print(f"  cells with any {kind:8s}     : {visited}/{total} ({100 * visited / total:.1f}% of field)")
    print(f"  envelope (>=2 of {N_SEEDS}) {kind:8s}: {env_cells}/{total} ({100 * env_cells / total:.1f}% of field)")
    print(f"  max {kind:8s} count          : {int(count_grid.max())} / {N_SEEDS}")
    if (count_grid > 0).any():
        print(f"  mean {kind:8s} count (occupied): {count_grid[count_grid > 0].mean():.2f}")



print("Ensuring grazing grids for abundant + scarce, seeds 1..9, landscape 7:")
for scenario in ("abundant", "scarce"):
    for seed in range(1, N_SEEDS + 1):
        ensure_graze_grid(scenario, seed)
print()


print("Accumulating count grids:")

scarce_visit = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
for s in range(1, N_SEEDS + 1):
    scarce_visit += load_visit_grid("scarce", s).astype(int)

scarce_graze   = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
abundant_graze = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
for s in range(1, N_SEEDS + 1):
    scarce_graze   += np.load(
        OUT_DIR / "two_landscapes" / "scarce"   / "seeds" / f"landscape_{LANDSCAPE_SEED:02d}" / f"seed_{s:02d}" / "graze_grid.npy"
    ).astype(int)
    abundant_graze += np.load(
        OUT_DIR / "two_landscapes" / "abundant" / "seeds" / f"landscape_{LANDSCAPE_SEED:02d}" / f"seed_{s:02d}" / "graze_grid.npy"
    ).astype(int)
print()


print("Rendering three heatmaps:\n")
render_heatmap(scarce_visit,   "scarce",   "visit",   OUT_DIR / "heatmap_scarce_n9.png")
print()
render_heatmap(scarce_graze,   "scarce",   "grazing", OUT_DIR / "grazing_heatmap_scarce_n9.png")
print()
render_heatmap(abundant_graze, "abundant", "grazing", OUT_DIR / "grazing_heatmap_abundant_n9.png")
