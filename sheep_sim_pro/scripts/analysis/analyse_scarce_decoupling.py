from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import center_of_mass, label

from sheep_sim.environment.food import build_food_landscape
from sheep_sim.scenarios import build_scenario_config


def find_patch_centres(
    ndvi: np.ndarray,
    percentile: float = 90.0,
    n_keep: int = 3,
) -> list[tuple[float, float]]:
    """Return the centroids (row, col) of the n_keep largest connected
    components above the given NDVI percentile. Naive 'all components above
    threshold' returns 60+ speckle blobs; the size distribution is bimodal so
    keeping the top-3 by area isolates the true patches."""
    threshold = np.percentile(ndvi, percentile)
    mask = ndvi >= threshold
    labelled, n = label(mask)
    if n == 0:
        return []
    sizes = np.array([(labelled == i).sum() for i in range(1, n + 1)])
    keep = np.argsort(sizes)[::-1][:n_keep] + 1
    centres = center_of_mass(mask, labelled, list(keep))
    return [(float(r), float(c)) for r, c in centres]


def corridor_mask(
    rows: int,
    cols: int,
    patch_centres: list[tuple[float, float]],
    half_width_cells: float,
    patch_exclude_radius_cells: float = 4.0,
) -> np.ndarray:
    """Cells within half_width_cells of any segment connecting two patches,
    excluding the patches themselves."""
    rr, cc = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
    mask = np.zeros((rows, cols), dtype=bool)
    for i, (r1, c1) in enumerate(patch_centres):
        for r2, c2 in patch_centres[i + 1 :]:
            dr, dc = r2 - r1, c2 - c1
            seg_len_sq = dr * dr + dc * dc
            if seg_len_sq < 1e-9:
                continue
            t = ((rr - r1) * dr + (cc - c1) * dc) / seg_len_sq
            t = np.clip(t, 0.0, 1.0)
            proj_r = r1 + t * dr
            proj_c = c1 + t * dc
            dist = np.sqrt((rr - proj_r) ** 2 + (cc - proj_c) ** 2)
            mask |= dist <= half_width_cells
    for r, c in patch_centres:
        patch_local = np.sqrt((rr - r) ** 2 + (cc - c) ** 2) <= patch_exclude_radius_cells
        mask &= ~patch_local
    return mask


def analyse_seed(
    scenario: str,
    landscape_seed: int,
    behaviour_seed: int,
    positions_csv: Path,
    half_width_cells: float = 6.0,
) -> dict:
    cfg = build_scenario_config(scenario, seed=behaviour_seed, landscape_seed=landscape_seed)
    rng = np.random.default_rng(cfg.landscape_seed)
    food = build_food_landscape(rng, cfg.field, cfg.food, cfg.scenario)
    ndvi = food.ndvi
    rows, cols = ndvi.shape
    width, height = cfg.field.width, cfg.field.height

    patch_centres = find_patch_centres(ndvi, percentile=90.0, n_keep=3)
    corridor = corridor_mask(rows, cols, patch_centres, half_width_cells=half_width_cells)
    median_ndvi = float(np.median(ndvi))
    above_median = ndvi >= median_ndvi

    df = pd.read_csv(positions_csv)
    cell_c = np.clip((df["x"].to_numpy() / width * cols).astype(int), 0, cols - 1)
    cell_r = np.clip((df["y"].to_numpy() / height * rows).astype(int), 0, rows - 1)
    states = df["state"].to_numpy()

    visits = np.zeros_like(ndvi, dtype=int)
    np.add.at(visits, (cell_r, cell_c), 1)

    above_median_per_sample = above_median[cell_r, cell_c]
    is_grazing = states == "grazing"
    is_travelling = states == "travelling"
    is_walking = states == "walking"

    return {
        "scenario": scenario,
        "landscape_seed": landscape_seed,
        "behaviour_seed": behaviour_seed,
        "n_patches_detected": len(patch_centres),
        "half_width_cells": half_width_cells,


        "mean_visits_corridor": float(visits[corridor].mean()) if corridor.any() else float("nan"),
        "mean_visits_noncorridor": float(visits[~corridor].mean()),

        "mean_visits_poor_offcorridor": float(visits[(~corridor) & (~above_median)].mean()),

        "frac_cells_with_visits_ge_5": float((visits >= 5).mean()),
        "frac_cells_any_visit": float((visits >= 1).mean()),

        "pct_grazing_above_median": (
            100.0 * float(above_median_per_sample[is_grazing].mean())
            if is_grazing.any() else float("nan")
        ),
        "pct_travelling_above_median": (
            100.0 * float(above_median_per_sample[is_travelling].mean())
            if is_travelling.any() else float("nan")
        ),
        "pct_walking_above_median": (
            100.0 * float(above_median_per_sample[is_walking].mean())
            if is_walking.any() else float("nan")
        ),
    }


def summarise(df: pd.DataFrame, label_text: str) -> None:
    n = len(df)
    if n == 0:
        print(f"{label_text}: no rows")
        return
    corridor_ratio = df["mean_visits_corridor"].mean() / max(df["mean_visits_noncorridor"].mean(), 1e-9)
    corridor_ratio_conservative = (
        df["mean_visits_corridor"].mean() / max(df["mean_visits_poor_offcorridor"].mean(), 1e-9)
    )
    print(f"\n=== {label_text} (n={n} seeds) ===")
    print(f"  patches detected (mean)              : {df['n_patches_detected'].mean():.2f}")
    print(f"  corridor / all non-corridor          : {corridor_ratio:.2f}x")
    print(f"  corridor / poor off-corridor         : {corridor_ratio_conservative:.2f}x  (conservative)")
    print(f"  mean visits per corridor cell        : {df['mean_visits_corridor'].mean():.2f}")
    print(f"  mean visits per non-corridor cell    : {df['mean_visits_noncorridor'].mean():.2f}")
    print(f"  cells with >=5 agent-steps           : {df['frac_cells_with_visits_ge_5'].mean()*100:.1f}%")
    print(f"  cells with >=1 agent-step            : {df['frac_cells_any_visit'].mean()*100:.1f}%")
    print(f"  GRAZING samples in above-median NDVI : {df['pct_grazing_above_median'].mean():.1f}%")
    print(f"  WALKING samples in above-median NDVI : {df['pct_walking_above_median'].mean():.1f}%")
    print(f"  TRAVELLING samples in above-median   : {df['pct_travelling_above_median'].mean():.1f}%")
    print(f"  GRAZING - TRAVELLING (pp)            : "
          f"{df['pct_grazing_above_median'].mean() - df['pct_travelling_above_median'].mean():.1f}")


def main() -> None:
    SCENARIO = "scarce"
    LANDSCAPE_SEED = 7
    BEHAVIOUR_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50]
    OUTPUTS_BASE = Path("outputs")
    HALF_WIDTHS = [4.0, 6.0, 8.0]

    rows_all = []
    for hw in HALF_WIDTHS:
        rows = []
        for bseed in BEHAVIOUR_SEEDS:
            positions_csv = OUTPUTS_BASE / f"{SCENARIO}_seed{bseed}" / "positions.csv"
            if not positions_csv.exists():
                print(f"skip {positions_csv}")
                continue
            rows.append(
                analyse_seed(SCENARIO, LANDSCAPE_SEED, bseed, positions_csv, half_width_cells=hw)
            )
        df_hw = pd.DataFrame(rows)
        rows_all.append(df_hw)
        summarise(df_hw, f"corridor half-width {hw:.0f} cells (~{hw*2.5:.0f} m)")

    df_all = pd.concat(rows_all, ignore_index=True)
    out_csv = Path("scarce_decoupling_stats.csv")
    df_all.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}  ({len(df_all)} rows = {len(BEHAVIOUR_SEEDS)} seeds x {len(HALF_WIDTHS)} half-widths)")


if __name__ == "__main__":
    main()
