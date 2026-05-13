
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.environment.food import build_food_landscape


def grid_idx(x: float, y: float, width: float, height: float,
             cols: int, rows: int) -> tuple[int, int]:
    c = min(cols - 1, max(0, int((x / width) * cols)))
    r = min(rows - 1, max(0, int((y / height) * rows)))
    return r, c


def quartile_stats_for_seed(
    scenario: str,
    landscape_seed: int,
    behaviour_seed: int,
    positions_csv: Path,
) -> dict:

    cfg = build_scenario_config(
        scenario, seed=behaviour_seed, landscape_seed=landscape_seed
    )
    rng = np.random.default_rng(cfg.landscape_seed)
    food = build_food_landscape(rng, cfg.field, cfg.food, cfg.scenario)
    ndvi = food.ndvi


    q1, q2, q3 = np.quantile(ndvi.flatten(), [0.25, 0.50, 0.75])

    quartile_mask = np.digitize(ndvi, [q1, q2, q3])


    df = pd.read_csv(positions_csv)


    cols, rows = cfg.field.grid_cols, cfg.field.grid_rows
    width, height = cfg.field.width, cfg.field.height
    cell_c = np.clip((df["x"].to_numpy() / width  * cols).astype(int), 0, cols - 1)
    cell_r = np.clip((df["y"].to_numpy() / height * rows).astype(int), 0, rows - 1)
    df_q = quartile_mask[cell_r, cell_c]


    is_grazing = (df["state"] == "grazing").to_numpy()


    visits_total = np.zeros_like(ndvi, dtype=int)
    visits_graze = np.zeros_like(ndvi, dtype=int)
    visits_other = np.zeros_like(ndvi, dtype=int)
    np.add.at(visits_total, (cell_r, cell_c), 1)
    np.add.at(visits_graze, (cell_r[ is_grazing], cell_c[ is_grazing]), 1)
    np.add.at(visits_other, (cell_r[~is_grazing], cell_c[~is_grazing]), 1)


    out = {"scenario": scenario,
           "landscape_seed": landscape_seed,
           "behaviour_seed": behaviour_seed}
    for q in range(4):
        mask = (quartile_mask == q)
        n_cells = int(mask.sum())
        out[f"Q{q+1}_n_cells"]            = n_cells
        out[f"Q{q+1}_mean_visits_total"]  = float(visits_total[mask].mean())
        out[f"Q{q+1}_mean_visits_graze"]  = float(visits_graze[mask].mean())
        out[f"Q{q+1}_mean_visits_other"]  = float(visits_other[mask].mean())
        out[f"Q{q+1}_total_visits"]       = int(visits_total[mask].sum())
    return out


def main() -> None:
    SCENARIO       = "scarce"
    LANDSCAPE_SEED = 7
    BEHAVIOUR_SEEDS = [42, 43, 44, 45, 46, 47, 48, 49, 50]
    OUTPUTS_BASE   = Path("outputs")

    rows = []
    for bseed in BEHAVIOUR_SEEDS:

        positions_csv = OUTPUTS_BASE / f"{SCENARIO}_seed{bseed}" / "positions.csv"
        if not positions_csv.exists():
            print(f"skip {positions_csv} (missing)")
            continue
        stats = quartile_stats_for_seed(SCENARIO, LANDSCAPE_SEED, bseed, positions_csv)
        rows.append(stats)

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    df.to_csv(f"ndvi_quartile_stats_{SCENARIO}.csv", index=False)


    print("\n=== Cross-seed means (n={}) ===".format(len(df)))
    q1_visits = df["Q1_mean_visits_total"].mean()
    q4_visits = df["Q4_mean_visits_total"].mean()
    q1_other  = df["Q1_mean_visits_other"].mean()
    q1_graze  = df["Q1_mean_visits_graze"].mean()
    print(f"Top quartile visits / bottom quartile visits = {q4_visits / q1_visits:.2f}x")
    print(f"Bottom quartile cells crossed (non-grazing) per cell, mean = {q1_other:.2f}")
    print(f"Bottom quartile cells grazed per cell, mean              = {q1_graze:.2f}")
    print(f"Non-grazing / grazing ratio in Q1                        = {q1_other / max(q1_graze, 1e-9):.2f}x")


if __name__ == "__main__":
    main()
