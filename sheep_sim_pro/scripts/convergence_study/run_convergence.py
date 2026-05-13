from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation



SCENARIOS                = ["abundant", "scarce"]
DEFAULT_MAX_SEEDS        = 18
DEFAULT_STEPS            = 1920
LANDSCAPE_SEED           = 7
GRID_COLS                = 88
GRID_ROWS                = 56
ENVELOPE_M               = 2
NEW_TERRITORY_THRESHOLD  = 0.03
CONSECUTIVE_BELOW        = 2
SENSITIVITY_THRESHOLDS   = [0.01, 0.02, 0.03, 0.04, 0.05]
SCENARIO_COLOUR          = {"abundant": "#2E7D32", "scarce": "#D84315"}

OUT_DIR   = HERE / "outputs"
SEEDS_DIR = OUT_DIR / "seeds"



def run_or_load_seed(scenario: str, seed: int, steps: int) -> np.ndarray:
    seed_dir = SEEDS_DIR / scenario / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    grid_path = seed_dir / "visit_grid.npy"

    if grid_path.exists():
        return np.load(grid_path)

    cfg = build_scenario_config(
        scenario, steps=steps, seed=seed, landscape_seed=LANDSCAPE_SEED
    )
    sim = SheepSimulation(cfg)
    field_w, field_h = cfg.field.width, cfg.field.height
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)

    for step in range(steps):
        sim.step(step)
        positions = np.asarray([s.position for s in sim.flock])
        col_idx = np.clip(
            (positions[:, 0] / field_w * GRID_COLS).astype(int), 0, GRID_COLS - 1
        )
        row_idx = np.clip(
            (positions[:, 1] / field_h * GRID_ROWS).astype(int), 0, GRID_ROWS - 1
        )
        grid[row_idx, col_idx] = True
        if step % 200 == 0 or step == steps - 1:
            print(f"\r    step {step + 1:>4d}/{steps}", end="", flush=True)
    print()

    np.save(grid_path, grid)
    return grid



def study_scenario(scenario: str, max_seeds: int, steps: int) -> tuple[
    list[dict], np.ndarray, np.ndarray
]:
    print(f"\n=== Scenario: {scenario} ===")
    field_area    = GRID_ROWS * GRID_COLS
    visit_count   = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    prev_envelope = np.zeros_like(visit_count, dtype=bool)
    rows: list[dict] = []

    for seed in range(1, max_seeds + 1):
        print(f"  seed {seed:2d}:")
        per_seed    = run_or_load_seed(scenario, seed, steps)
        visit_count = visit_count + per_seed.astype(int)
        envelope    = visit_count >= ENVELOPE_M
        new_cells   = int((envelope & ~prev_envelope).sum())
        new_frac    = new_cells / field_area
        env_frac    = envelope.sum() / field_area

        rows.append({
            "scenario":       scenario,
            "seed":           seed,
            "envelope_cells": int(envelope.sum()),
            "envelope_frac":  env_frac,
            "new_cells":      new_cells,
            "new_frac":       new_frac,
        })
        print(
            f"           envelope = {env_frac * 100:5.1f}% of field   "
            f"new this seed = {new_frac * 100:5.2f}%"
        )
        prev_envelope = envelope.copy()

    return rows, prev_envelope, visit_count



def find_convergence_point(
    df: pd.DataFrame, threshold: float, consecutive: int
) -> int | None:
    streak = 0
    for _, row in df.sort_values("seed").iterrows():
        if row["seed"] >= 2 and row["new_frac"] < threshold:
            streak += 1
            if streak >= consecutive:
                return int(row["seed"])
        else:
            streak = 0
    return None


def sensitivity_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for thr in SENSITIVITY_THRESHOLDS:
        for scenario in SCENARIOS:
            sub = df[df.scenario == scenario]
            n   = find_convergence_point(sub, thr, CONSECUTIVE_BELOW)
            rows.append({
                "threshold_pct":   thr * 100,
                "scenario":        scenario,
                "converged_at_N":  n,
            })
    return pd.DataFrame(rows)



def plot_convergence_curve(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), dpi=160)

    for scenario in SCENARIOS:
        sub = df[df.scenario == scenario].sort_values("seed")
        if sub.empty:
            continue
        sub2 = sub[sub.seed >= 2]
        ax.plot(
            sub2.seed, sub2.new_frac * 100,
            marker="o", linewidth=2.0, markersize=7,
            color=SCENARIO_COLOUR[scenario], label=scenario,
        )
        n_conv = find_convergence_point(sub, NEW_TERRITORY_THRESHOLD, CONSECUTIVE_BELOW)
        if n_conv:
            ax.axvline(
                n_conv, color=SCENARIO_COLOUR[scenario],
                linestyle=":", linewidth=1.4, alpha=0.85,
            )
            ax.annotate(
                f"{scenario}: N = {n_conv}",
                xy=(n_conv, 0), xytext=(n_conv + 0.15, 0.4),
                color=SCENARIO_COLOUR[scenario], fontsize=9, fontweight="bold",
            )

    ax.axhline(
        NEW_TERRITORY_THRESHOLD * 100, color="black",
        linestyle="--", linewidth=1.2, alpha=0.55,
        label=f"{NEW_TERRITORY_THRESHOLD * 100:.0f}% threshold",
    )

    max_seed = int(df["seed"].max())
    ax.set_xticks(range(2, max_seed + 1))
    ax.set_xlim(1.7, max_seed + 0.3)
    ax.set_ylim(0, None)
    ax.set_xlabel("Number of seeds run (N)", fontsize=12)
    ax.set_ylabel(
        "New envelope cells added by seed N\n(% of field area)", fontsize=11,
    )
    ax.set_title(
        f"Spatial-envelope convergence  "
        f"(M = {ENVELOPE_M}, threshold = {NEW_TERRITORY_THRESHOLD * 100:.0f}% "
        f"sustained over {CONSECUTIVE_BELOW} consecutive seeds)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_envelope(
    scenario: str,
    envelope: np.ndarray,
    visit_count: np.ndarray,
    steps: int,
    out_path: Path,
) -> None:
    cfg = build_scenario_config(
        scenario, steps=steps, seed=1, landscape_seed=LANDSCAPE_SEED
    )
    sim = SheepSimulation(cfg)
    ndvi = sim.food.ndvi
    field_w, field_h = cfg.field.width, cfg.field.height
    n_seeds_done = int(visit_count.max())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), dpi=160)

    ax = axes[0]
    ax.imshow(ndvi, origin="lower", extent=[0, field_w, 0, field_h],
              aspect="auto", cmap="YlGn", alpha=0.50)
    overlay = np.where(envelope, 1.0, np.nan)
    ax.imshow(overlay, origin="lower", extent=[0, field_w, 0, field_h],
              aspect="auto", cmap="Reds", alpha=0.55, vmin=0, vmax=1)
    ax.set_title(
        f"{scenario} — envelope (cells visited by ≥ {ENVELOPE_M} of "
        f"{n_seeds_done} seeds)", fontsize=11,
    )
    ax.set_xlabel("Field X (m)"); ax.set_ylabel("Field Y (m)")

    ax = axes[1]
    ax.imshow(ndvi, origin="lower", extent=[0, field_w, 0, field_h],
              aspect="auto", cmap="YlGn", alpha=0.45)
    masked_count = np.where(visit_count > 0, visit_count, np.nan)
    im = ax.imshow(
        masked_count, origin="lower",
        extent=[0, field_w, 0, field_h], aspect="auto",
        cmap="YlOrRd", alpha=0.85,
        vmin=1, vmax=max(n_seeds_done, ENVELOPE_M),
    )
    cbar = plt.colorbar(im, ax=ax, label="Number of seeds visiting cell")
    cbar.ax.axhline(ENVELOPE_M - 0.5, color="black", linewidth=1.5)
    ax.set_title(
        f"{scenario} — per-cell visit count across {n_seeds_done} seeds",
        fontsize=11,
    )
    ax.set_xlabel("Field X (m)"); ax.set_ylabel("Field Y (m)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-seeds", type=int, default=DEFAULT_MAX_SEEDS,
                   help="Number of behaviour seeds to run per scenario")
    p.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                   help="Simulation steps per seed (1920 = one circadian day)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SEEDS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Convergence study  ({args.max_seeds} seeds × {len(SCENARIOS)} scenarios "
          f"× {args.steps} steps)")
    print(f"Landscape seed (fixed): {LANDSCAPE_SEED}")
    print(f"Envelope M = {ENVELOPE_M}   threshold = {NEW_TERRITORY_THRESHOLD * 100:.0f}% "
          f"sustained over {CONSECUTIVE_BELOW} seeds")

    all_rows: list[dict] = []
    envelopes: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for scenario in SCENARIOS:
        rows, envelope, visit_count = study_scenario(
            scenario, max_seeds=args.max_seeds, steps=args.steps,
        )
        all_rows.extend(rows)
        envelopes[scenario] = (envelope, visit_count)

    df = pd.DataFrame(all_rows)
    csv_path = OUT_DIR / "convergence_data.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    plot_convergence_curve(df, OUT_DIR / "convergence_curve.png")

    for scenario, (envelope, visit_count) in envelopes.items():
        plot_envelope(
            scenario, envelope, visit_count, args.steps,
            OUT_DIR / f"envelope_{scenario}.png",
        )

    sens = sensitivity_table(df)
    sens_path = OUT_DIR / "sensitivity.csv"
    sens.to_csv(sens_path, index=False)
    print(f"Saved: {sens_path}")

    print("\n=== Convergence summary  (M = "
          f"{ENVELOPE_M}, {CONSECUTIVE_BELOW} consecutive seeds below threshold) ===")
    print(f"{'threshold':>10}  " + "  ".join(f"{s:>10}" for s in SCENARIOS))
    for thr in SENSITIVITY_THRESHOLDS:
        line = f"{thr * 100:>9.0f}%  "
        for scenario in SCENARIOS:
            n = find_convergence_point(
                df[df.scenario == scenario], thr, CONSECUTIVE_BELOW,
            )
            line += f"{(str(n) if n else '— '):>10}  "
        print(line)


if __name__ == "__main__":
    main()
