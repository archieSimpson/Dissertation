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


DEFAULT_SCENARIO        = "abundant"
DEFAULT_LANDSCAPES      = [7, 23]
MAX_SEEDS               = 18
STEPS                   = 1920
GRID_COLS               = 88
GRID_ROWS               = 56
ENVELOPE_M              = 2
NEW_TERRITORY_THRESHOLD = 0.03
CONSECUTIVE_BELOW       = 2

LANDSCAPE_PALETTE = ["#2E7D32", "#1565C0", "#7B1FA2", "#EF6C00"]


def out_dir_for(scenario: str) -> Path:
    return HERE / "outputs" / "two_landscapes" / scenario


def seeds_dir_for(scenario: str) -> Path:
    return out_dir_for(scenario) / "seeds"



def run_or_load_seed(
    scenario: str, landscape_seed: int, behaviour_seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    seed_dir = (
        seeds_dir_for(scenario)
        / f"landscape_{landscape_seed:02d}"
        / f"seed_{behaviour_seed:02d}"
    )
    seed_dir.mkdir(parents=True, exist_ok=True)
    visit_path = seed_dir / "visit_grid.npy"
    graze_path = seed_dir / "graze_grid.npy"

    if visit_path.exists() and graze_path.exists():
        return np.load(visit_path), np.load(graze_path)

    cfg = build_scenario_config(
        scenario, steps=STEPS, seed=behaviour_seed,
        landscape_seed=landscape_seed,
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

        if step % 200 == 0 or step == STEPS - 1:
            print(f"\r    step {step + 1:>4d}/{STEPS}", end="", flush=True)
    print()

    np.save(visit_path, visit_grid)
    np.save(graze_path, graze_grid)
    return visit_grid, graze_grid



def study_landscape(scenario: str, landscape_seed: int) -> tuple[
    list[dict], np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None
]:
    print(f"\n=== landscape_seed = {landscape_seed} ===")
    field_area      = GRID_ROWS * GRID_COLS
    visit_count     = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    graze_count     = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    prev_visit_env  = np.zeros_like(visit_count, dtype=bool)
    prev_graze_env  = np.zeros_like(graze_count, dtype=bool)
    rows: list[dict] = []
    consecutive_below = 0
    converged_at: int | None = None

    for seed in range(1, MAX_SEEDS + 1):
        print(f"  seed {seed:2d}:")
        visit_seed, graze_seed = run_or_load_seed(scenario, landscape_seed, seed)
        visit_count = visit_count + visit_seed.astype(int)
        graze_count = graze_count + graze_seed.astype(int)

        visit_env = visit_count >= ENVELOPE_M
        graze_env = graze_count >= ENVELOPE_M

        new_visit = int((visit_env & ~prev_visit_env).sum())
        new_graze = int((graze_env & ~prev_graze_env).sum())
        new_visit_frac = new_visit / field_area
        new_graze_frac = new_graze / field_area
        visit_frac = visit_env.sum() / field_area
        graze_frac = graze_env.sum() / field_area

        rows.append({
            "landscape_seed":         landscape_seed,
            "seed":                   seed,
            "coverage_cells":         int(visit_env.sum()),
            "coverage_frac":          visit_frac,
            "coverage_new_cells":     new_visit,
            "coverage_new_frac":      new_visit_frac,
            "grazing_cells":          int(graze_env.sum()),
            "grazing_frac":           graze_frac,
            "grazing_new_cells":      new_graze,
            "grazing_new_frac":       new_graze_frac,
            "grazing_in_coverage_pct":
                100.0 * graze_env.sum() / max(visit_env.sum(), 1),
        })
        print(
            f"           coverage envelope = {visit_frac * 100:5.1f}%   "
            f"new = {new_visit_frac * 100:5.2f}%"
        )
        print(
            f"           grazing  envelope = {graze_frac * 100:5.1f}%   "
            f"new = {new_graze_frac * 100:5.2f}%   "
            f"({100.0 * graze_env.sum() / max(visit_env.sum(), 1):.0f}% of coverage)"
        )

        prev_visit_env = visit_env.copy()
        prev_graze_env = graze_env.copy()


        if seed >= 2 and new_visit_frac < NEW_TERRITORY_THRESHOLD:
            consecutive_below += 1
            if consecutive_below >= CONSECUTIVE_BELOW:
                converged_at = seed
                print(f"  → CONVERGED (coverage) at N = {seed}  "
                      f"({CONSECUTIVE_BELOW} consecutive seeds below "
                      f"{NEW_TERRITORY_THRESHOLD * 100:.0f}%)")
                break
        else:
            consecutive_below = 0

    if converged_at is None:
        print(f"  → did not converge within {MAX_SEEDS} seeds")

    return rows, visit_env, visit_count, graze_env, graze_count, converged_at



def plot_combined_curve(
    scenario: str,
    df: pd.DataFrame,
    conv_points: dict[int, int | None],
    landscape_seeds: list[int],
    colours: dict[int, str],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)

    for landscape_seed in landscape_seeds:
        sub = df[df.landscape_seed == landscape_seed].sort_values("seed")
        if sub.empty:
            continue
        sub2 = sub[sub.seed >= 2]
        c = colours[landscape_seed]
        ax.plot(
            sub2.seed, sub2.coverage_new_frac * 100,
            marker="o", linewidth=2.0, markersize=7,
            color=c, linestyle="-",
            label=f"L{landscape_seed} — coverage",
        )
        ax.plot(
            sub2.seed, sub2.grazing_new_frac * 100,
            marker="s", linewidth=1.6, markersize=6,
            color=c, linestyle="--", alpha=0.85,
            label=f"L{landscape_seed} — grazing",
        )
        n_conv = conv_points.get(landscape_seed)
        if n_conv:
            ax.axvline(
                n_conv, color=c, linestyle=":", linewidth=1.4, alpha=0.85,
            )
            ax.annotate(
                f"L{landscape_seed}: N = {n_conv}",
                xy=(n_conv, 0), xytext=(n_conv + 0.15, 0.8),
                color=c, fontsize=9, fontweight="bold",
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
        f"{scenario.capitalize()} — coverage vs grazing envelope convergence  "
        f"(M = {ENVELOPE_M}, threshold = {NEW_TERRITORY_THRESHOLD * 100:.0f}% "
        f"sustained over {CONSECUTIVE_BELOW} seeds, on coverage)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_envelope(
    scenario: str,
    landscape_seed: int,
    visit_env: np.ndarray, visit_count: np.ndarray,
    graze_env: np.ndarray, graze_count: np.ndarray,
    out_path: Path,
) -> None:
    cfg = build_scenario_config(
        scenario, steps=1, seed=1, landscape_seed=landscape_seed,
    )
    sim = SheepSimulation(cfg)
    ndvi = sim.food.ndvi
    W, H = cfg.field.width, cfg.field.height
    n_seeds_done = int(visit_count.max())
    vmax = max(n_seeds_done, ENVELOPE_M)

    fig, axes = plt.subplots(2, 2, figsize=(16, 11), dpi=160)

    panels = [
        (axes[0, 0], "coverage envelope",
         visit_env, "Reds", "coverage"),
        (axes[1, 0], "grazing envelope",
         graze_env, "Oranges", "grazing"),
    ]
    for ax, label, env, cmap, _ in panels:
        ax.imshow(ndvi, origin="lower", extent=[0, W, 0, H],
                  aspect="auto", cmap="YlGn", alpha=0.50)
        overlay = np.where(env, 1.0, np.nan)
        ax.imshow(overlay, origin="lower", extent=[0, W, 0, H],
                  aspect="auto", cmap=cmap, alpha=0.55, vmin=0, vmax=1)
        ax.set_title(
            f"L{landscape_seed} — {label} (≥ {ENVELOPE_M} of {n_seeds_done} seeds): "
            f"{env.sum() / env.size * 100:.1f}% of field",
            fontsize=10.5,
        )
        ax.set_xlabel("Field X (m)"); ax.set_ylabel("Field Y (m)")

    count_panels = [
        (axes[0, 1], "coverage count", visit_count, "YlOrRd"),
        (axes[1, 1], "grazing count",  graze_count, "OrRd"),
    ]
    for ax, label, cnt, cmap in count_panels:
        ax.imshow(ndvi, origin="lower", extent=[0, W, 0, H],
                  aspect="auto", cmap="YlGn", alpha=0.45)
        masked = np.where(cnt > 0, cnt, np.nan)
        im = ax.imshow(
            masked, origin="lower", extent=[0, W, 0, H], aspect="auto",
            cmap=cmap, alpha=0.85, vmin=1, vmax=vmax,
        )
        cbar = plt.colorbar(im, ax=ax, label=f"Number of seeds — {label}")
        cbar.ax.axhline(ENVELOPE_M - 0.5, color="black", linewidth=1.5)
        ax.set_title(
            f"L{landscape_seed} — per-cell {label}, N = {n_seeds_done}",
            fontsize=10.5,
        )
        ax.set_xlabel("Field X (m)"); ax.set_ylabel("Field Y (m)")

    fig.suptitle(
        f"Coverage vs grazing envelopes — landscape_seed = {landscape_seed}",
        fontsize=12, y=0.995,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_heatmap(
    scenario: str,
    landscape_seed: int,
    visit_count: np.ndarray, graze_count: np.ndarray,
    conv_n: int,
    out_path: Path,
) -> None:
    cfg = build_scenario_config(
        scenario, steps=1, seed=1, landscape_seed=landscape_seed,
    )
    sim = SheepSimulation(cfg)
    ndvi = sim.food.ndvi
    W, H = cfg.field.width, cfg.field.height

    fig, axes = plt.subplots(1, 2, figsize=(18, 7), dpi=160)

    for ax, label, cnt, cmap in [
        (axes[0], "coverage (any visit)",       visit_count, "YlOrRd"),
        (axes[1], "grazing (intake > 0)",       graze_count, "OrRd"),
    ]:
        ax.imshow(ndvi, origin="lower", extent=[0, W, 0, H],
                  aspect="auto", cmap="YlGn", alpha=0.45)
        masked = np.where(cnt > 0, cnt, np.nan)
        im = ax.imshow(masked, origin="lower", extent=[0, W, 0, H],
                       aspect="auto", cmap=cmap, alpha=0.85,
                       vmin=1, vmax=conv_n)
        cbar = plt.colorbar(im, ax=ax, label=f"Seeds visiting cell — {label}")
        cbar.ax.axhline(1.5, color="black", linewidth=1.5)
        ax.set_title(
            f"L{landscape_seed} — {label}, N = {conv_n}",
            fontsize=11,
        )
        ax.set_xlabel("Field X (m)"); ax.set_ylabel("Field Y (m)")

    fig.suptitle(
        f"{scenario} — converged per-cell visit counts "
        f"(landscape_seed = {landscape_seed}, N = {conv_n})",
        fontsize=12, y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--scenario", type=str, default=DEFAULT_SCENARIO,
        choices=["abundant", "scarce", "uniform_low", "uniform_high",
                 "radial_increase", "ring"],
        help="Scenario to run (default: abundant)",
    )
    p.add_argument(
        "--landscape-seeds", type=int, nargs="+", default=DEFAULT_LANDSCAPES,
        help="Two landscape seeds to compare (default: 7 23)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    scenario = args.scenario
    landscape_seeds = args.landscape_seeds
    colours = {
        s: LANDSCAPE_PALETTE[i % len(LANDSCAPE_PALETTE)]
        for i, s in enumerate(landscape_seeds)
    }

    out_dir   = out_dir_for(scenario)
    seeds_dir = seeds_dir_for(scenario)
    out_dir.mkdir(parents=True, exist_ok=True)
    seeds_dir.mkdir(parents=True, exist_ok=True)

    print(f"Two-landscape convergence study  ({scenario}, early-exit)")
    print(f"Output dir     : {out_dir}")
    print(f"Landscape seeds: {landscape_seeds}")
    print(f"M = {ENVELOPE_M}   threshold = {NEW_TERRITORY_THRESHOLD * 100:.0f}% "
          f"sustained over {CONSECUTIVE_BELOW} consecutive seeds   "
          f"max-seeds cap = {MAX_SEEDS}")

    all_rows: list[dict] = []
    envelopes: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int | None]] = {}

    for landscape_seed in landscape_seeds:
        rows, visit_env, visit_count, graze_env, graze_count, conv_n = (
            study_landscape(scenario, landscape_seed)
        )
        all_rows.extend(rows)
        envelopes[landscape_seed] = (
            visit_env, visit_count, graze_env, graze_count, conv_n,
        )

    df = pd.DataFrame(all_rows)
    csv_path = out_dir / "convergence_data_two_landscapes.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved: {csv_path}")

    conv_points = {ls: env[4] for ls, env in envelopes.items()}

    plot_combined_curve(
        scenario, df, conv_points, landscape_seeds, colours,
        out_dir / "convergence_curve_two_landscapes.png",
    )

    for landscape_seed, (visit_env, visit_count, graze_env, graze_count, conv_n) in envelopes.items():
        plot_envelope(
            scenario, landscape_seed,
            visit_env, visit_count, graze_env, graze_count,
            out_dir / f"envelope_landscape_{landscape_seed:02d}.png",
        )
        if conv_n:
            plot_heatmap(
                scenario, landscape_seed, visit_count, graze_count, conv_n,
                out_dir / f"heatmap_landscape_{landscape_seed:02d}_n{conv_n}.png",
            )

    print("\n=== Convergence summary ===")
    for landscape_seed in landscape_seeds:
        n = conv_points.get(landscape_seed)
        last_row = df[
            (df.landscape_seed == landscape_seed) & (df.seed == (n or MAX_SEEDS))
        ].iloc[0]
        if n:
            print(
                f"  landscape_seed = {landscape_seed:2d}: "
                f"converged at N = {n}   "
                f"coverage = {last_row['coverage_frac'] * 100:.1f}%   "
                f"grazing = {last_row['grazing_frac'] * 100:.1f}%   "
                f"({last_row['grazing_in_coverage_pct']:.0f}% of coverage)"
            )
        else:
            print(f"  landscape_seed = {landscape_seed:2d}: "
                  f"did not converge in {MAX_SEEDS} seeds")


if __name__ == "__main__":
    main()
