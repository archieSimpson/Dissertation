"""
Diagnostic recording for §4.x behavioural confirmations.

Four scenarios × three behaviour seeds (landscape_seed = 7 fixed). Per-seed
outputs: metrics.csv (per-step group metrics), visit_grid.npy (binary
coverage), snapshots.npz (positions/velocities/states at 08:00, 14:00,
22:00 — radial_increase + ring only). Per-scenario figure suites tailored
to each scenario's evidence type.

  uniform_high     state proportions, mean speed, coverage heatmap
  uniform_low      + memory density
  radial_increase  + NND, cluster count, contagion, snapshots
  ring             + cumulative food, memory density, snapshots

Re-runs only re-simulate seeds whose cached files are absent. Plots are
always re-rendered from cached data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation



SCENARIOS        = ["uniform_high", "uniform_low", "radial_increase", "ring"]
SEEDS            = [1, 2, 3]
LANDSCAPE_SEED   = 7
STEPS            = 1920
N_SHEEP          = 40
SECONDS_PER_STEP = 30
START_HOUR       = 6

GRID_COLS = 88
GRID_ROWS = 56

SNAPSHOT_HOURS  = [8, 14, 22]
SNAPSHOT_LABELS = [f"{h:02d}:00" for h in SNAPSHOT_HOURS]
SNAPSHOT_STEPS  = [
    min(STEPS - 1, int((h - START_HOUR) * 3600 / SECONDS_PER_STEP))
    for h in SNAPSHOT_HOURS
]
SNAPSHOT_SCENARIOS = {"radial_increase", "ring"}

STATE_COLOURS = {
    "grazing":    "#4CAF50",
    "walking":    "#26A69A",
    "travelling": "#1E88E5",
    "regrouping": "#FB8C00",
    "resting":    "#757575",
}
SEED_COLOURS = {1: "#1565C0", 2: "#2E7D32", 3: "#7B1FA2"}

OUT_DIR = HERE / "outputs"



def run_or_load_seed(scenario: str, seed: int) -> Path:
    seed_dir = OUT_DIR / scenario / f"seed_{seed:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    metrics_path   = seed_dir / "metrics.csv"
    visit_path     = seed_dir / "visit_grid.npy"
    snapshots_path = seed_dir / "snapshots.npz"
    needs_snaps    = scenario in SNAPSHOT_SCENARIOS

    cached = (
        metrics_path.exists() and visit_path.exists()
        and (snapshots_path.exists() or not needs_snaps)
    )
    if cached:
        print(f"  {scenario:>16s}  seed {seed}: cached")
        return seed_dir

    print(f"  {scenario:>16s}  seed {seed}: simulating ", end="", flush=True)
    cfg = build_scenario_config(
        scenario, steps=STEPS, seed=seed, landscape_seed=LANDSCAPE_SEED,
    )
    sim = SheepSimulation(cfg)
    field_w, field_h = cfg.field.width, cfg.field.height
    visit_grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    snapshots: dict[int, dict] = {}

    for step in range(STEPS):
        sim.step(step)
        sim.metrics.record(step, sim.flock, scenario, sim.food)
        positions = np.asarray([s.position for s in sim.flock])
        col_idx = np.clip(
            (positions[:, 0] / field_w * GRID_COLS).astype(int), 0, GRID_COLS - 1,
        )
        row_idx = np.clip(
            (positions[:, 1] / field_h * GRID_ROWS).astype(int), 0, GRID_ROWS - 1,
        )
        visit_grid[row_idx, col_idx] = True

        if needs_snaps and step in SNAPSHOT_STEPS:
            snapshots[step] = {
                "positions":  positions.copy(),
                "velocities": np.asarray([s.velocity for s in sim.flock]),
                "states":     np.array([s.state.value for s in sim.flock]),
            }

        if step % 320 == 0:
            print(".", end="", flush=True)
    print(" done")

    sim.metrics.group_dataframe().to_csv(metrics_path, index=False)
    np.save(visit_path, visit_grid)
    if needs_snaps:
        save = {}
        for step, d in snapshots.items():
            save[f"s{step}_positions"]  = d["positions"]
            save[f"s{step}_velocities"] = d["velocities"]
            save[f"s{step}_states"]     = d["states"]
        np.savez(snapshots_path, **save)
    return seed_dir



def load_metrics(scenario: str) -> dict[int, pd.DataFrame]:
    return {
        s: pd.read_csv(OUT_DIR / scenario / f"seed_{s:02d}" / "metrics.csv")
        for s in SEEDS
    }


def load_visit_grids(scenario: str) -> dict[int, np.ndarray]:
    return {
        s: np.load(OUT_DIR / scenario / f"seed_{s:02d}" / "visit_grid.npy")
        for s in SEEDS
    }


def load_snapshots(scenario: str) -> dict[int, np.lib.npyio.NpzFile]:
    out = {}
    for s in SEEDS:
        path = OUT_DIR / scenario / f"seed_{s:02d}" / "snapshots.npz"
        if path.exists():
            out[s] = np.load(path)
    return out


def steps_to_hours(steps: np.ndarray) -> np.ndarray:
    return START_HOUR + steps * SECONDS_PER_STEP / 3600.0


def style_hour_axis(ax) -> None:
    ax.set_xlim(START_HOUR, START_HOUR + STEPS * SECONDS_PER_STEP / 3600.0)
    hour_ticks = list(range(START_HOUR, 23, 2))
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels([f"{h:02d}:00" for h in hour_ticks])
    ax.set_xlabel("Time of day", fontsize=11)



def plot_state_proportions(scenario: str, out_path: Path) -> None:
    metrics = load_metrics(scenario)
    state_cols = ["grazing_count", "walking_count", "travelling_count",
                  "regrouping_count", "resting_count"]
    state_keys = ["grazing", "walking", "travelling", "regrouping", "resting"]

    n_steps = len(next(iter(metrics.values())))
    fractions = np.zeros((len(state_keys), n_steps))
    for col_idx, col in enumerate(state_cols):
        per_seed = np.array([m[col].to_numpy() / N_SHEEP for m in metrics.values()])
        fractions[col_idx] = per_seed.mean(axis=0)

    steps = np.arange(n_steps)
    hours = steps_to_hours(steps)

    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)
    ax.stackplot(
        hours, fractions * 100,
        labels=state_keys,
        colors=[STATE_COLOURS[k] for k in state_keys],
        alpha=0.92,
    )
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of flock", fontsize=11)
    ax.set_title(
        f"{scenario} — state-time budget (mean across {len(SEEDS)} seeds)",
        fontsize=11,
    )
    style_hour_axis(ax)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.85, ncol=5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")



def plot_metric_lines(
    scenario: str, column: str, ylabel: str, title_suffix: str, out_path: Path,
    show_mean: bool = True,
) -> None:
    metrics = load_metrics(scenario)
    fig, ax = plt.subplots(figsize=(11, 5), dpi=160)

    series_list = []
    last_hours = None
    for seed in SEEDS:
        df = metrics[seed]
        steps = df["step"].to_numpy()
        hours = steps_to_hours(steps)
        last_hours = hours
        series = df[column].to_numpy()
        series_list.append(series)
        ax.plot(hours, series, color=SEED_COLOURS[seed], alpha=0.55,
                linewidth=1.4, label=f"seed {seed}")

    if show_mean:
        mean_series = np.array(series_list).mean(axis=0)
        ax.plot(last_hours, mean_series, color="black", linewidth=2.0,
                linestyle="--", label="mean", alpha=0.9)

    style_hour_axis(ax)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(f"{scenario} — {title_suffix}", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")



def plot_coverage_heatmap(scenario: str, out_path: Path) -> None:
    cfg = build_scenario_config(
        scenario, steps=1, seed=1, landscape_seed=LANDSCAPE_SEED,
    )
    sim = SheepSimulation(cfg)
    ndvi = sim.food.ndvi
    W, H = cfg.field.width, cfg.field.height

    grids = load_visit_grids(scenario)
    combined = sum(g.astype(int) for g in grids.values())

    fig, ax = plt.subplots(figsize=(11, 7), dpi=160)
    ax.imshow(ndvi, origin="lower", extent=[0, W, 0, H],
              aspect="auto", cmap="YlGn", alpha=0.45)
    masked = np.where(combined > 0, combined, np.nan)
    im = ax.imshow(masked, origin="lower", extent=[0, W, 0, H],
                   aspect="auto", cmap="YlOrRd", alpha=0.85,
                   vmin=1, vmax=len(SEEDS))
    plt.colorbar(im, ax=ax, label="Number of seeds visiting cell")
    ax.set_title(
        f"{scenario} — coverage envelope across {len(SEEDS)} seeds  "
        f"(landscape_seed = {LANDSCAPE_SEED})",
        fontsize=11,
    )
    ax.set_xlabel("Field X (m)"); ax.set_ylabel("Field Y (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")



def plot_snapshots(scenario: str, out_path: Path) -> None:
    cfg = build_scenario_config(
        scenario, steps=1, seed=1, landscape_seed=LANDSCAPE_SEED,
    )
    sim = SheepSimulation(cfg)
    ndvi = sim.food.ndvi
    W, H = cfg.field.width, cfg.field.height

    snaps = load_snapshots(scenario)
    if not snaps:
        return

    n_rows = len(SEEDS)
    n_cols = len(SNAPSHOT_STEPS)
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5 * n_cols, 4 * n_rows), dpi=160,
    )
    for row, seed in enumerate(SEEDS):
        d = snaps[seed]
        for col, (step, label) in enumerate(zip(SNAPSHOT_STEPS, SNAPSHOT_LABELS)):
            ax = axes[row, col]
            ax.imshow(ndvi, origin="lower", extent=[0, W, 0, H],
                      aspect="auto", cmap="YlGn", alpha=0.55)
            positions = d[f"s{step}_positions"]
            states    = d[f"s{step}_states"]
            colours   = [STATE_COLOURS[s] for s in states]
            ax.scatter(positions[:, 0], positions[:, 1], c=colours, s=42,
                       edgecolors="black", linewidths=0.4, zorder=3)
            ax.set_xlim(0, W); ax.set_ylim(0, H)
            if row == 0:
                ax.set_title(label, fontsize=11)
            if col == 0:
                ax.set_ylabel(f"seed {seed}\nField Y (m)", fontsize=10)
            if row == n_rows - 1:
                ax.set_xlabel("Field X (m)")

    fig.suptitle(
        f"{scenario} — flock snapshots at 08:00 / 14:00 / 22:00",
        fontsize=12, y=1.005,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {out_path.name}")



def figures_for_scenario(scenario: str) -> None:
    print(f"\n[{scenario}] generating figures...")
    sc_dir = OUT_DIR / scenario
    plot_state_proportions(scenario, sc_dir / "state_proportions.png")
    plot_metric_lines(
        scenario, "mean_speed", "Mean speed (m/step)",
        "mean speed per step", sc_dir / "mean_speed.png",
    )
    plot_coverage_heatmap(scenario, sc_dir / "coverage_heatmap.png")

    if scenario == "uniform_low":
        plot_metric_lines(
            scenario, "memory_density",
            "Memory density (fraction of grid cells with non-zero memory)",
            "memory map density", sc_dir / "memory_density.png",
        )

    if scenario == "radial_increase":
        plot_metric_lines(
            scenario, "mean_nearest_neighbour_distance",
            "Mean NND (m)", "nearest-neighbour distance",
            sc_dir / "nnd_per_step.png",
        )
        plot_metric_lines(
            scenario, "number_of_clusters",
            "Cluster count", "number of connected clusters",
            sc_dir / "clusters_per_step.png",
        )
        plot_metric_lines(
            scenario, "contagion_active",
            "Sheep with active contagion timer",
            "departure-following contagion",
            sc_dir / "contagion_per_step.png",
        )
        plot_snapshots(scenario, sc_dir / "snapshots.png")

    if scenario == "ring":
        plot_metric_lines(
            scenario, "mean_cumulative_food",
            "Mean cumulative food intake",
            "cumulative food intake (rises sharply on patch discovery)",
            sc_dir / "cumulative_food.png",
        )
        plot_metric_lines(
            scenario, "memory_density",
            "Memory density (fraction of grid cells with non-zero memory)",
            "memory map density", sc_dir / "memory_density.png",
        )
        plot_snapshots(scenario, sc_dir / "snapshots.png")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Diagnostic runs: {len(SCENARIOS)} scenarios × {len(SEEDS)} seeds = "
        f"{len(SCENARIOS) * len(SEEDS)} sims, {STEPS} steps each "
        f"(landscape_seed = {LANDSCAPE_SEED})"
    )
    print(f"Snapshot steps: {SNAPSHOT_STEPS} = {SNAPSHOT_LABELS}")

    for scenario in SCENARIOS:
        print(f"\n=== {scenario} ===")
        for seed in SEEDS:
            run_or_load_seed(scenario, seed)

    print("\n=== Generating figures ===")
    for scenario in SCENARIOS:
        figures_for_scenario(scenario)

    print(f"\nAll outputs at: {OUT_DIR}/")


if __name__ == "__main__":
    main()
