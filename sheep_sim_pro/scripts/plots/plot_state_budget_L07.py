"""State-budget time-series for abundant + scarce on landscape_seed = 7.

For each scenario, loads the converged-N metrics CSVs from chapter3_combined
(N = 7 for abundant, N = 6 for scarce on L07), averages the per-state agent
counts across seeds at each step, and renders a stacked-area plot showing the
percentage of the flock in each behavioural state across the 16-hour
simulated active window.

Output: outputs/chapter3_combined/state_budget_L07.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

LANDSCAPE        = 7
SCENARIOS        = ["abundant", "scarce"]
N_SHEEP          = 40
SECONDS_PER_STEP = 30
START_HOUR       = 6

STATES = ["grazing", "walking", "travelling", "regrouping", "resting"]
STATE_LABELS = {
    "grazing":    "Grazing",
    "walking":    "Walking",
    "travelling": "Travelling",
    "regrouping": "Regrouping",
    "resting":    "Resting",
}
STATE_COLOURS = {
    "grazing":    "#4CAF50",
    "walking":    "#26A69A",
    "travelling": "#1E88E5",
    "regrouping": "#FB8C00",
    "resting":    "#757575",
}

CSV_ROOTS = {
    "abundant": HERE / "outputs" / "chapter3_validation",
    "scarce":   HERE / "outputs" / "chapter3_validation_scarce",
}
SUMMARY_PATH = HERE / "outputs" / "chapter3_combined" / "summary.json"
OUT_PATH     = HERE / "outputs" / "chapter3_combined" / "state_budget_L07.png"


def converged_n(scenario: str) -> int:
    """Read the spatially-converged N for (scenario, L07) from the pipeline summary."""
    if not SUMMARY_PATH.exists():
        return 7 if scenario == "abundant" else 6
    summary = json.loads(SUMMARY_PATH.read_text())
    for s in summary["summaries"]:
        if s["scenario"] == scenario and s["landscape"] == LANDSCAPE:
            return int(s["N_used"])
    raise KeyError(f"no entry for {scenario} L{LANDSCAPE:02d}")


def load_state_counts(scenario: str, n_seeds: int) -> np.ndarray:
    """Return shape (n_states, n_steps) array of mean per-state agent counts."""
    csv_dir = CSV_ROOTS[scenario] / f"L{LANDSCAPE:02d}"
    seeds = []
    for s in range(1, n_seeds + 1):
        p = csv_dir / f"metrics_{scenario}_L{LANDSCAPE:02d}_S{s:02d}.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        seeds.append(pd.read_csv(p))
    n_steps = min(len(df) for df in seeds)
    counts = np.zeros((len(STATES), n_steps), dtype=float)
    for st_idx, state in enumerate(STATES):
        col = f"{state}_count"
        per_seed = np.array([df[col].to_numpy()[:n_steps] for df in seeds])
        counts[st_idx] = per_seed.mean(axis=0)
    return counts


def plot_panel(ax, counts: np.ndarray, scenario: str, n_seeds: int) -> None:
    n_steps = counts.shape[1]
    steps   = np.arange(n_steps)
    hours   = START_HOUR + steps * SECONDS_PER_STEP / 3600.0
    fractions_pct = counts / N_SHEEP * 100.0

    ax.stackplot(
        hours, fractions_pct,
        labels=[STATE_LABELS[s] for s in STATES],
        colors=[STATE_COLOURS[s] for s in STATES],
        alpha=0.92,
    )
    ax.set_ylim(0, 100)
    ax.set_xlim(START_HOUR, START_HOUR + n_steps * SECONDS_PER_STEP / 3600.0)
    hour_ticks = list(range(START_HOUR, 23, 2))
    ax.set_xticks(hour_ticks)
    ax.set_xticklabels([f"{h:02d}:00" for h in hour_ticks], fontsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.set_ylabel("% of flock", fontsize=16)
    ax.grid(True, alpha=0.3, axis="y")


def main() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), dpi=160, sharex=True)

    for ax, scenario in zip(axes, SCENARIOS):
        N = converged_n(scenario)
        counts = load_state_counts(scenario, N)
        plot_panel(ax, counts, scenario, N)

    axes[1].set_xlabel("Time of day", fontsize=16)

    handles = [mpatches.Patch(color=STATE_COLOURS[s], label=STATE_LABELS[s]) for s in STATES]
    fig.legend(
        handles=handles, loc="lower center", ncol=len(STATES),
        fontsize=14, framealpha=0.92, bbox_to_anchor=(0.5, -0.02),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1.0])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


    print("\nMean state-time budgets across the simulated day:")
    for scenario in SCENARIOS:
        N = converged_n(scenario)
        counts = load_state_counts(scenario, N)
        budgets = counts.mean(axis=1) / N_SHEEP * 100.0
        print(f"  {scenario}  (N={N}):")
        for state, pct in zip(STATES, budgets):
            print(f"    {state:10s}  {pct:5.2f}%")


if __name__ == "__main__":
    main()
