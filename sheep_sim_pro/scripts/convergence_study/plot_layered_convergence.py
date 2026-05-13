"""Layered convergence curve: 2 scenarios x 3 landscapes = 6 lines.

Reads the spatial_trace.csv produced by run_combined_validation.py for each
(scenario, landscape) pair and overlays them on a single figure. Matches the
visual style of run_convergence.py's plot_convergence_curve so this can sit
next to the existing convergence_curve.png in the dissertation.

Visual encoding:
  colour      → scenario (abundant green, scarce red)
  line style  → landscape (L07 solid, L23 dashed, L41 dotted)
  marker      → scenario (abundant circle, scarce square)
  vertical dashed line per series → that series' converged N (if any)
  horizontal dashed line at 3%   → the new-territory threshold
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
COMBINED_ROOT = HERE.parent / "outputs" / "chapter3_combined"
OUT_PATH      = HERE / "outputs" / "convergence_curve_layered.png"

SCENARIOS  = ["abundant", "scarce"]
LANDSCAPES = [7, 23, 41]

NEW_THRESHOLD   = 0.03
CONSEC_BELOW    = 2

SCENARIO_COLOUR = {"abundant": "#2E7D32", "scarce": "#D84315"}
SCENARIO_MARKER = {"abundant": "o",       "scarce": "s"}
LANDSCAPE_LSTYLE = {7: "-", 23: "--", 41: ":"}


def find_convergence(df: pd.DataFrame) -> int | None:
    streak = 0
    for _, row in df.sort_values("seed").iterrows():
        if row["seed"] >= 2 and row["new_frac"] < NEW_THRESHOLD:
            streak += 1
            if streak >= CONSEC_BELOW:
                return int(row["seed"])
        else:
            streak = 0
    return None


def main() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=160)

    max_seed = 0
    legend_handles = []
    for scenario in SCENARIOS:
        for L in LANDSCAPES:
            trace_path = COMBINED_ROOT / scenario / f"L{L:02d}" / "spatial_trace.csv"
            if not trace_path.exists():
                print(f"  WARN: missing {trace_path}, skipping")
                continue
            df = pd.read_csv(trace_path).sort_values("seed")
            df_plot = df[df.seed >= 2]
            if df_plot.empty:
                continue

            colour = SCENARIO_COLOUR[scenario]
            marker = SCENARIO_MARKER[scenario]
            lstyle = LANDSCAPE_LSTYLE[L]
            label  = f"{scenario:9s}  L{L:02d}"

            line, = ax.plot(
                df_plot.seed, df_plot.new_frac * 100,
                color=colour, linestyle=lstyle, linewidth=1.9,
                marker=marker, markersize=6, alpha=0.92,
                label=label,
            )
            legend_handles.append(line)

            n_conv = find_convergence(df)
            if n_conv:
                ax.axvline(
                    n_conv, color=colour, linestyle=lstyle,
                    linewidth=1.0, alpha=0.55,
                )
            max_seed = max(max_seed, int(df.seed.max()))


    thresh_line = ax.axhline(
        NEW_THRESHOLD * 100, color="black", linestyle="--", linewidth=1.2,
        alpha=0.55, label=f"{NEW_THRESHOLD * 100:.0f}% threshold",
    )

    ax.set_xticks(range(2, max_seed + 1))
    ax.set_xlim(1.7, max_seed + 0.3)
    ax.set_ylim(0, None)
    ax.set_xlabel("Number of seeds run (N)", fontsize=12)
    ax.set_ylabel(
        "New envelope cells added by seed N\n(% of field area)",
        fontsize=11,
    )
    ax.set_title(
        "Spatial-envelope convergence — 2 scenarios × 3 landscapes  "
        f"(M = 2, threshold = {NEW_THRESHOLD * 100:.0f}% over "
        f"{CONSEC_BELOW} consecutive seeds)",
        fontsize=11,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(
        handles=legend_handles + [thresh_line],
        loc="upper right", fontsize=9, ncol=2, framealpha=0.92,
    )
    fig.tight_layout()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PATH, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
