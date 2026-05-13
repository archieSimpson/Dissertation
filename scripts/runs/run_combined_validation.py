from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

SCENARIOS  = ["abundant", "scarce"]
LANDSCAPES = [7, 23, 41]
STEPS      = 1920
N_SHEEP    = 40
MAX_SEEDS  = 18
GRID_COLS  = 88
GRID_ROWS  = 56

ENV_M           = 2
NEW_THRESHOLD   = 0.03
CONSEC_BELOW    = 2

ROOT             = HERE / "outputs" / "chapter3_combined"
SPATIAL_ROOT     = HERE / "convergence_study" / "outputs" / "two_landscapes"
METRICS_ROOTS    = {
    "abundant": HERE / "outputs" / "chapter3_validation",
    "scarce":   HERE / "outputs" / "chapter3_validation_scarce",
}





def cache_paths(scenario: str, landscape: int, seed: int) -> tuple[Path, Path, Path]:
    spatial_dir = SPATIAL_ROOT / scenario / "seeds" / f"landscape_{landscape:02d}" / f"seed_{seed:02d}"
    metrics_dir = METRICS_ROOTS[scenario] / f"L{landscape:02d}"
    return (
        spatial_dir / "visit_grid.npy",
        spatial_dir / "graze_grid.npy",
        metrics_dir / f"metrics_{scenario}_L{landscape:02d}_S{seed:02d}.csv",
    )


def run_or_load(scenario: str, landscape: int, seed: int) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, bool]:
    visit_p, graze_p, metrics_p = cache_paths(scenario, landscape, seed)
    if visit_p.exists() and graze_p.exists() and metrics_p.exists():
        return np.load(visit_p), np.load(graze_p), pd.read_csv(metrics_p), False

    cfg = build_scenario_config(scenario, steps=STEPS, seed=seed, landscape_seed=landscape)
    sim = SheepSimulation(cfg)
    visit = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    graze = np.zeros((GRID_ROWS, GRID_COLS), dtype=bool)
    fw, fh = cfg.field.width, cfg.field.height

    for step in range(STEPS):
        sim.step(step)
        sim.metrics.record(step, sim.flock, scenario, sim.food)
        positions = np.asarray([s.position for s in sim.flock])
        intakes   = np.asarray([s.last_food_intake for s in sim.flock])
        col_idx = np.clip((positions[:, 0] / fw * GRID_COLS).astype(int), 0, GRID_COLS - 1)
        row_idx = np.clip((positions[:, 1] / fh * GRID_ROWS).astype(int), 0, GRID_ROWS - 1)
        visit[row_idx, col_idx] = True
        gm = intakes > 0
        if gm.any():
            graze[row_idx[gm], col_idx[gm]] = True

    visit_p.parent.mkdir(parents=True, exist_ok=True)
    np.save(visit_p, visit)
    np.save(graze_p, graze)
    metrics_df = sim.metrics.group_dataframe()
    metrics_p.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(metrics_p, index=False)
    return visit, graze, metrics_df, True





def four_metrics(metrics_df: pd.DataFrame) -> dict[str, float]:
    df = metrics_df
    active = df[df["resting_count"] < 26]
    nnd_active = (
        float(active["mean_nearest_neighbour_distance"].mean()) if len(active) else float("nan")
    )
    resting_pct    = float((df["resting_count"] / N_SHEEP).mean() * 100.0)
    daily_distance = float(df["mean_path_length"].iloc[-1])

    smoothed = df["mean_speed"].rolling(60, center=True, min_periods=1).mean().to_numpy()
    n = len(smoothed)
    pre_hi  = min(720, n - 1)
    post_lo = min(961, n - 1)
    pre_peaks  = [
        i for i in range(1, pre_hi)
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
    ]
    post_peaks = [
        i for i in range(post_lo, n - 1)
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
    ]
    pre_max  = float(smoothed[:pre_hi].max())  if pre_hi  > 0 else float("-inf")
    post_max = float(smoothed[960:n].max())    if n > 960  else float("-inf")
    bimodal  = 1.0 if (pre_peaks and post_peaks and post_max > pre_max) else 0.0

    return {
        "NND_active":     nnd_active,
        "resting_pct":    resting_pct,
        "daily_distance": daily_distance,
        "bimodal_pass":   bimodal,
    }





def study(scenario: str, landscape: int) -> dict:
    print(f"\n{'─' * 72}\n{scenario}  L{landscape:02d}\n{'─' * 72}", flush=True)
    L_dir = ROOT / scenario / f"L{landscape:02d}"
    L_dir.mkdir(parents=True, exist_ok=True)

    visit_count   = np.zeros((GRID_ROWS, GRID_COLS), dtype=int)
    prev_envelope = np.zeros_like(visit_count, dtype=bool)
    field_area    = GRID_ROWS * GRID_COLS
    spatial_rows: list[dict] = []
    seed_metrics_dfs: list[pd.DataFrame] = []
    consec_below = 0
    N_conv: int | None = None

    for S in range(1, MAX_SEEDS + 1):
        t0 = time.perf_counter()
        visit, _graze, metrics_df, ran_now = run_or_load(scenario, landscape, S)
        elapsed = time.perf_counter() - t0
        seed_metrics_dfs.append(metrics_df)

        visit_count += visit.astype(int)
        envelope    = visit_count >= ENV_M
        new_cells   = int((envelope & ~prev_envelope).sum())
        new_frac    = new_cells / field_area
        env_frac    = envelope.sum() / field_area
        spatial_rows.append({
            "seed":           S,
            "envelope_cells": int(envelope.sum()),
            "envelope_frac":  env_frac,
            "new_cells":      new_cells,
            "new_frac":       new_frac,
        })

        tag = f"ran ({elapsed:5.1f}s)" if ran_now else "cached       "
        print(f"  seed {S:2d}  {tag}   env={env_frac*100:5.1f}%   new={new_frac*100:5.2f}%",
              flush=True)

        if S >= 2 and new_frac < NEW_THRESHOLD:
            consec_below += 1
            if consec_below >= CONSEC_BELOW:
                N_conv = S
                print(f"  CONVERGED at N = {S}", flush=True)
                break
        else:
            consec_below = 0
        prev_envelope = envelope.copy()

    if N_conv is None:
        print(f"  did NOT converge within {MAX_SEEDS} seeds — using N = {MAX_SEEDS}",
              flush=True)
    N_used = N_conv if N_conv else MAX_SEEDS

    per_seed = [four_metrics(m) for m in seed_metrics_dfs[:N_used]]
    val_df = pd.DataFrame(per_seed)
    val_df.insert(0, "seed", range(1, N_used + 1))

    pd.DataFrame(spatial_rows).to_csv(L_dir / "spatial_trace.csv", index=False)
    val_df.to_csv(L_dir / "validation_per_seed.csv", index=False)

    metric_keys = ["NND_active", "resting_pct", "daily_distance", "bimodal_pass"]
    return {
        "scenario":     scenario,
        "landscape":    landscape,
        "N_conv":       N_conv,
        "N_used":       N_used,
        "converged":    N_conv is not None,
        "envelope_pct": float(spatial_rows[-1]["envelope_frac"] * 100),
        "metrics": {
            k: {
                "mean": float(val_df[k].mean()),
                "sd":   float(val_df[k].std(ddof=1)),
            } for k in metric_keys
        },
    }





def write_summary_md(summaries: list[dict], out_path: Path) -> None:
    metric_keys = ["NND_active", "resting_pct", "daily_distance", "bimodal_pass"]
    metric_label = {
        "NND_active":     "NND_active (m)",
        "resting_pct":    "resting_pct (%)",
        "daily_distance": "daily_distance (m)",
        "bimodal_pass":   "bimodal_pass (frac)",
    }

    lines = []
    lines.append("# Stage 1 — Spatial-coverage convergence\n")
    lines.append(
        "Rule: 3% new-envelope cells / step over 2 consecutive seeds, M = 2 (cells visited by >= 2 seeds). Cap N = 18.\n"
    )
    lines.append("| Scenario | Landscape | N_conv | Converged? | Final envelope (% field) |")
    lines.append("|---|---|---|---|---|")
    for s in summaries:
        lines.append(
            f"| {s['scenario']} | L{s['landscape']:02d} | {s['N_used']} | "
            f"{'yes' if s['converged'] else '**NO**'} | {s['envelope_pct']:.1f}% |"
        )
    lines.append("")

    for sc in SCENARIOS:
        lines.append(f"\n## Stage 2 — Benchmarks at converged N ({sc})\n")
        lines.append(
            f"Mean +- SD of the four validation metrics across the spatially-converged seed set.\n"
        )
        sc_summaries = [s for s in summaries if s["scenario"] == sc]
        header = "| Metric | " + " | ".join(
            f"L{s['landscape']:02d} (N={s['N_used']})" for s in sc_summaries
        ) + " |"
        sep = "|" + "---|" * (len(sc_summaries) + 1)
        lines.append(header)
        lines.append(sep)
        for k in metric_keys:
            cells = []
            for s in sc_summaries:
                m = s["metrics"][k]["mean"]
                sd = s["metrics"][k]["sd"]
                if k == "daily_distance":
                    cells.append(f"{m:.0f} ± {sd:.0f}")
                else:
                    cells.append(f"{m:.3f} ± {sd:.3f}")
            lines.append(f"| {metric_label[k]} | " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Combined spatial-convergence + validation pipeline")
    print(f"  scenarios  : {SCENARIOS}")
    print(f"  landscapes : {LANDSCAPES}")
    print(f"  steps      : {STEPS}")
    print(f"  cache root (spatial)  : {SPATIAL_ROOT}")
    print(f"  cache root (metrics) :  {METRICS_ROOTS}")
    print(f"  output dir  : {ROOT}")

    t0 = time.perf_counter()
    summaries: list[dict] = []
    for sc in SCENARIOS:
        for L in LANDSCAPES:
            summaries.append(study(sc, L))
    elapsed = time.perf_counter() - t0
    print(f"\nAll done in {elapsed/60:.1f} min", flush=True)

    write_summary_md(summaries, ROOT / "summary_table.md")
    (ROOT / "summary.json").write_text(
        json.dumps({"summaries": summaries, "wall_time_min": elapsed / 60}, indent=2)
    )
    print("\n=== SUMMARY ===\n")
    print((ROOT / "summary_table.md").read_text())


if __name__ == "__main__":
    main()
