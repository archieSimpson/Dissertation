"""Per-landscape converged validation metrics for Chapter 3 §3.6.

For abundant scenario across landscape seeds [7, 23, 41], runs behaviour
seeds 1..N (cap N at 18) and computes four validation metrics per seed:

  - NND_active: mean of mean_nearest_neighbour_distance over rows where
    resting_count < 26 (active-step proxy)
  - resting_pct: mean of (resting_count / 40) * 100 across all 1920 steps
  - daily_distance: mean_path_length at the final row (step 1919)
  - bimodal_pass: 1 if smoothed mean_speed (60-step centred rolling mean)
    has a local max in [1, 720) AND a local max in [961, 1919) AND the
    max value over [960, 1920) > max value over [0, 720); else 0

Convergence rule: starting from S=3, compare running mean at S vs S-1 for
all four metrics. Stop when ALL FOUR have rel_change < 2 % across TWO
consecutive seeds. Cap at S=18.

Outputs: per-landscape metrics_abundant_L<LL>_S<SS>.csv and trace_L<LL>.csv,
plus a summary.json and summary_table.md at the root.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE        = Path(__file__).resolve().parent
OUT_ROOT    = HERE / "outputs" / "chapter3_validation_scarce"
SCENARIO    = "scarce"
LANDSCAPES  = [7, 23, 41]
STEPS       = 1920
N_SHEEP     = 40
MAX_SEEDS   = 18
REL_THRESH  = 0.02
CONSEC_NEEDED = 2
EPS         = 1e-9
PYTHON      = sys.executable


def run_simulation(landscape: int, seed: int, run_dir: Path) -> Path:
    """Invoke `python -m sheep_sim` and return the path to metrics.csv."""
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON, "-m", "sheep_sim",
        "--scenario", SCENARIO,
        "--seed", str(seed),
        "--landscape-seed", str(landscape),
        "--steps", str(STEPS),
        "--render", "none",
        "--output-dir", str(run_dir),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
    if res.returncode != 0:
        raise RuntimeError(
            f"Simulation failed for L={landscape} S={seed}\n"
            f"--- stdout (tail) ---\n{res.stdout[-1000:]}\n"
            f"--- stderr (tail) ---\n{res.stderr[-1000:]}"
        )
    return run_dir / "metrics.csv"


def compute_metrics(metrics_csv: Path) -> dict[str, float]:
    df = pd.read_csv(metrics_csv)
    if len(df) != STEPS:
        print(f"  WARN: expected {STEPS} rows, got {len(df)} in {metrics_csv.name}",
              flush=True)

    active_mask = df["resting_count"] < 26
    if active_mask.any():
        nnd_active = float(df.loc[active_mask, "mean_nearest_neighbour_distance"].mean())
    else:
        nnd_active = float("nan")

    resting_pct = float((df["resting_count"] / N_SHEEP).mean() * 100.0)
    daily_dist  = float(df["mean_path_length"].iloc[-1])

    smoothed = (
        df["mean_speed"]
        .rolling(window=60, center=True, min_periods=1)
        .mean()
        .to_numpy()
    )
    n = len(smoothed)
    pre_hi  = min(720, n - 1)
    post_lo = min(961, n - 1)
    post_hi = n - 1

    pre_peaks  = [
        i for i in range(1, pre_hi)
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
    ]
    post_peaks = [
        i for i in range(post_lo, post_hi)
        if smoothed[i] > smoothed[i - 1] and smoothed[i] > smoothed[i + 1]
    ]
    pre_max  = float(smoothed[:pre_hi].max())  if pre_hi  > 0 else float("-inf")
    post_max = float(smoothed[960:n].max())    if n > 960  else float("-inf")
    bimodal  = 1 if (pre_peaks and post_peaks and post_max > pre_max) else 0

    return {
        "NND_active":     nnd_active,
        "resting_pct":    resting_pct,
        "daily_distance": daily_dist,
        "bimodal_pass":   float(bimodal),
    }


def relative_change(now: float, prev: float) -> float:
    if abs(prev) < EPS:
        return 0.0 if abs(now) < EPS else float("inf")
    return abs(now - prev) / abs(prev)


def study_landscape(landscape: int) -> dict:
    L_dir   = OUT_ROOT / f"L{landscape:02d}"
    tmp_dir = L_dir / "_tmp_run"
    L_dir.mkdir(parents=True, exist_ok=True)

    per_seed_metrics: list[dict] = []
    rolling_rows:     list[dict] = []
    rc_rows:          list[dict] = []
    consec_passes      = 0
    converged_at: int | None = None

    metric_keys = ["NND_active", "resting_pct", "daily_distance", "bimodal_pass"]

    for S in range(1, MAX_SEEDS + 1):
        t0 = time.perf_counter()
        print(f"[L{landscape:02d}]  seed {S:2d}: simulating ...", end="", flush=True)
        metrics_csv = run_simulation(landscape, S, tmp_dir)

        renamed = L_dir / f"metrics_{SCENARIO}_L{landscape:02d}_S{S:02d}.csv"
        shutil.move(str(metrics_csv), str(renamed))

        for f in ("positions.csv", "field_health.csv"):
            p = tmp_dir / f
            if p.exists():
                p.unlink()
        if tmp_dir.exists():
            try:
                shutil.rmtree(tmp_dir)
            except OSError:
                pass

        m = compute_metrics(renamed)
        per_seed_metrics.append(m)
        elapsed = time.perf_counter() - t0
        print(f"  done ({elapsed:5.1f}s)  "
              f"NND_active={m['NND_active']:.2f}  "
              f"resting_pct={m['resting_pct']:.2f}  "
              f"daily_distance={m['daily_distance']:.0f}  "
              f"bimodal_pass={int(m['bimodal_pass'])}", flush=True)

        df_so_far = pd.DataFrame(per_seed_metrics)
        rmean = {k: float(df_so_far[k].mean()) for k in metric_keys}
        rolling_rows.append({"seed": S, **{f"rmean_{k}": rmean[k] for k in metric_keys}})

        rc_row = {"seed": S, "passes": 0}
        if S >= 3:
            prev_rmean = {
                k: rolling_rows[-2][f"rmean_{k}"] for k in metric_keys
            }
            rc = {k: relative_change(rmean[k], prev_rmean[k]) for k in metric_keys}
            for k in metric_keys:
                rc_row[f"rc_{k}"] = rc[k]
            all_below = all(rc[k] < REL_THRESH for k in metric_keys)
            if all_below:
                consec_passes += 1
                rc_row["passes"] = 1
            else:
                consec_passes = 0
            print(f"             rel_change: " + "  ".join(
                f"{k}={rc[k]:.4f}{'*' if rc[k] < REL_THRESH else ' '}"
                for k in metric_keys
            ) + f"   consec={consec_passes}", flush=True)
        else:
            for k in metric_keys:
                rc_row[f"rc_{k}"] = float("nan")
        rc_rows.append(rc_row)

        if S >= 3 and consec_passes >= CONSEC_NEEDED:
            converged_at = S
            print(f"[L{landscape:02d}]  CONVERGED at N = {S}", flush=True)
            break

    if converged_at is None:
        print(f"[L{landscape:02d}]  did NOT converge within {MAX_SEEDS} seeds",
              flush=True)

    seed_df = pd.DataFrame(
        [{"seed": i + 1, **m} for i, m in enumerate(per_seed_metrics)]
    )
    rmean_df = pd.DataFrame(rolling_rows)
    rc_df    = pd.DataFrame(rc_rows)
    trace = seed_df.merge(rmean_df, on="seed").merge(rc_df, on="seed")
    trace["converged_here"] = (trace["seed"] == converged_at).astype(int) if converged_at else 0

    trace_path = L_dir / f"trace_L{landscape:02d}.csv"
    trace.to_csv(trace_path, index=False)
    print(f"[L{landscape:02d}]  trace -> {trace_path}", flush=True)

    N_used   = converged_at if converged_at else len(per_seed_metrics)
    used_df  = pd.DataFrame(per_seed_metrics[:N_used])
    summary  = {
        "landscape":     landscape,
        "N":             N_used,
        "converged":     converged_at is not None,
        "converged_at":  converged_at,
        "metrics":       {
            k: {
                "mean": float(used_df[k].mean()),
                "sd":   float(used_df[k].std(ddof=0)),
            }
            for k in metric_keys
        },
    }
    return summary


def write_summary_md(summaries: list[dict], out_path: Path) -> None:
    metric_keys = ["NND_active", "resting_pct", "daily_distance", "bimodal_pass"]
    metric_label = {
        "NND_active":     "NND_active (m)",
        "resting_pct":    "resting_pct (%)",
        "daily_distance": "daily_distance (m)",
        "bimodal_pass":   "bimodal_pass (frac)",
    }
    lines = []
    lines.append(f"# Validation metrics — {SCENARIO} scenario\n")
    lines.append("Per-landscape converged behaviour-seed counts:\n")
    for s in summaries:
        flag = "" if s["converged"] else "  **(NOT CONVERGED — capped at N=18)**"
        lines.append(f"- L{s['landscape']:02d}: N = {s['N']}{flag}")
    lines.append("")
    header = "| Metric | " + " | ".join(
        f"L{s['landscape']:02d} mean ± SD (N={s['N']})" for s in summaries
    ) + " |"
    sep    = "|" + "---|" * (len(summaries) + 1)
    lines.append(header)
    lines.append(sep)
    for k in metric_keys:
        cells = []
        for s in summaries:
            mean = s["metrics"][k]["mean"]
            sd   = s["metrics"][k]["sd"]
            if k == "daily_distance":
                cells.append(f"{mean:.0f} ± {sd:.0f}")
            elif k == "bimodal_pass":
                cells.append(f"{mean:.3f} ± {sd:.3f}")
            else:
                cells.append(f"{mean:.3f} ± {sd:.3f}")
        lines.append(f"| {metric_label[k]} | " + " | ".join(cells) + " |")
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 72, flush=True)
    print(f"Validation metrics — {SCENARIO} × {LANDSCAPES}", flush=True)
    print(f"  steps per run     : {STEPS}", flush=True)
    print(f"  max seeds per L   : {MAX_SEEDS}", flush=True)
    print(f"  rel-change thresh : {REL_THRESH * 100:.1f}%", flush=True)
    print(f"  consec required   : {CONSEC_NEEDED}", flush=True)
    print(f"  output dir        : {OUT_ROOT}", flush=True)
    print("=" * 72, flush=True)

    summaries = []
    t_all = time.perf_counter()
    for L in LANDSCAPES:
        print(f"\n{'─' * 72}\nLandscape seed = {L}\n{'─' * 72}", flush=True)
        summaries.append(study_landscape(L))

    elapsed = time.perf_counter() - t_all
    print(f"\nAll landscapes done in {elapsed/60:.1f} min", flush=True)

    summary_json = {"landscapes": summaries, "wall_time_min": elapsed / 60}
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary_json, indent=2))
    write_summary_md(summaries, OUT_ROOT / "summary_table.md")

    print("\n=== SUMMARY ===", flush=True)
    print((OUT_ROOT / "summary_table.md").read_text(), flush=True)


if __name__ == "__main__":
    main()
