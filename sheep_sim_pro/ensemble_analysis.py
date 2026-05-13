from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings("ignore", category=UserWarning)

import sys
sys.path.insert(0, str(Path(__file__).parent))

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

STATE_COLOURS = {
    "grazing":    "#4CAF50",
    "walking":    "#26A69A",
    "travelling": "#1E88E5",
    "regrouping": "#FB8C00",
    "resting":    "#757575",
}
CMAP_HEAT   = "YlOrRd"
CMAP_FIELD  = "YlGn"
OUTPUT_DIR  = Path("outputs/ensemble")

def run_ensemble(
    n_seeds: int,
    landscape_seed: int,
    steps: int,
    scenario: str,
    output_dir: Path,
) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    for seed in range(1, n_seeds + 1):
        run_dir = output_dir / f"seed_{seed}"
        run_dir.mkdir(exist_ok=True)
        pos_path = run_dir / "positions.csv"
        grp_path = run_dir / "metrics.csv"
        fld_path = run_dir / "field_health.csv"

        if pos_path.exists() and grp_path.exists():
            print(f"  seed={seed}: loading cached CSVs")
            pos_df = pd.read_csv(pos_path)
            grp_df = pd.read_csv(grp_path)
            fld_df = pd.read_csv(fld_path)
            cfg = build_scenario_config(scenario, steps=steps,
                                        seed=seed, landscape_seed=landscape_seed)
            sim = SheepSimulation(cfg)
        else:
            print(f"  seed={seed}: running {steps} steps ... ", end="", flush=True)
            cfg = build_scenario_config(scenario, steps=steps,
                                        seed=seed, landscape_seed=landscape_seed)
            sim = SheepSimulation(cfg)
            result = sim.run(render="none", output_dir=run_dir)
            pos_df = result.metrics.position_dataframe()
            grp_df = result.metrics.group_dataframe()
            fld_df = result.metrics.field_dataframe()
            print("done")

        results.append({
            "seed":         seed,
            "cfg":          cfg,
            "sim":          sim,
            "pos_df":       pos_df,
            "grp_df":       grp_df,
            "fld_df":       fld_df,
            "field_width":  cfg.field.width,
            "field_height": cfg.field.height,
        })

    return results

def build_utilisation_grid(
    pos_df: pd.DataFrame,
    field_width: float,
    field_height: float,
    grid_cols: int = 88,
    grid_rows: int = 56,
) -> np.ndarray:
    xs = pos_df["x"].to_numpy()
    ys = pos_df["y"].to_numpy()
    col_idx = np.clip((xs / field_width  * grid_cols).astype(int), 0, grid_cols - 1)
    row_idx = np.clip((ys / field_height * grid_rows).astype(int), 0, grid_rows - 1)
    grid = np.zeros((grid_rows, grid_cols), dtype=float)
    np.add.at(grid, (row_idx, col_idx), 1)
    total = grid.sum()
    return grid / total if total > 0 else grid

def plot_heatmap(results: list[dict], output_dir: Path, smooth_sigma: float = 1.2) -> None:
    r0 = results[0]
    field_w = r0["field_width"]
    field_h = r0["field_height"]
    ndvi    = r0["sim"].food.ndvi

    combined = np.zeros((56, 88), dtype=float)
    for r in results:
        combined += build_utilisation_grid(r["pos_df"], field_w, field_h)
    combined /= len(results)
    combined_smooth = gaussian_filter(combined, sigma=smooth_sigma)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    ax.imshow(ndvi, origin="lower", extent=[0, field_w, 0, field_h],
              aspect="auto", cmap=CMAP_FIELD, alpha=0.45)
    im = ax.imshow(combined_smooth, origin="lower",
                   extent=[0, field_w, 0, field_h],
                   aspect="auto", cmap=CMAP_HEAT, alpha=0.75,
                   vmin=0, vmax=float(np.percentile(combined_smooth, 99)))
    plt.colorbar(im, ax=ax, label="Mean visitation density (normalised)")
    ax.set_title(f"Ensemble utilisation heatmap  ({len(results)} seeds, landscape={results[0]['cfg'].landscape_seed})",
                 fontsize=13)
    ax.set_xlabel("Field X (m)")
    ax.set_ylabel("Field Y (m)")

    ax2 = axes[1]
    ax2.imshow(ndvi, origin="lower", extent=[0, field_w, 0, field_h],
               aspect="auto", cmap=CMAP_FIELD, alpha=0.55)
    colours = plt.cm.tab10(np.linspace(0, 0.9, len(results)))
    xs_grid = np.linspace(0, field_w, 88)
    ys_grid = np.linspace(0, field_h, 56)
    for i, r in enumerate(results):
        g = gaussian_filter(
            build_utilisation_grid(r["pos_df"], field_w, field_h), sigma=smooth_sigma
        )
        threshold = float(np.percentile(g[g > 0], 60)) if (g > 0).any() else 0
        ax2.contour(xs_grid, ys_grid, g, levels=[threshold],
                    colors=[colours[i]], linewidths=1.2, alpha=0.85)
    handles = [mpatches.Patch(color=colours[i], label=f"seed={r['seed']}")
               for i, r in enumerate(results)]
    ax2.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.8)
    ax2.set_title("60th-percentile utilisation contour per seed", fontsize=13)
    ax2.set_xlabel("Field X (m)")
    ax2.set_ylabel("Field Y (m)")

    fig.tight_layout()
    path = output_dir / "heatmap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def plot_convergence(results: list[dict], output_dir: Path) -> None:
    field_w = results[0]["field_width"]
    field_h = results[0]["field_height"]

    grids = [
        build_utilisation_grid(r["pos_df"], field_w, field_h).ravel()
        for r in results
    ]

    jsd_values   = []
    coverage_pct = []
    eps = 1e-12

    cumulative = grids[0].copy()
    for i in range(1, len(grids)):
        prev_norm = cumulative / (cumulative.sum() + eps)
        cumulative = cumulative + grids[i]
        curr_norm  = cumulative / (cumulative.sum() + eps)
        jsd = float(jensenshannon(prev_norm, curr_norm, base=2))
        jsd_values.append(jsd)

        grid_2d = cumulative.reshape(56, 88) / (cumulative.reshape(56, 88).sum() + eps)
        threshold = float(np.percentile(grid_2d[grid_2d > 0], 10)) if (grid_2d > 0).any() else 0
        coverage  = float(np.mean(grid_2d > threshold)) * 100
        coverage_pct.append(coverage)

    n_seeds = list(range(2, len(results) + 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(n_seeds, jsd_values, "o-", color="#1E88E5", linewidth=2, markersize=7)
    ax.axhline(0.01, color="red", linestyle="--", linewidth=1.2, label="JSD = 0.01 threshold")
    ax.fill_between(n_seeds, 0, jsd_values, alpha=0.12, color="#1E88E5")
    ax.set_xlabel("Number of seeds in ensemble (N)", fontsize=12)
    ax.set_ylabel("Jensen-Shannon divergence\n(N-seed vs (N-1)-seed distribution)", fontsize=11)
    ax.set_title("Convergence: marginal information gain per new seed", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_xlim(2, len(results))
    ax.set_xticks(n_seeds)
    ax.grid(True, alpha=0.3)

    converged_at = next(
        (n for n, j in zip(n_seeds, jsd_values) if j < 0.01), None
    )
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle=":", linewidth=1.5,
                   label=f"Converged at N={converged_at}")
        ax.legend(fontsize=10)
        ax.text(converged_at + 0.05, max(jsd_values) * 0.85,
                f"N={converged_at}", color="green", fontsize=10)

    ax2 = axes[1]
    all_n = list(range(1, len(results) + 1))
    g1 = grids[0].reshape(56, 88)
    g1_norm = g1 / (g1.sum() + eps)
    thr1 = float(np.percentile(g1_norm[g1_norm > 0], 10)) if (g1_norm > 0).any() else 0
    cov1 = float(np.mean(g1_norm > thr1)) * 100
    all_coverage = [cov1] + coverage_pct

    ax2.plot(all_n, all_coverage, "s-", color="#4CAF50", linewidth=2, markersize=7)
    ax2.fill_between(all_n, 0, all_coverage, alpha=0.12, color="#4CAF50")
    ax2.set_xlabel("Number of seeds in ensemble (N)", fontsize=12)
    ax2.set_ylabel("Field coverage (% cells above 10th pct threshold)", fontsize=11)
    ax2.set_title("Spatial coverage growth with ensemble size", fontsize=12)
    ax2.set_xlim(1, len(results))
    ax2.set_xticks(all_n)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = output_dir / "convergence.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    if converged_at:
        print(f"\n  Convergence point: N = {converged_at} seeds "
              f"(JSD drops below 0.01 at this point)")
    else:
        print(f"\n  No convergence below JSD=0.01 — consider running more seeds")

def plot_behavioural_metrics(results: list[dict], output_dir: Path) -> None:
    summary_rows = []
    state_budget_rows = []

    for r in results:
        grp = r["grp_df"]
        final = grp.iloc[-1]
        n_sheep = r["cfg"].flock.n_sheep
        total_records = len(grp)

        summary_rows.append({
            "seed":           r["seed"],
            "spread":         final["spread"],
            "polarization":   final["polarization"],
            "mean_food_intake": final["mean_food_intake"],
            "degraded_area_pct": final["degraded_area_pct"],
            "field_biomass_pct": final["field_biomass_pct"],
            "mean_speed":     grp["mean_speed"].mean(),
            "n_clusters_mean": grp["number_of_clusters"].mean(),
        })

        for state in ["grazing", "walking", "travelling", "regrouping", "resting"]:
            col = f"{state}_count"
            if col in grp.columns:
                frac = (grp[col] / n_sheep).mean()
                state_budget_rows.append({
                    "seed": r["seed"], "state": state, "fraction": frac
                })

    summary_df = pd.DataFrame(summary_rows)
    budget_df  = pd.DataFrame(state_budget_rows)

    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    metrics = [
        ("spread",           "Final flock spread (m)"),
        ("polarization",     "Final polarisation [0–1]"),
        ("mean_food_intake", "Mean food intake (final step)"),
        ("degraded_area_pct","Degraded area (%)"),
        ("field_biomass_pct","Remaining biomass (%)"),
        ("mean_speed",       "Mean speed (m/step)"),
        ("n_clusters_mean",  "Mean number of clusters"),
    ]

    for idx, (col, label) in enumerate(metrics):
        row, c = divmod(idx, 4)
        ax = fig.add_subplot(gs[row, c])
        vals = summary_df[col].tolist()
        bp = ax.boxplot(vals, patch_artist=True,
                        boxprops=dict(facecolor="#B3D9F5", linewidth=1.2),
                        medianprops=dict(color="#1E88E5", linewidth=2),
                        whiskerprops=dict(linewidth=1.2),
                        capprops=dict(linewidth=1.2),
                        flierprops=dict(marker="o", markersize=5, color="#1E88E5"))
        ax.scatter([1] * len(vals), vals, color="#1565C0", zorder=3,
                   s=30, label="Per-seed value")
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_ylabel(label.split("(")[-1].replace(")", "") if "(" in label else "", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)

    ax_b = fig.add_subplot(gs[1, 3])
    state_list = ["grazing", "walking", "travelling", "regrouping", "resting"]
    means = [budget_df[budget_df.state == s]["fraction"].mean() * 100 for s in state_list]
    stds  = [budget_df[budget_df.state == s]["fraction"].std()  * 100 for s in state_list]
    colours = [STATE_COLOURS[s] for s in state_list]
    bars = ax_b.bar(state_list, means, yerr=stds, color=colours, edgecolor="black",
                    linewidth=0.7, capsize=4, error_kw={"linewidth": 1.2})
    ax_b.set_title("Mean state time budget (± 1 SD)", fontsize=10)
    ax_b.set_ylabel("% of steps", fontsize=9)
    ax_b.tick_params(axis="x", labelsize=8, rotation=30)
    ax_b.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Behavioural metrics across {len(results)} seeds  "
        f"(landscape={results[0]['cfg'].landscape_seed})",
        fontsize=13, y=1.01,
    )

    path = output_dir / "behavioural_metrics.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    csv_path = output_dir / "ensemble_summary.csv"
    summary_df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

def plot_depletion_overlap(results: list[dict], output_dir: Path) -> None:
    field_w = results[0]["field_width"]
    field_h = results[0]["field_height"]
    eps = 1e-12

    grids = []
    for r in results:
        g = build_utilisation_grid(r["pos_df"], field_w, field_h).ravel()
        g = g / (g.sum() + eps)
        grids.append(g)

    n = len(grids)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = float(np.sum(np.sqrt(grids[i] * grids[j])))

    labels = [f"seed={r['seed']}" for r in results]
    fig, ax = plt.subplots(figsize=(max(6, n), max(5, n - 1)))
    im = ax.imshow(matrix, cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Bhattacharyya coefficient\n(1 = identical, 0 = disjoint)")
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(n)); ax.set_yticklabels(labels)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black" if matrix[i, j] > 0.3 else "white")
    ax.set_title("Pairwise spatial overlap (Bhattacharyya coefficient)\nof utilisation distributions", fontsize=12)
    fig.tight_layout()
    path = output_dir / "depletion_overlap.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

    mean_off_diag = (matrix.sum() - np.trace(matrix)) / (n * (n - 1))
    print(f"  Mean pairwise Bhattacharyya coefficient: {mean_off_diag:.3f}")
    print(f"  (1.0 = all seeds use identical space; lower = more diversity across seeds)")

def plot_final_positions_grid(results: list[dict], output_dir: Path) -> None:
    n   = len(results)
    ncols = min(4, n)
    nrows = int(np.ceil(n / ncols))
    field_w = results[0]["field_width"]
    field_h = results[0]["field_height"]
    ndvi    = results[0]["sim"].food.ndvi

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5.5, nrows * 4.2))
    axes = np.array(axes).flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        pos = r["pos_df"]
        food_img = np.clip(
            0.8 * r["sim"].food.biomass + 0.15 * r["sim"].food.ndvi
            - 0.15 * (1.0 - r["sim"].food.health), 0.0, None,
        )
        ax.imshow(food_img, origin="lower", extent=[0, field_w, 0, field_h],
                  aspect="auto", cmap=CMAP_FIELD, alpha=0.88)

        final_step = pos["step"].max()
        final_pos  = pos[pos["step"] == final_step]
        for _, row in final_pos.iterrows():
            ax.scatter(row["x"], row["y"],
                       c=STATE_COLOURS.get(row["state"], "#999"),
                       s=28, edgecolors="black", linewidths=0.3, zorder=3)

        ax.set_title(f"seed={r['seed']}  (landscape={r['cfg'].landscape_seed})",
                     fontsize=10)
        ax.set_xlim(0, field_w); ax.set_ylim(0, field_h)
        ax.set_xlabel("Field X (m)", fontsize=8)
        ax.set_ylabel("Field Y (m)", fontsize=8)
        ax.tick_params(labelsize=7)

    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    legend_handles = [
        mpatches.Patch(color=STATE_COLOURS[s], label=s.capitalize())
        for s in STATE_COLOURS
    ]
    fig.legend(handles=legend_handles, loc="lower right",
               ncol=len(STATE_COLOURS), fontsize=9, framealpha=0.9)

    fig.suptitle(
        f"Final sheep positions — {n} seeds, landscape={results[0]['cfg'].landscape_seed}",
        fontsize=13, y=1.01,
    )
    fig.tight_layout()
    path = output_dir / "final_positions_grid.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ensemble analysis for sheep_sim")
    p.add_argument("--n-seeds",        type=int,   default=8,
                   help="Number of behaviour seeds to run (seeds 1..N)")
    p.add_argument("--landscape-seed", type=int,   default=7,
                   help="Fixed landscape RNG seed")
    p.add_argument("--steps",          type=int,   default=2500,
                   help="Simulation steps per run")
    p.add_argument("--scenario",       type=str,   default="abundant",
                   choices=["abundant", "scarce"])
    p.add_argument("--load-only",      action="store_true",
                   help="Skip running — load cached CSVs from previous run")
    return p.parse_args()

def main() -> None:
    args  = parse_args()
    out   = OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nEnsemble analysis")
    print(f"  scenario      : {args.scenario}")
    print(f"  landscape_seed: {args.landscape_seed}  (fixed)")
    print(f"  seeds         : 1 .. {args.n_seeds}")
    print(f"  steps         : {args.steps}")
    print(f"  output dir    : {out}\n")

    print("Running / loading simulations...")
    results = run_ensemble(
        n_seeds=args.n_seeds,
        landscape_seed=args.landscape_seed,
        steps=args.steps,
        scenario=args.scenario,
        output_dir=out,
    )

    print("\nGenerating plots...")
    print("  [1/5] Spatial heatmap")
    plot_heatmap(results, out)

    print("  [2/5] Convergence analysis")
    plot_convergence(results, out)

    print("  [3/5] Behavioural metrics")
    plot_behavioural_metrics(results, out)

    print("  [4/5] Depletion footprint overlap")
    plot_depletion_overlap(results, out)

    print("  [5/5] Final positions grid")
    plot_final_positions_grid(results, out)

    print(f"\nAll outputs written to {out}/")
    print("Files:")
    for f in sorted(out.glob("*.png")):
        print(f"  {f.name}")
    for f in sorted(out.glob("*.csv")):
        print(f"  {f.name}")

if __name__ == "__main__":
    main()
