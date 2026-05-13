"""Clean (no axis labels, no title, no colour-key panel) variant of the
extract_memory_map.py figure. Same simulation setup (abundant, seed=1,
landscape_seed=7, 2500 steps), same memory-map cmap, same trajectory and
u_mem arrow. Differences:

  - single panel (the right-side colour-key explanation panel is removed)
  - no axis title, no x/y tick labels, no axis labels
  - "Memory value" colorbar label larger and bolder
  - highest-memory-cell star marker enlarged with a white ring around it
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

SCENARIO     = "abundant"
STEPS        = 2500
SEEDS        = [1, 2, 3]
LANDSCAPE    = 23
AGENT_INDEX  = None
CAPTURE_STEP = 2499
OUTPUT_DIR   = Path("outputs/memory_map")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CMAP = mcolors.LinearSegmentedColormap.from_list("sheep_memory", [
    (0.00, "#1a1a18"),
    (0.05, "#0c2a45"),
    (0.18, "#1a5f7a"),
    (0.40, "#1d9e75"),
    (0.65, "#d4850a"),
    (0.85, "#e85d24"),
    (1.00, "#fcd34d"),
])


def run_and_render(seed: int) -> dict:
    print(f"\nRunning {SCENARIO} for {STEPS} steps (seed={seed}) ...")
    cfg = build_scenario_config(SCENARIO, steps=STEPS, seed=seed, landscape_seed=LANDSCAPE)
    sim = SheepSimulation(cfg)

    captured_maps: dict = {}
    captured_pos:  dict = {}
    all_positions: list = []

    _stdout = sys.stdout
    sys.stdout = io.StringIO()
    for step in range(STEPS):
        sim.step(step)
        sim.metrics.record(step, sim.flock, SCENARIO, sim.food)
        if step % 10 == 0:
            for sh in sim.flock:
                all_positions.append((sh.sheep_id, float(sh.position[0]), float(sh.position[1])))
        if step == CAPTURE_STEP:
            captured_maps[step] = {sh.sheep_id: sh.memory_map.copy() for sh in sim.flock}
            captured_pos[step]  = {sh.sheep_id: sh.position.copy()   for sh in sim.flock}
    sys.stdout = _stdout

    maps_at = captured_maps[CAPTURE_STEP]
    pos_at  = captured_pos[CAPTURE_STEP]

    best_id = AGENT_INDEX if AGENT_INDEX is not None else max(maps_at, key=lambda a: maps_at[a].max())
    memory  = maps_at[best_id]
    pos     = pos_at[best_id]

    flat_idx = np.argmax(memory)
    best_row = flat_idx // memory.shape[1]
    best_col = flat_idx %  memory.shape[1]
    cell_w   = cfg.field.width  / memory.shape[1]
    cell_h   = cfg.field.height / memory.shape[0]
    best_x   = (best_col + 0.5) * cell_w
    best_y   = (best_row + 0.5) * cell_h
    agent_x  = float(pos[0])
    agent_y  = float(pos[1])

    dx, dy = best_x - agent_x, best_y - agent_y
    dist   = float(np.sqrt(dx * dx + dy * dy))
    ux, uy = (dx / dist, dy / dist) if dist > 1e-9 else (0.0, 0.0)

    traj_x = [p[1] for p in all_positions if p[0] == best_id]
    traj_y = [p[2] for p in all_positions if p[0] == best_id]

    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=180)
    fig.patch.set_facecolor("#0d0d0b")
    ax.set_facecolor("#0d0d0b")

    extent = [0, cfg.field.width, cfg.field.height, 0]
    im = ax.imshow(
        memory, cmap=CMAP, vmin=0.0, vmax=max(memory.max(), 0.01),
        extent=extent, interpolation="nearest", aspect="auto", origin="upper",
    )

    ax.plot(traj_x, traj_y, color="white", alpha=0.12, linewidth=0.6, zorder=2)
    ax.scatter(agent_x, agent_y, s=70, c="white", edgecolors="#888",
               linewidths=0.8, zorder=5)

    arrow_len = min(dist * 0.80, 30.0)
    ax.annotate("", xy=(agent_x + ux*arrow_len, agent_y + uy*arrow_len),
                xytext=(agent_x, agent_y),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=1.4,
                                linestyle="dashed", mutation_scale=12), zorder=6)

    ax.scatter(best_x, best_y, s=320, marker="*",
               c="#fcd34d", edgecolors="white", linewidths=2.4, zorder=7)
    ring = plt.Circle((best_x, best_y), 3.6, fill=False,
                      edgecolor="white", linewidth=1.6, alpha=0.9, zorder=6)
    ax.add_patch(ring)

    for xg in range(0, int(cfg.field.width)  + 1, 20):
        ax.axvline(xg, color="white", alpha=0.08, linewidth=0.4)
    for yg in range(0, int(cfg.field.height) + 1, 20):
        ax.axhline(yg, color="white", alpha=0.08, linewidth=0.4)

    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    cbar = plt.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cbar.set_label("Memory value", color="#ddd", fontsize=16,
                   fontweight="bold", labelpad=10)
    cbar.ax.tick_params(colors="#aaa", labelsize=10)
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="#aaa")

    plt.tight_layout(pad=0.4)
    png_path = OUTPUT_DIR / f"figure3a_memory_map_clean_L{LANDSCAPE:02d}_seed{seed:02d}.png"
    pdf_path = OUTPUT_DIR / f"figure3a_memory_map_clean_L{LANDSCAPE:02d}_seed{seed:02d}.pdf"
    plt.savefig(png_path, dpi=180, bbox_inches="tight", facecolor="#0d0d0b")
    plt.savefig(pdf_path,            bbox_inches="tight", facecolor="#0d0d0b")
    plt.close()

    return {
        "seed":     seed,
        "agent_id": best_id,
        "agent":    (agent_x, agent_y),
        "best":     (best_x, best_y),
        "peak":     float(memory.max()),
        "mean":     float(memory.mean()),
        "coverage": float((memory > 0.01).mean() * 100),
        "png":      png_path,
    }


results = [run_and_render(s) for s in SEEDS]

print("\n=== summary ===")
print(f"{'seed':>4}  {'agent':>5}  {'peak':>7}  {'mean':>7}  {'cov%':>5}  png")
for r in results:
    print(f"{r['seed']:>4d}  {r['agent_id']:>5d}  "
          f"{r['peak']:>7.3f}  {r['mean']:>7.4f}  {r['coverage']:>5.1f}  {r['png'].name}")
