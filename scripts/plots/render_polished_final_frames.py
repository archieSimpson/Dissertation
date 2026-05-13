from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from sheep_sim.core.agents import BehaviourState
from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation
from sheep_sim.rendering.live import STATE_COLOURS, STATE_LABELS

ROOT  = Path("outputs/final_frames")
STEPS = 1920

RUNS = [
    ("abundant",        "abundant",        7,  42),
    ("scarce",          "scarce",          7,  42),
    ("uniform_low",     "uniform_low",     7,  42),
    ("uniform_high",    "uniform_high",    7,  42),
    ("radial_increase", "radial_increase", 7,  42),
    ("ring",            "ring",            7,  42),
    ("corridors",       "corridors",       7,  42),
    ("corridors_L41",   "corridors",       41, 42),
]


def render_polished(env, food, flock, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    field_img = np.clip(
        0.80 * food.biomass + 0.15 * food.ndvi - 0.15 * (1.0 - food.health),
        0.0, 1.0,
    )
    ax.imshow(
        field_img, origin="lower", extent=[0, env.width, 0, env.height],
        aspect="auto", alpha=0.90, cmap="YlGn", vmin=0.0, vmax=1.0,
    )
    ax.contour(
        food.x_coords, food.y_coords, 1.0 - food.health,
        levels=[0.2, 0.35, 0.5], colors=["#d95f02"], linewidths=0.8,
    )

    xs = [s.position[0] for s in flock]
    ys = [s.position[1] for s in flock]
    cs = [STATE_COLOURS[s.state] for s in flock]
    ax.scatter(xs, ys, c=cs, s=34, edgecolors="black", linewidths=0.3)

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_xlabel("Field X (m)", fontsize=16)
    ax.set_ylabel("Field Y (m)", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)

    handles = [mpatches.Patch(color=STATE_COLOURS[s], label=STATE_LABELS[s])
               for s in BehaviourState]
    handles.append(mpatches.Patch(color="#d95f02", label="Degradation contours"))
    ax.legend(handles=handles, loc="upper right", frameon=True,
              fontsize=14, framealpha=0.92)

    fig.tight_layout(pad=0.4)
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


for out_name, scenario, L, S in RUNS:
    out_dir = ROOT / out_name
    if not out_dir.exists():
        print(f"  skip {out_name} (no output dir)")
        continue
    cfg = build_scenario_config(scenario, steps=STEPS, seed=S, landscape_seed=L)
    sim = SheepSimulation(cfg)
    for step in range(STEPS):
        sim.step(step)
    out_path = out_dir / "final_frame_polished.png"
    render_polished(sim.environment, sim.food, sim.flock, out_path)
    print(f"  saved {out_path}")
