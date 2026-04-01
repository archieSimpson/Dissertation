from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

from sheep_sim.core.agents import BehaviourState, SheepAgent
from sheep_sim.environment.field import FieldEnvironment
from sheep_sim.environment.food import FoodLandscape
from sheep_sim.rendering.live import STATE_COLOURS, STATE_LABELS

def save_final_frame(
    env: FieldEnvironment,
    food: FoodLandscape,
    flock: list[SheepAgent],
    out_path: Path,
    title: str,
    dpi: int = 170,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))

    field_img = np.clip(
        0.80 * food.biomass + 0.15 * food.ndvi - 0.15 * (1.0 - food.health), 0.0, None
    )
    ax.imshow(field_img, origin="lower", extent=[0, env.width, 0, env.height],
              aspect="auto", alpha=0.90, cmap="YlGn")
    ax.contour(food.x_coords, food.y_coords, 1.0 - food.health,
               levels=[0.2, 0.35, 0.5], colors=["#d95f02"], linewidths=0.8)

    xs = [s.position[0] for s in flock]
    ys = [s.position[1] for s in flock]
    cs = [STATE_COLOURS[s.state] for s in flock]
    ax.scatter(xs, ys, c=cs, s=34, edgecolors="black", linewidths=0.3)

    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Field X (m)")
    ax.set_ylabel("Field Y (m)")

    handles = [mpatches.Patch(color=STATE_COLOURS[s], label=STATE_LABELS[s]) for s in BehaviourState]
    handles.append(mpatches.Patch(color="#d95f02", label="Degradation contours"))
    ax.legend(handles=handles, loc="upper right", frameon=True, fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[renderer] Saved final frame → {out_path}")
