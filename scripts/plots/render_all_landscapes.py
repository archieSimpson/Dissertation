from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
from pathlib import Path
import io, sys

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation


LANDSCAPE_SEED = 7
OUTPUT_DIR     = Path("outputs/landscapes")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = [
    ("abundant",      "Abundant",      "18 patches, full biomass — resource-rich baseline"),
    ("scarce",        "Scarce",        "8 sparse patches, 58% biomass — resource-limited"),
    ("uniform_low",   "Uniform low",   "Homogeneous NDVI = 0.1 — bare pasture"),
    ("uniform_high",  "Uniform high",  "Homogeneous NDVI = 1.0 — uniformly rich"),
    ("radial_increase",  "Radial increase",  "Poor centre, quality increases outward"),
    ("ring",          "Ring",          "Medium centre, poor gap, rich outer ring"),
]


def build_landscape(scenario_name: str):
    cfg = build_scenario_config(scenario_name, steps=1, seed=1,
                                landscape_seed=LANDSCAPE_SEED)
    sim = SheepSimulation(cfg)
    old = sys.stdout; sys.stdout = io.StringIO()
    sim.step(0)
    sys.stdout = old
    return sim.food, sim.environment

def make_field_img(food):
    img = np.clip(
        0.80 * food.biomass + 0.15 * food.ndvi - 0.15 * (1.0 - food.health),
        0.0, None
    )
    return img / img.max() if img.max() > 0 else img

print("Generating individual landscape figures...")
for i, (name, label, desc) in enumerate(SCENARIOS):
    print(f"  {label}...", end=" ")
    food, env = build_landscape(name)
    field_img = make_field_img(food)

    fig, ax = plt.subplots(figsize=(8, 5.2), dpi=180)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    im = ax.imshow(
        field_img,
        origin="lower",
        extent=[0, env.width, 0, env.height],
        aspect="auto",
        cmap="YlGn",
        vmin=0.0, vmax=0.95,
        alpha=0.95,
    )

    ax.contour(
        food.x_coords, food.y_coords,
        1.0 - food.health,
        levels=[0.1, 0.25, 0.4],
        colors=["#d95f02"],
        linewidths=0.7,
        alpha=0.6,
    )

    for xg in range(0, int(env.width) + 1, 20):
        ax.axvline(xg, color="#aaa", linewidth=0.3, alpha=0.4)
    for yg in range(0, int(env.height) + 1, 20):
        ax.axhline(yg, color="#aaa", linewidth=0.3, alpha=0.4)

    cbar = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Vegetation quality", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_xlabel("Field x-coordinate (m)", fontsize=9)
    ax.set_ylabel("Field y-coordinate (m)", fontsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
    ax.tick_params(labelsize=7)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)

    contour_line = mlines.Line2D([], [], color="#d95f02", linewidth=0.9,
                                  label="Degradation contours")
    ax.legend(handles=[contour_line], loc="upper right", fontsize=7, framealpha=0.8)


    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"{name}_landscape.png", dpi=180,
                bbox_inches="tight", facecolor="white")
    plt.savefig(OUTPUT_DIR / f"{name}_landscape.pdf",
                bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved.")


print("\nGenerating combined 2×3 panel figure...")
fig, axes = plt.subplots(2, 3, figsize=(16, 9), dpi=150)
fig.patch.set_facecolor("white")

panel_labels = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"]

for idx, (name, label, desc) in enumerate(SCENARIOS):
    row, col = divmod(idx, 3)
    ax = axes[row, col]
    food, env = build_landscape(name)
    field_img = make_field_img(food)

    im = ax.imshow(
        field_img,
        origin="lower",
        extent=[0, env.width, 0, env.height],
        aspect="auto",
        cmap="YlGn",
        vmin=0.0, vmax=0.95,
        alpha=0.95,
    )

    ax.contour(
        food.x_coords, food.y_coords,
        1.0 - food.health,
        levels=[0.15, 0.35],
        colors=["#d95f02"],
        linewidths=0.6,
        alpha=0.5,
    )

    ax.set_title(f"{panel_labels[idx]} {label}", fontsize=9,
                 fontweight="bold", loc="left", pad=3)
    ax.set_xlabel("x (m)", fontsize=7)
    ax.set_ylabel("y (m)", fontsize=7)
    ax.tick_params(labelsize=6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(40))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(40))
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)

    ax.text(0.02, 0.04, desc, transform=ax.transAxes,
            fontsize=6.5, color="#555",
            bbox=dict(facecolor="white", edgecolor="#ccc",
                      boxstyle="round,pad=0.2", linewidth=0.4, alpha=0.85))

fig.suptitle(
    f"Figure 2.x  Initial vegetation landscapes for all six scenarios "
    f"(landscape seed {LANDSCAPE_SEED})",
    fontsize=10, y=0.98, color="#333"
)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUTPUT_DIR / "all_landscapes_panel.png", dpi=150,
            bbox_inches="tight", facecolor="white")
plt.savefig(OUTPUT_DIR / "all_landscapes_panel.pdf",
            bbox_inches="tight", facecolor="white")
plt.close()
print("Combined panel saved.")

print(f"\nAll outputs saved to {OUTPUT_DIR}/")
print("\nFiles produced:")
for f in sorted(OUTPUT_DIR.iterdir()):
    print(f"  {f.name}")
