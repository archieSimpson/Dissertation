from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

from sheep_sim.core.agents import BehaviourState, SheepAgent
from sheep_sim.core.utils import polarization
from sheep_sim.environment.field import FieldEnvironment
from sheep_sim.environment.food import FoodLandscape

if TYPE_CHECKING:
    pass

STATE_COLOURS = {
    BehaviourState.GRAZING:    "#4CAF50",
    BehaviourState.WALKING:    "#26A69A",
    BehaviourState.TRAVELLING: "#1E88E5",
    BehaviourState.REGROUPING: "#FB8C00",
    BehaviourState.RESTING:    "#757575",
}

STATE_LABELS = {
    BehaviourState.GRAZING:    "Grazing",
    BehaviourState.WALKING:    "Walking",
    BehaviourState.TRAVELLING: "Travelling",
    BehaviourState.REGROUPING: "Regrouping",
    BehaviourState.RESTING:    "Resting",
}

STATE_DESCRIPTIONS = {
    BehaviourState.GRAZING:    "Area-restricted search; short steps, high turning",
    BehaviourState.WALKING:    "Routine relocation between nearby productive patches",
    BehaviourState.TRAVELLING: "Extended search; long moves, low turning, memory-guided",
    BehaviourState.REGROUPING: "Social pull dominates; centripetal cohesion surge",
    BehaviourState.RESTING:    "Circadian low-activity phase; near-zero displacement",
}

class LiveRenderer:

    def __init__(
        self,
        env: FieldEnvironment,
        food: FoodLandscape,
        real_seconds_per_step: float,
        render_fps: int,
        start_hour: int = 6,
        start_minute: int = 0,
    ) -> None:
        self.env = env
        self.food = food
        self.real_seconds_per_step = real_seconds_per_step
        self.render_fps = render_fps
        self.start_seconds = start_hour * 3600 + start_minute * 60

        plt.ion()
        self.fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 3, figure=self.fig, width_ratios=[3.5, 0.15, 1.6], height_ratios=[1.0, 1.0])
        self.ax_field  = self.fig.add_subplot(gs[:, 0])
        self.ax_bar    = self.fig.add_subplot(gs[:, 1])
        self.ax_panel  = self.fig.add_subplot(gs[:, 2])
        self.ax_panel.axis("off")
        self.fig.subplots_adjust(wspace=0.08, hspace=0.05)
        self._cbar = None

    def draw(self, flock: list[SheepAgent], step: int, scenario: str) -> None:
        self.ax_field.clear()
        self.ax_panel.clear()
        self.ax_panel.axis("off")

        composite = np.clip(
            0.78 * self.food.biomass + 0.12 * self.food.ndvi - 0.18 * (1.0 - self.food.health),
            0.0, None,
        )
        im = self.ax_field.imshow(
            composite, origin="lower",
            extent=[0, self.env.width, 0, self.env.height],
            aspect="auto", alpha=0.92, cmap="YlGn",
            vmin=float(np.min(composite)), vmax=float(np.max(composite) + 1e-9),
        )
        self._refresh_colorbar(im)

        self.ax_field.contour(
            self.food.x_coords, self.food.y_coords, 1.0 - self.food.health,
            levels=[0.20, 0.35, 0.50], colors=["#d95f02"], linewidths=[0.6], alpha=0.75,
        )

        xs_t = np.linspace(0, self.env.width, self.env.terrain.shape[1])
        ys_t = np.linspace(0, self.env.height, self.env.terrain.shape[0])
        self.ax_field.contour(xs_t, ys_t, self.env.terrain, levels=6, linewidths=0.6, alpha=0.25, colors="black")

        positions  = np.array([s.position for s in flock])
        velocities = np.array([s.velocity for s in flock])
        centroid   = positions.mean(axis=0)
        spread     = float(np.mean(np.linalg.norm(positions - centroid, axis=1)))
        colours    = [STATE_COLOURS[s.state] for s in flock]
        sizes      = [44 if s.state != BehaviourState.RESTING else 34 for s in flock]

        self.ax_field.scatter(positions[:, 0], positions[:, 1], c=colours, s=sizes,
                              edgecolors="black", linewidths=0.35, zorder=3)
        self.ax_field.quiver(positions[:, 0], positions[:, 1],
                             velocities[:, 0], velocities[:, 1],
                             angles="xy", scale_units="xy", scale=1.7, width=0.0028, alpha=0.85, zorder=4)
        self.ax_field.scatter([centroid[0]], [centroid[1]], marker="x", s=120, c="white", linewidths=2.0, zorder=5)
        self.ax_field.add_patch(plt.Circle(
            (centroid[0], centroid[1]), max(spread, 1.0),
            fill=False, linestyle="--", linewidth=1.4, color="white", alpha=0.7,
        ))
        self.ax_field.set_xlim(0, self.env.width)
        self.ax_field.set_ylim(0, self.env.height)
        self.ax_field.set_xlabel("Field X (m)")
        self.ax_field.set_ylabel("Field Y (m)")
        self.ax_field.set_title(
            f"Sheep Simulation | scenario={scenario} | step={step}", fontsize=13, pad=10
        )
        clock   = self._clock(step)
        speedup = self.real_seconds_per_step * self.render_fps
        self.ax_field.text(
            0.015, 0.985,
            f"{clock}\n1 step = {self.real_seconds_per_step:.0f} s  |  ≈{speedup:.0f}x real time",
            transform=self.ax_field.transAxes, va="top", ha="left", fontsize=10, color="white",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.55, edgecolor="white"),
        )

        self._draw_panel(
            Counter(s.state for s in flock),
            float(np.mean(np.linalg.norm(velocities, axis=1))),
            spread, polarization(velocities), scenario, step,
        )

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(max(0.001, 1.0 / max(self.render_fps, 1)))

    def _refresh_colorbar(self, image) -> None:
        self.ax_bar.clear()
        self.fig.colorbar(image, cax=self.ax_bar).set_label("Usable biomass / productivity", rotation=90)

    def _draw_panel(self, counts: Counter, mean_speed: float, spread: float,
                    pol: float, scenario: str, step: int) -> None:
        speedup = self.real_seconds_per_step * self.render_fps
        y, line = 0.98, 0.062

        self.ax_panel.text(0.0, y, "State key", fontsize=13, fontweight="bold", va="top")
        y -= line
        for state in BehaviourState:
            patch = mpatches.Rectangle((0.0, y - 0.022), 0.055, 0.028, facecolor=STATE_COLOURS[state], edgecolor="black")
            self.ax_panel.add_patch(patch)
            self.ax_panel.text(0.075, y, f"{STATE_LABELS[state]} — {STATE_DESCRIPTIONS[state]}", fontsize=8.8, va="center")
            y -= line

        y -= 0.01
        self.ax_panel.text(0.0, y, "Runtime summary", fontsize=13, fontweight="bold", va="top")
        y -= line
        for txt in [f"Scenario: {scenario}", f"Step: {step}",
                    f"Mean speed: {mean_speed:.2f}", f"Flock spread: {spread:.2f}",
                    f"Polarisation: {pol:.2f}", f"Speedup: ~{speedup:.0f}x"]:
            self.ax_panel.text(0.0, y, txt, fontsize=9.8, va="center")
            y -= line * 0.80

        y -= 0.015
        self.ax_panel.text(0.0, y, "Field condition", fontsize=13, fontweight="bold", va="top")
        y -= line
        fh = self.food.health_percentage()
        fb = self.food.biomass_percentage()
        fd = self.food.degraded_area_percentage()
        mn = 100.0 * float(np.mean(self.food.ndvi))
        for label, val, colour in [
            ("Field health", fh, "#2E7D32"),
            ("Biomass", fb, "#66BB6A"),
            ("NDVI", mn, "#43A047"),
            ("Degraded area", fd, "#D95F02"),
        ]:
            w = 0.62 * np.clip(val / 100.0, 0.0, 1.0)
            self.ax_panel.add_patch(mpatches.Rectangle((0.0, y - 0.018), w, 0.024, facecolor=colour, edgecolor="none"))
            self.ax_panel.add_patch(mpatches.Rectangle((0.0, y - 0.018), 0.62, 0.024, facecolor="none", edgecolor="#999", linewidth=0.8))
            self.ax_panel.text(0.65, y - 0.005, f"{label}: {val:.1f}%", fontsize=9.2, va="center")
            y -= line * 0.86

        y -= 0.01
        self.ax_panel.text(0.0, y, "Sheep per state", fontsize=13, fontweight="bold", va="top")
        y -= line
        total = max(sum(counts.values()), 1)
        for state in BehaviourState:
            n = counts.get(state, 0)
            self.ax_panel.add_patch(mpatches.Rectangle((0.0, y - 0.018), 0.62 * (n / total), 0.024, facecolor=STATE_COLOURS[state], edgecolor="none"))
            self.ax_panel.add_patch(mpatches.Rectangle((0.0, y - 0.018), 0.62, 0.024, facecolor="none", edgecolor="#999", linewidth=0.8))
            self.ax_panel.text(0.65, y - 0.005, f"{STATE_LABELS[state]}: {n}", fontsize=9.2, va="center")
            y -= line * 0.86

        self.ax_panel.set_xlim(0, 1)
        self.ax_panel.set_ylim(0, 1)

    def _clock(self, step: int) -> str:
        total = self.start_seconds + int(step * self.real_seconds_per_step)
        d = total // 86400
        h = (total % 86400) // 3600
        m = (total % 3600) // 60
        s = total % 60
        return f"Day {d + 1}  {h:02d}:{m:02d}:{s:02d}"

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)
