from __future__ import annotations

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

from sheep_sim.agents import BehaviourState, SheepAgent
from sheep_sim.environment import FieldEnvironment
from sheep_sim.food import FoodLandscape
from sheep_sim.utils import polarization


STATE_COLOURS = {
    BehaviourState.GRAZING: "#4CAF50",
    BehaviourState.TRAVELLING: "#1E88E5",
    BehaviourState.REGROUPING: "#FB8C00",
    BehaviourState.RESTING: "#757575",
}

STATE_LABELS = {
    BehaviourState.GRAZING: "Grazing",
    BehaviourState.TRAVELLING: "Travelling",
    BehaviourState.REGROUPING: "Regrouping",
    BehaviourState.RESTING: "Resting",
}

STATE_DESCRIPTIONS = {
    BehaviourState.GRAZING: "Short steps, local patch use, high turning",
    BehaviourState.TRAVELLING: "Longer moves, active search or route switching",
    BehaviourState.REGROUPING: "Social pull dominates, flock cohesion increases",
    BehaviourState.RESTING: "Very low speed, circadian low-activity phase",
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
        self.ax_field = self.fig.add_subplot(gs[:, 0])
        self.ax_foodbar = self.fig.add_subplot(gs[:, 1])
        self.ax_panel = self.fig.add_subplot(gs[:, 2])
        self.ax_panel.axis("off")
        self.fig.subplots_adjust(wspace=0.08, hspace=0.05)

    def draw(self, flock: list[SheepAgent], step: int, scenario: str) -> None:
        self.ax_field.clear()
        self.ax_panel.clear()
        self.ax_panel.axis("off")

        # Composite background: greener areas reflect productive NDVI, redder overlay reflects degradation.
        food_img = self.food.biomass
        degraded = 1.0 - self.food.health
        composite = np.clip(0.78 * food_img + 0.12 * self.food.ndvi - 0.18 * degraded, 0.0, None)
        im = self.ax_field.imshow(
            composite,
            origin="lower",
            extent=[0, self.env.width, 0, self.env.height],
            aspect="auto",
            alpha=0.92,
            cmap="YlGn",
            vmin=float(np.min(composite)),
            vmax=float(np.max(composite) + 1e-9),
        )
        self._update_colorbar(im)
        self.ax_field.contour(
            self.food.x_coords,
            self.food.y_coords,
            degraded,
            levels=[0.20, 0.35, 0.50],
            colors=["#d95f02"],
            linewidths=[0.6],
            alpha=0.75,
        )

        terrain_contours = np.linspace(float(np.min(self.env.terrain)), float(np.max(self.env.terrain)), 6)
        xs = np.linspace(0, self.env.width, self.env.terrain.shape[1])
        ys = np.linspace(0, self.env.height, self.env.terrain.shape[0])
        self.ax_field.contour(xs, ys, self.env.terrain, levels=terrain_contours, linewidths=0.6, alpha=0.25, colors="black")

        positions = np.array([s.position for s in flock])
        velocities = np.array([s.velocity for s in flock])
        centroid = positions.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(positions - centroid, axis=1)))
        mean_speed = float(np.mean(np.linalg.norm(velocities, axis=1)))
        pol = polarization(velocities)
        counts = Counter(s.state for s in flock)

        colours = [STATE_COLOURS[s.state] for s in flock]
        sizes = [44 if s.state != BehaviourState.RESTING else 34 for s in flock]
        self.ax_field.scatter(positions[:, 0], positions[:, 1], c=colours, s=sizes, edgecolors="black", linewidths=0.35, zorder=3)
        self.ax_field.quiver(
            positions[:, 0],
            positions[:, 1],
            velocities[:, 0],
            velocities[:, 1],
            angles="xy",
            scale_units="xy",
            scale=1.7,
            width=0.0028,
            alpha=0.85,
            zorder=4,
        )
        self.ax_field.scatter([centroid[0]], [centroid[1]], marker="x", s=120, c="white", linewidths=2.0, zorder=5)
        circle = plt.Circle((centroid[0], centroid[1]), max(spread, 1.0), fill=False, linestyle="--", linewidth=1.4, color="white", alpha=0.7)
        self.ax_field.add_patch(circle)

        self.ax_field.set_xlim(0, self.env.width)
        self.ax_field.set_ylim(0, self.env.height)
        self.ax_field.set_xlabel("Field X")
        self.ax_field.set_ylabel("Field Y")
        self.ax_field.set_title(f"Sheep Simulation Dashboard | scenario={scenario} | step={step}", fontsize=14, pad=12)

        sim_clock = self._format_simulated_time(step)
        speedup = self.real_seconds_per_step * self.render_fps
        self.ax_field.text(
            0.015,
            0.985,
            f"Simulated time: {sim_clock}\n1 step = {self.real_seconds_per_step:.0f} s real time\nAnimation speed ≈ {speedup:.0f}x real time",
            transform=self.ax_field.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            color="white",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="black", alpha=0.55, edgecolor="white"),
        )

        self._draw_side_panel(counts, mean_speed, spread, pol, scenario, step)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(max(0.001, 1.0 / max(self.render_fps, 1)))

    def _update_colorbar(self, image) -> None:
        self.ax_foodbar.clear()
        cbar = self.fig.colorbar(image, cax=self.ax_foodbar)
        cbar.set_label("Usable biomass / field productivity", rotation=90)

    def _draw_side_panel(self, counts: Counter, mean_speed: float, spread: float, pol: float, scenario: str, step: int) -> None:
        speedup = self.real_seconds_per_step * self.render_fps
        field_health = self.food.health_percentage()
        field_biomass = self.food.biomass_percentage()
        degraded_area = self.food.degraded_area_percentage()
        mean_pressure = self.food.mean_pressure()
        mean_ndvi = 100.0 * float(np.mean(self.food.ndvi))

        y = 0.98
        line = 0.06

        self.ax_panel.text(0.0, y, "Legend / State key", fontsize=13, fontweight="bold", va="top")
        y -= line
        for state in BehaviourState:
            patch = mpatches.Rectangle((0.0, y - 0.025), 0.06, 0.03, facecolor=STATE_COLOURS[state], edgecolor="black")
            self.ax_panel.add_patch(patch)
            self.ax_panel.text(0.08, y, f"{STATE_LABELS[state]}: {STATE_DESCRIPTIONS[state]}", fontsize=9.6, va="center")
            y -= line

        y -= 0.01
        self.ax_panel.text(0.0, y, "Runtime summary", fontsize=13, fontweight="bold", va="top")
        y -= line
        summary_lines = [
            f"Scenario: {scenario}",
            f"Step: {step}",
            f"Mean speed: {mean_speed:.2f} field units / step",
            f"Flock spread: {spread:.2f}",
            f"Polarization: {pol:.2f}",
            f"Animation speed: ~{speedup:.0f}x real time",
        ]
        for text in summary_lines:
            self.ax_panel.text(0.0, y, text, fontsize=10.2, va="center")
            y -= line * 0.82

        y -= 0.02
        self.ax_panel.text(0.0, y, "Field condition", fontsize=13, fontweight="bold", va="top")
        y -= line
        for label, value, color in [
            ("Grass field health", field_health, "#2E7D32"),
            ("Remaining biomass", field_biomass, "#66BB6A"),
            ("Mean NDVI/productivity", mean_ndvi, "#43A047"),
            ("Degraded area", degraded_area, "#D95F02"),
        ]:
            width = 0.62 * np.clip(value / 100.0, 0.0, 1.0)
            self.ax_panel.add_patch(mpatches.Rectangle((0.0, y - 0.02), width, 0.028, facecolor=color, edgecolor="none"))
            self.ax_panel.add_patch(mpatches.Rectangle((0.0, y - 0.02), 0.62, 0.028, facecolor="none", edgecolor="#999999", linewidth=0.8))
            self.ax_panel.text(0.66, y - 0.005, f"{label}: {value:.1f}%", fontsize=9.8, va="center")
            y -= line * 0.88

        self.ax_panel.text(0.0, y, f"Mean grazing pressure: {mean_pressure:.3f}", fontsize=10, va="center")
        y -= 0.06

        self.ax_panel.text(0.0, y, "Sheep per state", fontsize=13, fontweight="bold", va="top")
        y -= line
        total = max(sum(counts.values()), 1)
        for state in BehaviourState:
            count = counts.get(state, 0)
            frac = count / total
            self.ax_panel.add_patch(mpatches.Rectangle((0.0, y - 0.02), 0.62 * frac, 0.028, facecolor=STATE_COLOURS[state], edgecolor="none"))
            self.ax_panel.add_patch(mpatches.Rectangle((0.0, y - 0.02), 0.62, 0.028, facecolor="none", edgecolor="#999999", linewidth=0.8))
            self.ax_panel.text(0.66, y - 0.005, f"{STATE_LABELS[state]}: {count}", fontsize=9.8, va="center")
            y -= line * 0.88

        y -= 0.015
        self.ax_panel.text(0.0, y, "How to read the animation", fontsize=13, fontweight="bold", va="top")
        y -= line
        notes = [
            "Green agents lingering on high-biomass cells indicate patch residence in rich grass.",
            "Blue, straighter paths indicate search/travel; stronger and longer in scarce fields.",
            "Orange surges show regrouping when isolation or spread increases.",
            "Orange/red contour zones highlight degraded pasture where health has fallen.",
            "Grass field health declines under repeated grazing and only recovers slowly over simulated days.",
        ]
        for note in notes:
            self.ax_panel.text(0.0, y, f"• {note}", fontsize=9.4, va="top", wrap=True)
            y -= 0.072

        self.ax_panel.set_xlim(0, 1)
        self.ax_panel.set_ylim(0, 1)

    def _format_simulated_time(self, step: int) -> str:
        total_seconds = self.start_seconds + int(step * self.real_seconds_per_step)
        days = total_seconds // 86400
        within_day = total_seconds % 86400
        hour = within_day // 3600
        minute = (within_day % 3600) // 60
        second = within_day % 60
        return f"Day {days + 1} | {hour:02d}:{minute:02d}:{second:02d}"

    def close(self) -> None:
        plt.ioff()
        plt.close(self.fig)



def save_final_frame(
    env: FieldEnvironment,
    food: FoodLandscape,
    flock: list[SheepAgent],
    out_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 8))
    field_img = np.clip(0.8 * food.biomass + 0.15 * food.ndvi - 0.15 * (1.0 - food.health), 0.0, None)
    ax.imshow(
        field_img,
        origin="lower",
        extent=[0, env.width, 0, env.height],
        aspect="auto",
        alpha=0.90,
        cmap="YlGn",
    )
    ax.contour(food.x_coords, food.y_coords, 1.0 - food.health, levels=[0.2, 0.35, 0.5], colors=["#d95f02"], linewidths=0.8)
    xs = [s.position[0] for s in flock]
    ys = [s.position[1] for s in flock]
    colours = [STATE_COLOURS[s.state] for s in flock]
    ax.scatter(xs, ys, c=colours, s=34, edgecolors="black", linewidths=0.3)
    ax.set_xlim(0, env.width)
    ax.set_ylim(0, env.height)
    ax.set_title(title)
    ax.set_xlabel("Field X")
    ax.set_ylabel("Field Y")
    legend_handles = [mpatches.Patch(color=STATE_COLOURS[s], label=STATE_LABELS[s]) for s in BehaviourState]
    legend_handles.append(mpatches.Patch(color="#d95f02", label="Degradation contours"))
    ax.legend(handles=legend_handles, loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
