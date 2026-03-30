from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sheep_sim.agents import SheepAgent
from sheep_sim.behaviour import (
    BehaviourContext,
    desired_velocity,
    initialise_flock,
    update_agent_state,
)
from sheep_sim.config import SimulationConfig
from sheep_sim.environment import FieldEnvironment, build_environment
from sheep_sim.food import FoodLandscape, build_food_landscape
from sheep_sim.metrics import MetricsRecorder
from sheep_sim.rendering import LiveRenderer, save_final_frame
from sheep_sim.utils import circadian_factor, safe_unit


@dataclass(slots=True)
class SimulationResult:
    cfg: SimulationConfig
    flock: list[SheepAgent]
    environment: FieldEnvironment
    food: FoodLandscape
    metrics: MetricsRecorder


class SheepSimulation:
    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.environment = build_environment(self.rng, cfg.field)
        self.food = build_food_landscape(self.rng, cfg.field, cfg.food, cfg.scenario)
        self.flock = initialise_flock(cfg, self.rng)
        self.metrics = MetricsRecorder()

    def run(self, render: str = "none", output_dir: Path | None = None) -> SimulationResult:
        output_dir = output_dir or Path("outputs") / self.cfg.scenario
        output_dir.mkdir(parents=True, exist_ok=True)

        renderer = (
            LiveRenderer(
                self.environment,
                self.food,
                real_seconds_per_step=self.cfg.time.real_seconds_per_step,
                render_fps=self.cfg.time.render_fps,
                start_hour=self.cfg.time.start_hour,
                start_minute=self.cfg.time.start_minute,
            )
            if render == "live"
            else None
        )

        try:
            for step in range(self.cfg.steps):
                self.step(step)
                if step % self.cfg.output.record_every == 0:
                    self.metrics.record(
                        step=step,
                        flock=self.flock,
                        scenario=self.cfg.scenario,
                        food=self.food,
                    )
                if renderer is not None and step % self.cfg.output.render_every_n_steps == 0:
                    renderer.draw(self.flock, step, self.cfg.scenario)
        finally:
            if renderer is not None:
                renderer.close()

        self.metrics.group_dataframe().to_csv(output_dir / "metrics.csv", index=False)
        self.metrics.position_dataframe().to_csv(output_dir / "positions.csv", index=False)
        self.metrics.field_dataframe().to_csv(output_dir / "field_health.csv", index=False)

        if render in {"final", "live"}:
            save_final_frame(
                env=self.environment,
                food=self.food,
                flock=self.flock,
                out_path=output_dir / "final_frame.png",
                title=f"Sheep simulation final frame ({self.cfg.scenario})",
            )

        return SimulationResult(
            cfg=self.cfg,
            flock=self.flock,
            environment=self.environment,
            food=self.food,
            metrics=self.metrics,
        )

    def step(self, step: int) -> None:
        active_factor = circadian_factor(
            step=step,
            period=self.cfg.circadian.day_length_steps,
            phase_shift=self.cfg.circadian.active_peak_shift,
            amplitude=self.cfg.circadian.active_amplitude,
        )

        positions = np.array([s.position for s in self.flock])
        centroid = positions.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(positions - centroid, axis=1)))

        contexts: dict[int, BehaviourContext] = {}
        for sheep in self.flock:
            local_food = self.food.local_mean(
                sheep.position[0],
                sheep.position[1],
                self.cfg.food.sensory_radius,
                self.cfg.field,
            )
            local_health = self.food.local_health(
                sheep.position[0],
                sheep.position[1],
                self.cfg.food.sensory_radius,
                self.cfg.field,
            )
            sensory_grad = self.food.sensory_gradient(
                sheep.position[0],
                sheep.position[1],
                self.cfg.field,
                self.cfg.food.sensory_radius,
            )
            neighbour_count = self._count_neighbours(sheep)
            memory_target = self._memory_target(sheep)
            shade_val = self.environment.sample_shade(*sheep.position)
            terrain_val = self.environment.sample_terrain(*sheep.position)
            contexts[sheep.sheep_id] = BehaviourContext(
                local_food=local_food,
                local_health=local_health,
                sensory_gradient=safe_unit(sensory_grad),
                flock_centroid=centroid,
                flock_spread=spread,
                neighbour_count=neighbour_count,
                active_factor=active_factor,
                memory_target=memory_target,
                shade_value=shade_val,
                terrain_value=terrain_val,
            )

        for sheep in self.flock:
            update_agent_state(sheep, contexts[sheep.sheep_id], self.cfg, self.rng)

        new_velocities: dict[int, np.ndarray] = {}
        for sheep in self.flock:
            vel = desired_velocity(
                sheep=sheep,
                flock=self.flock,
                env=self.environment,
                food=self.food,
                ctx=contexts[sheep.sheep_id],
                cfg=self.cfg,
                rng=self.rng,
            )
            new_velocities[sheep.sheep_id] = vel

        for sheep in self.flock:
            sheep.velocity = new_velocities[sheep.sheep_id]
            sheep.heading = float(np.arctan2(sheep.velocity[1], sheep.velocity[0])) if sheep.speed() > 1e-9 else sheep.heading
            new_position = sheep.position + sheep.velocity * self.cfg.dt
            if not self.environment.in_bounds(new_position):
                new_position = self.environment.clamp_position(new_position)
                sheep.velocity = 0.65 * sheep.velocity * -1.0
            sheep.path_length += float(np.linalg.norm(new_position - sheep.position))
            sheep.position = new_position

            intake = 0.0
            if sheep.state.value == "grazing":
                intake = self.food.deplete(
                    sheep.position[0],
                    sheep.position[1],
                    self.cfg.field,
                    self.cfg.food.depletion_rate,
                    self.cfg.food,
                )
            sheep.last_food_intake = intake
            sheep.cumulative_food += intake
            sheep.energy = min(
                1.0,
                max(
                    0.0,
                    sheep.energy
                    - self.cfg.transitions.energy_loss_per_step * (0.3 + sheep.speed())
                    + self.cfg.transitions.food_gain_scale * intake
                    + self.cfg.transitions.fatigue_recovery_scale * (1.0 if sheep.state.value == "resting" else 0.0),
                ),
            )
            self._update_memory_map(sheep)

        dt_days = self.cfg.time.real_seconds_per_step / 86400.0
        self.food.regrow(dt_days, self.cfg.food)

    def _count_neighbours(self, focal: SheepAgent) -> int:
        count = 0
        for other in self.flock:
            if other.sheep_id == focal.sheep_id:
                continue
            dist = np.linalg.norm(other.position - focal.position)
            if dist <= self.cfg.flock.neighbour_radius:
                count += 1
        return count

    def _update_memory_map(self, sheep: SheepAgent) -> None:
        assert sheep.memory_map is not None
        sheep.memory_map *= (1.0 - self.cfg.food.memory_decay)
        row, col = self._grid_index(sheep.position)
        local_val = self.food.values[row, col] * self.food.health[row, col]
        sheep.memory_map[row, col] = max(sheep.memory_map[row, col], local_val)

    def _memory_target(self, sheep: SheepAgent) -> np.ndarray:
        assert sheep.memory_map is not None
        if np.max(sheep.memory_map) <= 1e-9:
            return sheep.position.copy()
        flat_idx = int(np.argmax(sheep.memory_map))
        row, col = np.unravel_index(flat_idx, sheep.memory_map.shape)
        x = (col + 0.5) / self.cfg.field.grid_cols * self.cfg.field.width
        y = (row + 0.5) / self.cfg.field.grid_rows * self.cfg.field.height
        return np.array([x, y], dtype=float)

    def _grid_index(self, pos: np.ndarray) -> tuple[int, int]:
        col = min(self.cfg.field.grid_cols - 1, max(0, int((pos[0] / self.cfg.field.width) * self.cfg.field.grid_cols)))
        row = min(self.cfg.field.grid_rows - 1, max(0, int((pos[1] / self.cfg.field.height) * self.cfg.field.grid_rows)))
        return row, col
