from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sheep_sim.agents import BehaviourState, SheepAgent
from sheep_sim.behaviour import BehaviourContext, desired_velocity, initialise_flock, update_agent_state
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
                    self.metrics.record(step=step, flock=self.flock, scenario=self.cfg.scenario, food=self.food)
                if renderer is not None and step % self.cfg.output.render_every_n_steps == 0:
                    renderer.draw(self.flock, step, self.cfg.scenario)
        finally:
            if renderer is not None:
                renderer.close()

        self.metrics.group_dataframe().to_csv(output_dir / "metrics.csv", index=False)
        self.metrics.position_dataframe().to_csv(output_dir / "positions.csv", index=False)
        self.metrics.field_dataframe().to_csv(output_dir / "field_health.csv", index=False)
        if render in {"final", "live"}:
            save_final_frame(self.environment, self.food, self.flock, output_dir / "final_frame.png", f"Sheep simulation final frame ({self.cfg.scenario})")

        return SimulationResult(self.cfg, self.flock, self.environment, self.food, self.metrics)

    def step(self, step: int) -> None:
        active_factor = circadian_factor(
            step=step,
            period=self.cfg.circadian.day_length_steps,
            phase_shift=self.cfg.circadian.active_peak_shift,
            amplitude=self.cfg.circadian.active_amplitude,
        )

        positions = np.array([s.position for s in self.flock], dtype=float)
        centroid = positions.mean(axis=0)
        spread = float(np.mean(np.linalg.norm(positions - centroid, axis=1)))

        contexts: dict[int, BehaviourContext] = {}
        for sheep in self.flock:
            local_food = self.food.local_mean(*sheep.position, self.cfg.food.sensory_radius, self.cfg.field)
            local_health = self.food.local_health(*sheep.position, self.cfg.food.sensory_radius, self.cfg.field)
            local_quality = self.food.local_quality(*sheep.position, self.cfg.food.sensory_radius, self.cfg.field)
            cell_food = self.food.sample(*sheep.position, self.cfg.field)
            cell_health = self.food.sample_health(*sheep.position, self.cfg.field)
            cell_quality = self.food.sample_quality(*sheep.position, self.cfg.field)
            sensory_grad = self.food.sensory_gradient(*sheep.position, self.cfg.field, self.cfg.food.sensory_radius)
            contexts[sheep.sheep_id] = BehaviourContext(
                local_food=local_food,
                local_health=local_health,
                local_quality=local_quality,
                cell_food=cell_food,
                cell_health=cell_health,
                cell_quality=cell_quality,
                sensory_gradient=safe_unit(sensory_grad),
                flock_centroid=centroid,
                flock_spread=spread,
                neighbour_count=self._count_neighbours(sheep),
                active_factor=active_factor,
                memory_target=self._memory_target(sheep),
                home_target=self._home_target(sheep),
                shade_value=self.environment.sample_shade(*sheep.position),
                terrain_value=self.environment.sample_terrain(*sheep.position),
                recent_intake_rate=sheep.last_food_intake,
            )

        for sheep in self.flock:
            update_agent_state(sheep, contexts[sheep.sheep_id], self.cfg, self.rng)

        new_velocities = {
            sheep.sheep_id: desired_velocity(
                sheep=sheep,
                flock=self.flock,
                env=self.environment,
                food=self.food,
                ctx=contexts[sheep.sheep_id],
                cfg=self.cfg,
                rng=self.rng,
            )
            for sheep in self.flock
        }

        for sheep in self.flock:
            sheep.velocity = new_velocities[sheep.sheep_id]
            if sheep.speed() > 1e-9:
                sheep.heading = float(np.arctan2(sheep.velocity[1], sheep.velocity[0]))
            new_position = sheep.position + sheep.velocity * self.cfg.dt
            if not self.environment.in_bounds(new_position):
                new_position = self.environment.clamp_position(new_position)
                sheep.velocity = 0.65 * sheep.velocity * -1.0
            sheep.path_length += float(np.linalg.norm(new_position - sheep.position))
            sheep.position = new_position

            intake = 0.0
            if sheep.state == BehaviourState.GRAZING:
                intake = self.food.deplete(*sheep.position, self.cfg.field, self.cfg.food.depletion_rate, self.cfg.food)
            sheep.last_food_intake = intake
            sheep.cumulative_food += intake
            sheep.energy = min(
                1.0,
                max(
                    0.0,
                    sheep.energy
                    - self.cfg.transitions.energy_loss_per_step * (0.3 + sheep.speed())
                    + self.cfg.transitions.food_gain_scale * intake
                    + self.cfg.transitions.fatigue_recovery_scale * (1.0 if sheep.state == BehaviourState.RESTING else 0.0),
                ),
            )
            self._update_memory_map(sheep)

        self.food.regrow(self.cfg.time.real_seconds_per_step / 86400.0, self.cfg.food)

    def _count_neighbours(self, focal: SheepAgent) -> int:
        return sum(
            1 for other in self.flock
            if other.sheep_id != focal.sheep_id and np.linalg.norm(other.position - focal.position) <= self.cfg.flock.neighbour_radius
        )

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

        values = sheep.memory_map.copy()
        rows, cols = values.shape
        pos_row, pos_col = self._grid_index(sheep.position)
        yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        dist_cells = np.sqrt((yy - pos_row) ** 2 + (xx - pos_col) ** 2)

        if self.cfg.scenario == "abundant":
            min_dist = 4
            distance_bonus = np.clip(dist_cells / 12.0, 0.0, 1.0)
            weighted = np.where(dist_cells >= min_dist, values * (0.88 + 0.12 * distance_bonus), -np.inf)
        else:
            min_dist = 5
            weighted = np.where(dist_cells >= min_dist, values, -np.inf)

        flat_idx = int(np.argmax(values if np.all(~np.isfinite(weighted)) else weighted))
        row, col = np.unravel_index(flat_idx, values.shape)
        return np.array([
            (col + 0.5) / self.cfg.field.grid_cols * self.cfg.field.width,
            (row + 0.5) / self.cfg.field.grid_rows * self.cfg.field.height,
        ], dtype=float)

    def _home_target(self, sheep: SheepAgent) -> np.ndarray:
        assert sheep.home_target is not None
        if sheep.last_food_intake > 0.004:
            blend = 0.18 if self.cfg.scenario == "abundant" else 0.08
            sheep.home_target = (1.0 - blend) * sheep.home_target + blend * sheep.position
        elif self.cfg.scenario == "abundant":
            sheep.home_target = 0.995 * sheep.home_target + 0.005 * sheep.position
        return sheep.home_target.copy()

    def _grid_index(self, pos: np.ndarray) -> tuple[int, int]:
        col = min(self.cfg.field.grid_cols - 1, max(0, int((pos[0] / self.cfg.field.width) * self.cfg.field.grid_cols)))
        row = min(self.cfg.field.grid_rows - 1, max(0, int((pos[1] / self.cfg.field.height) * self.cfg.field.grid_rows)))
        return row, col
