from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sheep_sim.core.agents import BehaviourState, SheepAgent
from sheep_sim.core.config import SimulationConfig
from sheep_sim.core.utils import circadian_factor, safe_unit
from sheep_sim.behaviour.states import BehaviourContext, initialise_flock, update_agent_state
from sheep_sim.behaviour.movement import desired_velocity
from sheep_sim.behaviour.stochastic import initialise_stochastic_traits, step_ou
from sheep_sim.environment.field import FieldEnvironment, build_environment
from sheep_sim.environment.food import FoodLandscape, build_food_landscape
from sheep_sim.metrics.recorder import MetricsRecorder
from sheep_sim.rendering.live import LiveRenderer
from sheep_sim.rendering.static import save_final_frame

@dataclass
class SimulationResult:
    cfg: SimulationConfig
    flock: list[SheepAgent]
    environment: FieldEnvironment
    food: FoodLandscape
    metrics: MetricsRecorder

class SheepSimulation:

    def __init__(self, cfg: SimulationConfig) -> None:
        self.cfg = cfg

        self.landscape_rng = np.random.default_rng(cfg.landscape_seed)
        self.behaviour_rng  = np.random.default_rng(cfg.seed)

        self.environment = build_environment(self.landscape_rng, cfg.field)
        self.food        = build_food_landscape(
            self.landscape_rng, cfg.field, cfg.food, cfg.scenario
        )

        self.flock = initialise_flock(cfg, self.behaviour_rng)
        for sheep in self.flock:
            initialise_stochastic_traits(
                sheep, self.behaviour_rng, personality=cfg.features.personality
            )

        self.metrics = MetricsRecorder()

    def run(
        self,
        render: str = "none",
        output_dir: Path | None = None,
    ) -> SimulationResult:
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
                if step % 2 == 0:
                    pct = step / max(self.cfg.steps - 1, 1) * 100
                    states = {s.state.value: 0 for s in self.flock}
                    for sh in self.flock:
                        states[sh.state.value] = states.get(sh.state.value, 0) + 1
                    state_str = " ".join(f"{k[0].upper()}:{v}" for k, v in sorted(states.items()))
                    print(f"\rStep {step:>5}/{self.cfg.steps}  ({pct:5.1f}%)  [{state_str}]", end="", flush=True)
                if renderer is not None and step % self.cfg.output.render_every_n_steps == 0:
                    renderer.draw(self.flock, step, self.cfg.scenario)
        finally:
            print()
            if renderer is not None:
                renderer.close()

        self.metrics.group_dataframe().to_csv(output_dir / "metrics.csv", index=False)
        self.metrics.position_dataframe().to_csv(output_dir / "positions.csv", index=False)
        self.metrics.field_dataframe().to_csv(output_dir / "field_health.csv", index=False)

        if render in {"final", "live"}:
            save_final_frame(
                self.environment,
                self.food,
                self.flock,
                output_dir / "final_frame.png",
                (
                    f"Sheep simulation — {self.cfg.scenario} "
                    f"(landscape={self.cfg.landscape_seed}, seed={self.cfg.seed})"
                ),
            )

        return SimulationResult(
            cfg=self.cfg,
            flock=self.flock,
            environment=self.environment,
            food=self.food,
            metrics=self.metrics,
        )

    def step(self, step: int) -> None:
        if self.cfg.features.circadian:
            active_factor = circadian_factor(
                step=step,
                period=self.cfg.circadian.day_length_steps,
                phase_shift=self.cfg.circadian.active_peak_shift,
                amplitude=self.cfg.circadian.active_amplitude,
            )
        else:
            active_factor = 1.0

        positions = np.array([s.position for s in self.flock], dtype=float)
        centroid  = positions.mean(axis=0)
        spread    = float(np.mean(np.linalg.norm(positions - centroid, axis=1)))

        if self.cfg.features.personality:
            for sheep in self.flock:
                step_ou(sheep, self.behaviour_rng, dt=self.cfg.dt)

        contexts: dict[int, BehaviourContext] = {}
        for sheep in self.flock:
            contexts[sheep.sheep_id] = self._build_context(
                sheep, centroid, spread, active_factor
            )


        prev_states = {sh.sheep_id: sh.state for sh in self.flock}

        for sheep in self.flock:
            update_agent_state(
                sheep, contexts[sheep.sheep_id], self.cfg, self.behaviour_rng
            )


        if self.cfg.features.social:
            from sheep_sim.core.agents import BehaviourState as _BS
            departures = [
                sh for sh in self.flock
                if prev_states[sh.sheep_id] == _BS.GRAZING
                and sh.state in (_BS.WALKING, _BS.TRAVELLING)
            ]
            if departures:
                positions_arr = np.array([sh.position for sh in self.flock])
                for dep in departures:
                    dists = np.linalg.norm(positions_arr - dep.position, axis=1)
                    for i, sh in enumerate(self.flock):
                        if sh.sheep_id != dep.sheep_id and dists[i] < self.cfg.flock.neighbour_radius * 0.8:
                            sh.contagion_steps = max(sh.contagion_steps, 6)


            for sh in self.flock:
                if sh.contagion_steps > 0:
                    sh.contagion_steps -= 1

        new_velocities = {
            sheep.sheep_id: desired_velocity(
                sheep=sheep,
                flock=self.flock,
                env=self.environment,
                food=self.food,
                ctx=contexts[sheep.sheep_id],
                cfg=self.cfg,
                rng=self.behaviour_rng,
            )
            for sheep in self.flock
        }

        for sheep in self.flock:
            sheep.velocity = new_velocities[sheep.sheep_id]

            if sheep.speed() > 1e-9:
                sheep.heading = float(
                    np.arctan2(sheep.velocity[1], sheep.velocity[0])
                )

            new_pos = sheep.position + sheep.velocity * self.cfg.dt
            if not self.environment.in_bounds(new_pos):
                new_pos = self.environment.clamp_position(new_pos)
                sheep.velocity = 0.65 * sheep.velocity * -1.0

            sheep.path_length += float(np.linalg.norm(new_pos - sheep.position))
            sheep.position = new_pos

            intake = 0.0
            if sheep.state == BehaviourState.GRAZING:
                intake = self.food.deplete(
                    *sheep.position,
                    self.cfg.field,
                    self.cfg.food.depletion_rate,
                    self.cfg.food,
                )
            sheep.last_food_intake = intake
            sheep.cumulative_food += intake

            if self.cfg.features.memory:
                self._update_memory(sheep)

    def _build_context(
        self,
        sheep: SheepAgent,
        centroid: np.ndarray,
        spread: float,
        active_factor: float,
    ) -> BehaviourContext:
        x, y = sheep.position
        return BehaviourContext(
            local_food    = self.food.local_mean    (x, y, self.cfg.food.sensory_radius, self.cfg.field),
            local_health  = self.food.local_health  (x, y, self.cfg.food.sensory_radius, self.cfg.field),
            local_quality = self.food.local_quality (x, y, self.cfg.food.sensory_radius, self.cfg.field),
            cell_food     = self.food.sample        (x, y, self.cfg.field),
            cell_health   = self.food.sample_health (x, y, self.cfg.field),
            cell_quality  = self.food.sample_quality(x, y, self.cfg.field),
            sensory_gradient = safe_unit(
                self.food.sensory_gradient(x, y, self.cfg.field, self.cfg.food.sensory_radius)
            ),
            flock_centroid     = centroid,
            flock_spread       = spread,
            neighbour_count    = self._count_neighbours(sheep),
            active_factor      = active_factor,
            memory_target      = self._memory_target(sheep) if self.cfg.features.memory else sheep.position.copy(),
            terrain_value      = self.environment.sample_terrain(x, y),
            recent_intake_rate = sheep.last_food_intake,
        )

    def _count_neighbours(self, focal: SheepAgent) -> int:
        return sum(
            1 for other in self.flock
            if other.sheep_id != focal.sheep_id
            and np.linalg.norm(other.position - focal.position)
            <= self.cfg.flock.neighbour_radius
        )

    def _update_memory(self, sheep: SheepAgent) -> None:
        assert sheep.memory_map is not None
        sheep.memory_map *= (1.0 - self.cfg.food.memory_decay)
        r, c = self._grid_idx(sheep.position)
        local_val = self.food.values[r, c] * self.food.health[r, c]
        sheep.memory_map[r, c] = max(sheep.memory_map[r, c], local_val)

    def _memory_target(self, sheep: SheepAgent) -> np.ndarray:
        assert sheep.memory_map is not None
        if np.max(sheep.memory_map) <= 1e-9:
            return sheep.position.copy()

        values = sheep.memory_map.copy()
        rows, cols = values.shape
        pr, pc = self._grid_idx(sheep.position)
        yy, xx = np.meshgrid(np.arange(rows), np.arange(cols), indexing="ij")
        dist_cells = np.sqrt((yy - pr) ** 2 + (xx - pc) ** 2)

        if self.cfg.scenario == "abundant":
            distance_bonus = np.clip(dist_cells / 12.0, 0.0, 1.0)
            weighted = np.where(
                dist_cells >= 4,
                values * (0.88 + 0.12 * distance_bonus),
                -np.inf,
            )
        else:
            weighted = np.where(dist_cells >= 5, values, -np.inf)

        use      = values if np.all(~np.isfinite(weighted)) else weighted
        flat_idx = int(np.argmax(use))
        r, c     = np.unravel_index(flat_idx, values.shape)
        return np.array([
            (c + 0.5) / self.cfg.field.grid_cols * self.cfg.field.width,
            (r + 0.5) / self.cfg.field.grid_rows * self.cfg.field.height,
        ], dtype=float)

    def _grid_idx(self, pos: np.ndarray) -> tuple[int, int]:
        c = min(
            self.cfg.field.grid_cols - 1,
            max(0, int((pos[0] / self.cfg.field.width) * self.cfg.field.grid_cols)),
        )
        r = min(
            self.cfg.field.grid_rows - 1,
            max(0, int((pos[1] / self.cfg.field.height) * self.cfg.field.grid_rows)),
        )
        return r, c