from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sheep_sim.core.agents import BehaviourState, SheepAgent
from sheep_sim.core.config import FlockConfig, SimulationConfig, StateTransitionConfig
from sheep_sim.core.utils import clamp, rotate, safe_unit
from sheep_sim.environment.field import FieldEnvironment
from sheep_sim.environment.food import FoodLandscape

@dataclass(slots=True)
class BehaviourContext:
    local_food: float
    local_health: float
    local_quality: float
    cell_food: float
    cell_health: float
    cell_quality: float
    sensory_gradient: np.ndarray

    flock_centroid: np.ndarray
    flock_spread: float
    neighbour_count: int

    active_factor: float

    memory_target: np.ndarray
    home_target: np.ndarray

    shade_value: float
    terrain_value: float
    recent_intake_rate: float

def initialise_flock(cfg: SimulationConfig, rng: np.random.Generator) -> list[SheepAgent]:
    from sheep_sim.behaviour.movement import heading_to_velocity

    flock: list[SheepAgent] = []
    centre = np.array([cfg.field.width * 0.5, cfg.field.height * 0.5], dtype=float)
    spread = 10.0 if cfg.scenario == "abundant" else 14.0

    for i in range(cfg.flock.n_sheep):
        pos = centre + rng.normal(0.0, spread, size=2)
        pos[0] = clamp(float(pos[0]), 0.0, cfg.field.width)
        pos[1] = clamp(float(pos[1]), 0.0, cfg.field.height)
        heading = float(rng.uniform(-np.pi, np.pi))

        if cfg.scenario == "abundant":
            speed          = rng.uniform(0.01, 0.08)
            state          = BehaviourState.GRAZING
            sociability    = float(rng.uniform(0.90, 1.18))
            boldness       = float(rng.uniform(0.86, 1.12))
            movement_vigor = float(rng.uniform(0.90, 1.10))
            turning_bias   = float(rng.uniform(1.00, 1.35))
            patch_leave_bias   = float(rng.uniform(0.88, 1.08))
            preferred_spacing  = float(rng.uniform(0.95, 1.18))
        else:
            speed          = rng.uniform(0.08, 0.24)
            state          = BehaviourState.TRAVELLING
            sociability    = float(rng.uniform(0.75, 1.15))
            boldness       = float(rng.uniform(0.80, 1.20))
            movement_vigor = float(rng.uniform(0.85, 1.15))
            turning_bias   = float(rng.uniform(0.85, 1.20))
            patch_leave_bias   = float(rng.uniform(0.85, 1.15))
            preferred_spacing  = float(rng.uniform(0.85, 1.20))

        flock.append(SheepAgent(
            sheep_id=i,
            position=pos.astype(float),
            velocity=heading_to_velocity(heading, speed),
            heading=heading,
            state=state,
            memory_map=np.zeros((cfg.field.grid_rows, cfg.field.grid_cols), dtype=float),
            site_fidelity=float(rng.uniform(0.90, 1.15)),
            sociability=sociability,
            boldness=boldness,
            movement_vigor=movement_vigor,
            turning_bias=turning_bias,
            patch_leave_bias=patch_leave_bias,
            preferred_spacing=preferred_spacing,
            home_target=pos.astype(float).copy(),
        ))
    return flock

def update_agent_state(
    sheep: SheepAgent,
    ctx: BehaviourContext,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> None:
    tcfg: StateTransitionConfig = cfg.transitions
    sheep.state_age += 1
    if sheep.regroup_cooldown > 0:
        sheep.regroup_cooldown -= 1

    field_scale     = max(cfg.field.width, cfg.field.height)
    dist_centroid   = float(np.linalg.norm(ctx.flock_centroid - sheep.position))
    close_enough    = dist_centroid < cfg.flock.neighbour_radius * 1.05
    far_from_group  = dist_centroid > cfg.flock.neighbour_radius * 2.0
    isolated        = ctx.neighbour_count <= 0
    socially_connected = ctx.neighbour_count >= 2
    flock_compact   = ctx.flock_spread < 0.14 * field_scale

    local_food_enter = tcfg.local_food_enter_graze * sheep.site_fidelity
    local_food_exit  = tcfg.local_food_exit_graze  * sheep.site_fidelity

    if isolated or far_from_group:
        sheep.separation_steps += 1
    else:
        sheep.separation_steps = max(0, sheep.separation_steps - 1)

    if cfg.scenario == "abundant":
        if ctx.active_factor < cfg.circadian.resting_threshold:
            if sheep.state != BehaviourState.REGROUPING:
                rest_p = 0.18 + 0.30 * (
                    (cfg.circadian.resting_threshold - ctx.active_factor)
                    / max(cfg.circadian.resting_threshold, 1e-9)
                )
                if sheep.state == BehaviourState.GRAZING:
                    rest_p += 0.12
                if rng.random() < clamp(rest_p, 0.0, 0.72):
                    _transition(sheep, BehaviourState.RESTING)
                    return
        elif sheep.state == BehaviourState.RESTING:
            wake_p = clamp(
                0.22 + 0.90 * (ctx.active_factor - cfg.circadian.resting_threshold),
                0.10, 0.95,
            )
            if rng.random() < wake_p:
                next_state = (
                    BehaviourState.GRAZING
                    if ctx.local_quality >= local_food_enter * 0.82 and ctx.cell_quality > 0.07
                    else BehaviourState.WALKING
                )
                _transition(sheep, next_state)
            return
    else:
        if ctx.active_factor < cfg.circadian.resting_threshold:
            if sheep.state != BehaviourState.REGROUPING:
                _transition(sheep, BehaviourState.RESTING)
                return
        if sheep.state == BehaviourState.RESTING:
            if ctx.active_factor >= cfg.circadian.resting_threshold:
                next_state = (
                    BehaviourState.GRAZING
                    if ctx.local_quality >= local_food_enter and ctx.cell_quality > 0.08
                    else BehaviourState.TRAVELLING
                )
                _transition(sheep, next_state)
            return

    if sheep.state == BehaviourState.REGROUPING:
        min_steps = 5  if cfg.scenario == "abundant" else 8
        max_steps = 12 if cfg.scenario == "abundant" else 16
        if (
            (sheep.state_age >= min_steps and close_enough and socially_connected and flock_compact)
            or sheep.state_age >= max_steps
        ):
            sheep.separation_steps = 0
            sheep.regroup_cooldown = 16 if cfg.scenario == "abundant" else 12
            next_state = (
                BehaviourState.GRAZING
                if cfg.scenario == "abundant"
                   and ctx.local_quality >= local_food_exit * 0.72
                   and ctx.cell_quality > 0.07
                else BehaviourState.WALKING if cfg.scenario == "abundant"
                else BehaviourState.TRAVELLING
            )
            _transition(sheep, next_state)
        return

    if sheep.regroup_cooldown == 0:
        if cfg.scenario == "abundant":
            if (
                sheep.separation_steps >= int(12 * sheep.boldness)
                and (isolated or far_from_group)
                and ctx.local_quality < local_food_exit * 0.88
            ):
                _transition(sheep, BehaviourState.REGROUPING)
                return
        else:
            if sheep.separation_steps >= 8 and (isolated or far_from_group):
                _transition(sheep, BehaviourState.REGROUPING)
                return

    if cfg.scenario == "abundant":
        stale_steps = int(max(18, tcfg.stale_patch_steps_abundant * sheep.patch_leave_bias * 1.25))
        stale_patch = (
            ctx.cell_quality < tcfg.stale_patch_quality_threshold
            and sheep.state_age >= stale_steps
        )
        low_gain = ctx.recent_intake_rate < 0.0020 and sheep.state_age >= stale_steps

        if sheep.state == BehaviourState.GRAZING:
            keep = (
                ctx.local_quality >= local_food_exit * 0.72
                and ctx.local_health > 0.14
                and ctx.cell_quality > 0.07
                and sheep.energy > 0.12
            )
            should_walk = (
                stale_patch or low_gain
                or ctx.cell_quality < 0.05
                or (ctx.local_quality < local_food_exit * 0.62 and sheep.state_age > 34)
            )
            should_travel = (
                ctx.local_quality < local_food_exit * 0.42
                or ctx.local_health < 0.07
                or sheep.state_age > 420
            )
            if should_travel:
                _transition(sheep, BehaviourState.TRAVELLING)
                return
            if (not keep and should_walk) or stale_patch:
                _transition(sheep, BehaviourState.WALKING)
            return

        if sheep.state == BehaviourState.WALKING:
            if sheep.state_age < 8:
                return
            if ctx.local_quality >= local_food_exit * 0.68 and ctx.cell_quality > 0.07 and ctx.local_health > 0.14:
                _transition(sheep, BehaviourState.GRAZING)
                return
            if ctx.local_quality < local_food_exit * 0.46 and sheep.state_age >= 24:
                _transition(sheep, BehaviourState.TRAVELLING)
            return

        if sheep.state == BehaviourState.TRAVELLING:
            if sheep.state_age < 14:
                return
            if ctx.local_quality >= local_food_exit * 0.58:
                _transition(sheep, BehaviourState.WALKING)
            return

        if sheep.state == BehaviourState.RESTING:
            return

        _transition(sheep, BehaviourState.GRAZING)
        return

    if sheep.state == BehaviourState.GRAZING:
        if (
            ctx.local_quality < local_food_enter * 0.85
            or ctx.cell_quality < 0.08
            or sheep.state_age > 24
        ):
            _transition(sheep, BehaviourState.WALKING)
        return

    if sheep.state == BehaviourState.WALKING:
        if sheep.state_age < 8:
            return
        if ctx.local_quality >= local_food_enter and ctx.cell_quality > 0.09:
            _transition(sheep, BehaviourState.GRAZING)
            return
        if ctx.local_quality < local_food_enter * 0.75:
            _transition(sheep, BehaviourState.TRAVELLING)
        return

    if sheep.state == BehaviourState.TRAVELLING:
        if sheep.state_age < 24:
            return
        if ctx.local_quality >= local_food_enter * 0.95:
            _transition(sheep, BehaviourState.WALKING)
        return

    if sheep.state == BehaviourState.RESTING:
        return

    _transition(
        sheep,
        BehaviourState.WALKING if ctx.local_quality < local_food_enter else BehaviourState.GRAZING,
    )

def _transition(sheep: SheepAgent, new_state: BehaviourState) -> None:
    sheep.state = new_state
    sheep.state_age = 0
