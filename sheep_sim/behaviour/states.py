from __future__ import annotations
 
from dataclasses import dataclass
 
import numpy as np
 
from sheep_sim.core.agents import BehaviourState, SheepAgent
from sheep_sim.core.config import FlockConfig, SimulationConfig, StateTransitionConfig
from sheep_sim.core.utils import clamp
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
    terrain_value: float
    recent_intake_rate: float
 
 
def initialise_flock(cfg: SimulationConfig, rng: np.random.Generator) -> list[SheepAgent]:
    from sheep_sim.behaviour.movement import heading_to_velocity
 
    flock: list[SheepAgent] = []
 







    margin = 8.0
    centre_spawn = cfg.scenario in {"uniform_low", "uniform_high", "radial_increase", "ring", "corridors"}
    cx = cfg.field.width  / 2.0
    cy = cfg.field.height / 2.0


    spawn_sigma = 5.0 if cfg.scenario == "corridors" else 10.0

    for i in range(cfg.flock.n_sheep):
        if centre_spawn:
            x = float(rng.normal(cx, spawn_sigma))
            y = float(rng.normal(cy, spawn_sigma))
            x = min(cfg.field.width  - margin, max(margin, x))
            y = min(cfg.field.height - margin, max(margin, y))
            pos = np.array([x, y], dtype=float)
        else:
            pos = np.array([
                float(rng.uniform(margin, cfg.field.width  - margin)),
                float(rng.uniform(margin, cfg.field.height - margin)),
            ], dtype=float)
        heading = float(rng.uniform(-np.pi, np.pi))
 
        if cfg.scenario in {"abundant", "uniform_high", "radial_increase", "ring", "corridors"}:
            speed          = rng.uniform(0.5, 2.0)
            state          = BehaviourState.GRAZING
            sociability    = float(rng.uniform(0.90, 1.18))
            boldness       = float(rng.uniform(0.86, 1.12))
            movement_vigor = float(rng.uniform(0.90, 1.10))
            turning_bias   = float(rng.uniform(1.00, 1.35))
            patch_leave_bias   = float(rng.uniform(0.88, 1.08))
            preferred_spacing  = float(rng.uniform(0.95, 1.18))
            site_fidelity      = float(rng.uniform(0.90, 1.15))
        else:
            speed          = rng.uniform(5.0, 15.0)
            state          = BehaviourState.TRAVELLING
            sociability    = float(rng.uniform(0.75, 1.15))
            boldness       = float(rng.uniform(0.80, 1.20))
            movement_vigor = float(rng.uniform(0.85, 1.15))
            turning_bias   = float(rng.uniform(0.85, 1.20))
            patch_leave_bias   = float(rng.uniform(0.85, 1.15))
            preferred_spacing  = float(rng.uniform(0.85, 1.20))
            site_fidelity      = float(rng.uniform(0.90, 1.15))
 
        if not cfg.features.personality:
            sociability = boldness = movement_vigor = 1.0
            turning_bias = patch_leave_bias = preferred_spacing = 1.0
            site_fidelity = 1.0
 
        flock.append(SheepAgent(
            sheep_id=i,
            position=pos.astype(float),
            velocity=heading_to_velocity(heading, speed),
            heading=heading,
            state=state,
            memory_map=np.zeros((cfg.field.grid_rows, cfg.field.grid_cols), dtype=float),
            site_fidelity=site_fidelity,
            sociability=sociability,
            boldness=boldness,
            movement_vigor=movement_vigor,
            turning_bias=turning_bias,
            patch_leave_bias=patch_leave_bias,
            preferred_spacing=preferred_spacing,
        ))
    return flock
 
 
def update_agent_state(
    sheep: SheepAgent,
    ctx: BehaviourContext,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> None:
    from sheep_sim.behaviour.stochastic import markov_bias_for
 
    tcfg: StateTransitionConfig = cfg.transitions
    sheep.state_age += 1
    if sheep.regroup_cooldown > 0:
        sheep.regroup_cooldown -= 1
 
    field_scale     = max(cfg.field.width, cfg.field.height)
    dist_centroid   = float(np.linalg.norm(ctx.flock_centroid - sheep.position))
    close_enough    = dist_centroid < cfg.flock.neighbour_radius * 1.05
    far_from_group  = dist_centroid > cfg.flock.neighbour_radius * 2.0

    shy_offset  = max(0, round(2.0 * (1.0 - sheep.boldness)))
    isolated    = ctx.neighbour_count <= shy_offset
    socially_connected = ctx.neighbour_count >= 2
    flock_compact   = ctx.flock_spread < 0.14 * field_scale
 
    local_food_enter = tcfg.local_food_enter_graze
    local_food_exit  = tcfg.local_food_exit_graze
 


    cell_graze_guard   = local_food_enter * 0.30
    cell_health_guard  = max(0.05, tcfg.local_food_exit_graze * 0.4)
 
    if isolated or far_from_group:
        sheep.separation_steps += 1
    else:
        sheep.separation_steps = max(0, sheep.separation_steps - 1)
 
    is_rich = cfg.scenario in {"abundant", "uniform_high", "radial_increase", "ring", "corridors"}
 

    if not cfg.features.circadian:
        if sheep.state == BehaviourState.RESTING:
            _transition(sheep, BehaviourState.GRAZING if is_rich else BehaviourState.WALKING)
    else:


        wake_fallback = BehaviourState.WALKING if is_rich else BehaviourState.TRAVELLING
        if ctx.active_factor < cfg.circadian.resting_threshold:
            if sheep.state != BehaviourState.REGROUPING:
                rest_p = 0.18 + 0.30 * (
                    (cfg.circadian.resting_threshold - ctx.active_factor)
                    / max(cfg.circadian.resting_threshold, 1e-9)
                )
                if sheep.state == BehaviourState.GRAZING:
                    rest_p += 0.12
                bias = markov_bias_for(sheep, BehaviourState.RESTING)
                if rng.random() < clamp(rest_p * bias, 0.0, 0.72):
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
                    if ctx.local_quality >= local_food_enter * 0.82
                    and ctx.cell_quality >= cell_graze_guard
                    else wake_fallback
                )
                _transition(sheep, next_state)
            return
 

    if not cfg.features.social:
        if sheep.state == BehaviourState.REGROUPING:
            _transition(sheep, BehaviourState.GRAZING if is_rich else BehaviourState.WALKING)
    else:
        if sheep.state == BehaviourState.REGROUPING:
            min_steps = 5  if is_rich else 8
            max_steps = 12 if is_rich else 16
            if (
                (sheep.state_age >= min_steps and close_enough and socially_connected and flock_compact)
                or sheep.state_age >= max_steps
            ):
                sheep.separation_steps = 0
                sheep.regroup_cooldown = 16 if is_rich else 12
                next_state = (
                    BehaviourState.GRAZING
                    if is_rich
                    and ctx.local_quality >= local_food_exit * 0.72
                    and ctx.cell_quality >= cell_graze_guard
                    else BehaviourState.WALKING if is_rich
                    else BehaviourState.TRAVELLING
                )
                _transition(sheep, next_state)
            return
 
        if sheep.regroup_cooldown == 0:
            if is_rich:
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
 


    if not cfg.features.foraging:
        if sheep.state not in {BehaviourState.RESTING, BehaviourState.REGROUPING}:
            if sheep.state != BehaviourState.GRAZING:
                _transition(sheep, BehaviourState.GRAZING)
        return
 
    if is_rich:
        stale_steps = int(max(18, tcfg.stale_patch_steps_abundant * sheep.patch_leave_bias * 1.25))
        stale_patch = (
            ctx.cell_quality < tcfg.stale_patch_quality_threshold
            and sheep.state_age >= stale_steps
        )
        low_gain = ctx.recent_intake_rate < 0.0020 and sheep.state_age >= stale_steps
 
        if sheep.state == BehaviourState.GRAZING:

            contagion_factor = 0.85 if sheep.contagion_steps > 0 else 1.0
            keep = (
                ctx.local_quality >= local_food_exit * 0.55 * contagion_factor
                and ctx.local_health > cell_health_guard
                and ctx.cell_quality >= cell_graze_guard
            )
            should_walk = (
                stale_patch or low_gain
                or ctx.cell_quality < cell_graze_guard * 0.5
                or (ctx.local_quality < local_food_exit * 0.45 and sheep.state_age > 60)
            )
            should_travel = (
                ctx.local_quality < local_food_exit * 0.42
                or ctx.local_health < cell_health_guard * 0.5
                or sheep.state_age > 420
            )
            if should_travel:
                _transition(sheep, BehaviourState.TRAVELLING)
                return
            if (not keep and should_walk) or stale_patch:
                _transition(sheep, BehaviourState.WALKING)
            return
 
        if sheep.state == BehaviourState.WALKING:
            if sheep.state_age < 6:
                return
            if (ctx.local_quality >= local_food_exit * 0.50
                    and ctx.cell_quality >= cell_graze_guard
                    and ctx.local_health > cell_health_guard):
                _transition(sheep, BehaviourState.GRAZING)
                return
            if ctx.local_quality < local_food_exit * 0.36 and sheep.state_age >= 30:
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
            ctx.local_quality < local_food_enter * 0.55
            or ctx.cell_quality < cell_graze_guard
            or sheep.state_age > 90
        ):
            _transition(sheep, BehaviourState.WALKING)
        return
 
    if sheep.state == BehaviourState.WALKING:
        if sheep.state_age < 6:
            return
        if ctx.local_quality >= local_food_enter * 0.75 and ctx.cell_quality >= cell_graze_guard:
            _transition(sheep, BehaviourState.GRAZING)
            return
        if ctx.local_quality < local_food_enter * 0.50 and sheep.state_age >= 20:
            _transition(sheep, BehaviourState.TRAVELLING)
        return
 
    if sheep.state == BehaviourState.TRAVELLING:
        if sheep.state_age < 18:
            return
        if ctx.local_quality >= local_food_enter * 0.80:
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