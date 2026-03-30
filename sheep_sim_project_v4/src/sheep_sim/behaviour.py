from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sheep_sim.agents import BehaviourState, SheepAgent
from sheep_sim.config import FlockConfig, SimulationConfig, StateTransitionConfig
from sheep_sim.environment import FieldEnvironment
from sheep_sim.food import FoodLandscape
from sheep_sim.utils import clamp, rotate, safe_unit


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
    flock: list[SheepAgent] = []
    centre = np.array([cfg.field.width * 0.5, cfg.field.height * 0.5], dtype=float)
    spread = 10.0 if cfg.scenario == "abundant" else 14.0

    for i in range(cfg.flock.n_sheep):
        pos = centre + rng.normal(0.0, spread, size=2)
        pos[0] = clamp(float(pos[0]), 0.0, cfg.field.width)
        pos[1] = clamp(float(pos[1]), 0.0, cfg.field.height)
        heading = float(rng.uniform(-np.pi, np.pi))
        if cfg.scenario == "abundant":
            speed = rng.uniform(0.02, 0.12)
            state = BehaviourState.GRAZING
        else:
            speed = rng.uniform(0.08, 0.24)
            state = BehaviourState.TRAVELLING

        flock.append(
            SheepAgent(
                sheep_id=i,
                position=pos.astype(float),
                velocity=heading_to_velocity(heading, speed),
                heading=heading,
                state=state,
                memory_map=np.zeros((cfg.field.grid_rows, cfg.field.grid_cols), dtype=float),
                site_fidelity=float(rng.uniform(0.85, 1.20)),
                sociability=float(rng.uniform(0.75, 1.20)),
                boldness=float(rng.uniform(0.80, 1.20)),
                movement_vigor=float(rng.uniform(0.85, 1.15)),
                turning_bias=float(rng.uniform(0.85, 1.20)),
                patch_leave_bias=float(rng.uniform(0.85, 1.15)),
                preferred_spacing=float(rng.uniform(0.85, 1.20)),
                home_target=pos.astype(float).copy(),
            )
        )
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

    field_scale = max(cfg.field.width, cfg.field.height)
    dist_to_centroid = float(np.linalg.norm(ctx.flock_centroid - sheep.position))
    close_enough = dist_to_centroid < cfg.flock.neighbour_radius * 0.95
    far_from_group = dist_to_centroid > cfg.flock.neighbour_radius * 2.3
    isolated = ctx.neighbour_count <= 0
    socially_connected = ctx.neighbour_count >= 2
    flock_compact = ctx.flock_spread < 0.16 * field_scale

    local_food_enter = tcfg.local_food_enter_graze * sheep.site_fidelity
    local_food_exit = tcfg.local_food_exit_graze * sheep.site_fidelity

    if isolated or far_from_group:
        sheep.separation_steps += 1
    else:
        sheep.separation_steps = max(0, sheep.separation_steps - 1)

    if ctx.active_factor < cfg.circadian.resting_threshold:
        if sheep.state != BehaviourState.REGROUPING:
            sheep.state = BehaviourState.RESTING
            sheep.state_age = 0
            return

    if sheep.state == BehaviourState.REGROUPING:
        min_regroup_steps = 4 if cfg.scenario == "abundant" else 8
        max_regroup_steps = 10 if cfg.scenario == "abundant" else 16
        if (
            (sheep.state_age >= min_regroup_steps and close_enough and socially_connected and flock_compact)
            or sheep.state_age >= max_regroup_steps
        ):
            sheep.separation_steps = 0
            sheep.regroup_cooldown = 24 if cfg.scenario == "abundant" else 12
            sheep.state = (
                BehaviourState.GRAZING
                if cfg.scenario == "abundant" and ctx.local_quality >= local_food_exit * 0.80 and ctx.cell_quality > 0.08
                else BehaviourState.WALKING if cfg.scenario == "abundant" else BehaviourState.TRAVELLING
            )
            sheep.state_age = 0
        return

    if sheep.state == BehaviourState.RESTING:
        if ctx.active_factor >= cfg.circadian.resting_threshold:
            sheep.state = (
                BehaviourState.GRAZING
                if ctx.local_quality >= local_food_enter and ctx.cell_quality > 0.08
                else BehaviourState.WALKING if cfg.scenario == "abundant" else BehaviourState.TRAVELLING
            )
            sheep.state_age = 0
        return

    if sheep.regroup_cooldown == 0:
        if cfg.scenario == "abundant":
            if (
                sheep.separation_steps >= int(22 * sheep.boldness)
                and isolated
                and dist_to_centroid > cfg.flock.neighbour_radius * 2.3
                and ctx.local_quality < local_food_exit * 0.70
            ):
                sheep.state = BehaviourState.REGROUPING
                sheep.state_age = 0
                return
        else:
            if sheep.separation_steps >= 8 and (isolated or far_from_group):
                sheep.state = BehaviourState.REGROUPING
                sheep.state_age = 0
                return

    if cfg.scenario == "abundant":
        stale_steps = int(tcfg.stale_patch_steps_abundant * sheep.patch_leave_bias)
        stale_patch = ctx.cell_quality < tcfg.stale_patch_quality_threshold and sheep.state_age >= stale_steps
        low_gain = ctx.recent_intake_rate < 0.0025 and sheep.state_age >= stale_steps

        if sheep.state == BehaviourState.GRAZING:
            keep = (
                ctx.local_quality >= local_food_exit * 0.90
                and ctx.local_health > 0.18
                and ctx.cell_quality > 0.10
                and sheep.energy > 0.16
            )
            should_walk = stale_patch or low_gain or ctx.cell_quality < 0.08 or (ctx.local_quality < local_food_exit * 0.72 and sheep.state_age > 18)
            should_travel = ctx.local_quality < local_food_exit * 0.52 or ctx.local_health < 0.10 or sheep.state_age > 260
            if should_travel:
                sheep.state = BehaviourState.TRAVELLING
                sheep.state_age = 0
                return
            if (not keep and should_walk) or stale_patch:
                sheep.state = BehaviourState.WALKING
                sheep.state_age = 0
                return
            return

        if sheep.state == BehaviourState.WALKING:
            if sheep.state_age < 10:
                return
            if ctx.local_quality >= local_food_exit * 0.84 and ctx.cell_quality > 0.09 and ctx.local_health > 0.18:
                sheep.state = BehaviourState.GRAZING
                sheep.state_age = 0
                return
            if ctx.local_quality < local_food_exit * 0.55 and sheep.state_age >= 20:
                sheep.state = BehaviourState.TRAVELLING
                sheep.state_age = 0
                return
            return

        if sheep.state == BehaviourState.TRAVELLING:
            if sheep.state_age < 16:
                return
            if ctx.local_quality >= local_food_exit * 0.75:
                sheep.state = BehaviourState.WALKING
                sheep.state_age = 0
            return

        sheep.state = BehaviourState.GRAZING
        sheep.state_age = 0
        return

    # scarce scenario
    if sheep.state == BehaviourState.GRAZING:
        if ctx.local_quality < local_food_enter * 0.85 or ctx.cell_quality < 0.08 or sheep.state_age > 24:
            sheep.state = BehaviourState.WALKING
            sheep.state_age = 0
        return

    if sheep.state == BehaviourState.WALKING:
        if sheep.state_age < 8:
            return
        if ctx.local_quality >= local_food_enter and ctx.cell_quality > 0.09:
            sheep.state = BehaviourState.GRAZING
            sheep.state_age = 0
            return
        if ctx.local_quality < local_food_enter * 0.75:
            sheep.state = BehaviourState.TRAVELLING
            sheep.state_age = 0
        return

    if sheep.state == BehaviourState.TRAVELLING:
        if sheep.state_age < 24:
            return
        if ctx.local_quality >= local_food_enter * 0.95:
            sheep.state = BehaviourState.WALKING
            sheep.state_age = 0
        return

    sheep.state = BehaviourState.WALKING if ctx.local_quality < local_food_enter else BehaviourState.GRAZING
    sheep.state_age = 0


def desired_velocity(
    sheep: SheepAgent,
    flock: list[SheepAgent],
    env: FieldEnvironment,
    food: FoodLandscape,
    ctx: BehaviourContext,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    boundary = env.boundary_force(sheep.position) * cfg.field.boundary_turn_strength
    terrain_force = -env.terrain_gradient(*sheep.position) * cfg.field.terrain_strength
    shade_force = env.shade_gradient(*sheep.position) * cfg.field.shade_strength * max(0.0, 0.45 - ctx.active_factor)

    prev_dir = safe_unit(sheep.velocity) if np.linalg.norm(sheep.velocity) > 1e-9 else sheep.heading_vector()
    mem_dir = memory_vector(sheep, ctx)
    home_dir = safe_unit(ctx.home_target - sheep.position)
    poor_patch_push = safe_unit(ctx.sensory_gradient) * max(0.0, 0.22 - ctx.cell_quality)

    if sheep.state == BehaviourState.GRAZING:
        social = state_social_force(
            sheep, flock, cfg.flock,
            attraction_scale=0.15 * sheep.sociability if cfg.scenario == "abundant" else 0.34 * sheep.sociability,
            alignment_scale=0.08 * sheep.sociability if cfg.scenario == "abundant" else 0.18 * sheep.sociability,
            repulsion_scale=1.55 if cfg.scenario == "abundant" else 1.30,
            spacing_scale=sheep.preferred_spacing,
        )
        raw_dir = 1.16 * ctx.sensory_gradient + 0.28 * poor_patch_push + 0.08 * home_dir * sheep.site_fidelity + 0.05 * mem_dir + 0.05 * social + 0.03 * prev_dir
        speed = cfg.flock.grazing_speed * sheep.movement_vigor * (0.58 + 0.48 * ctx.active_factor)
        turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 1.35 * sheep.turning_bias)
        inertia = 0.08 if cfg.scenario == "abundant" else 0.16
        base_dir = safe_unit(raw_dir if np.linalg.norm(raw_dir) > 1e-9 else prev_dir)

    elif sheep.state == BehaviourState.WALKING:
        social = state_social_force(
            sheep, flock, cfg.flock,
            attraction_scale=0.08 * sheep.sociability if cfg.scenario == "abundant" else 0.12 * sheep.sociability,
            alignment_scale=0.04 * sheep.sociability if cfg.scenario == "abundant" else 0.08 * sheep.sociability,
            repulsion_scale=0.95,
            spacing_scale=sheep.preferred_spacing,
        )
        raw_dir = 0.44 * prev_dir + 0.28 * ctx.sensory_gradient + 0.18 * poor_patch_push + 0.14 * mem_dir + 0.12 * home_dir * sheep.site_fidelity + 0.04 * social
        speed = cfg.flock.walking_speed * sheep.movement_vigor * (0.86 + 0.08 * ctx.active_factor)
        turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.55 * sheep.turning_bias)
        inertia = 0.48 if cfg.scenario == "abundant" else 0.56
        base_dir = safe_unit(raw_dir if np.linalg.norm(raw_dir) > 1e-9 else prev_dir)

    elif sheep.state == BehaviourState.TRAVELLING:
        if cfg.scenario == "abundant":
            social = state_social_force(
                sheep, flock, cfg.flock,
                attraction_scale=0.02 * sheep.sociability,
                alignment_scale=0.02 * sheep.sociability,
                repulsion_scale=0.75,
                spacing_scale=sheep.preferred_spacing,
            )
            raw_dir = 0.58 * prev_dir + 0.22 * mem_dir + 0.16 * home_dir * sheep.site_fidelity + 0.24 * ctx.sensory_gradient + 0.18 * poor_patch_push + 0.02 * social
            speed = cfg.flock.travel_speed * sheep.movement_vigor * (0.88 + 0.04 * ctx.active_factor)
            turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.32 * sheep.turning_bias)
            inertia = 0.70
        else:
            social = state_social_force(
                sheep, flock, cfg.flock,
                attraction_scale=0.06 * sheep.sociability,
                alignment_scale=0.06 * sheep.sociability,
                repulsion_scale=0.90,
                spacing_scale=sheep.preferred_spacing,
            )
            raw_dir = 0.78 * prev_dir + 0.42 * mem_dir + 0.18 * ctx.sensory_gradient + 0.04 * social
            speed = cfg.flock.travel_speed * sheep.movement_vigor * (1.00 + 0.24 * ctx.active_factor)
            turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.26 * sheep.turning_bias)
            inertia = 0.90
        base_dir = safe_unit(raw_dir if np.linalg.norm(raw_dir) > 1e-9 else prev_dir)

    elif sheep.state == BehaviourState.REGROUPING:
        to_centroid = safe_unit(ctx.flock_centroid - sheep.position)
        if cfg.scenario == "abundant":
            social = state_social_force(
                sheep, flock, cfg.flock,
                attraction_scale=0.10 * sheep.sociability,
                alignment_scale=0.04 * sheep.sociability,
                repulsion_scale=1.05,
                spacing_scale=sheep.preferred_spacing,
            )
            raw_dir = 0.14 * to_centroid + 0.08 * social + 0.32 * ctx.sensory_gradient + 0.24 * prev_dir
            speed = cfg.flock.regroup_speed * sheep.movement_vigor * 0.55
            turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.32 * sheep.turning_bias)
            inertia = 0.40
        else:
            social = state_social_force(
                sheep, flock, cfg.flock,
                attraction_scale=0.55 * sheep.sociability,
                alignment_scale=0.20 * sheep.sociability,
                repulsion_scale=1.00,
                spacing_scale=sheep.preferred_spacing,
            )
            raw_dir = 0.46 * to_centroid + 0.24 * social + 0.14 * ctx.sensory_gradient + 0.16 * prev_dir
            speed = cfg.flock.regroup_speed * sheep.movement_vigor * (0.72 + 0.18 * ctx.active_factor)
            turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.38 * sheep.turning_bias)
            inertia = 0.52
        base_dir = safe_unit(raw_dir if np.linalg.norm(raw_dir) > 1e-9 else to_centroid)

    else:
        base_dir = prev_dir
        speed = cfg.flock.resting_speed
        turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.10)
        inertia = 0.10

    env_dir = safe_unit(base_dir + boundary + terrain_force + shade_force)
    if np.linalg.norm(env_dir) < 1e-9:
        env_dir = prev_dir
    env_dir = rotate(env_dir, turn_noise)
    desired = safe_unit(env_dir) * clamp(speed, 0.0, cfg.flock.max_speed)
    blended = inertia * sheep.velocity + (1.0 - inertia) * desired
    if np.linalg.norm(blended) < 1e-9:
        blended = desired
    final_speed = clamp(float(np.linalg.norm(blended)), 0.0, cfg.flock.max_speed)
    return safe_unit(blended) * final_speed


def state_social_force(
    sheep: SheepAgent,
    flock: list[SheepAgent],
    cfg: FlockConfig,
    attraction_scale: float = 1.0,
    alignment_scale: float = 1.0,
    repulsion_scale: float = 1.0,
    spacing_scale: float = 1.0,
) -> np.ndarray:
    repulsion = np.zeros(2, dtype=float)
    attraction = np.zeros(2, dtype=float)
    alignment = np.zeros(2, dtype=float)
    count = 0
    effective_repulsion_radius = cfg.repulsion_radius * spacing_scale

    for other in flock:
        if other.sheep_id == sheep.sheep_id:
            continue
        offset = other.position - sheep.position
        dist = np.linalg.norm(offset)
        if dist < 1e-9 or dist > cfg.neighbour_radius:
            continue
        count += 1
        if dist < effective_repulsion_radius:
            repulsion -= safe_unit(offset) * (effective_repulsion_radius - dist) / max(effective_repulsion_radius, 1e-9)
        attraction += safe_unit(offset)
        if np.linalg.norm(other.velocity) > 1e-9:
            alignment += safe_unit(other.velocity)

    if count == 0:
        return np.zeros(2, dtype=float)

    attraction = attraction / count
    alignment = alignment / count
    social = (
        cfg.repulsion_weight * repulsion_scale * repulsion
        + cfg.attraction_weight * attraction_scale * attraction
        + cfg.alignment_weight * alignment_scale * alignment
    )
    return social


def social_force(sheep: SheepAgent, flock: list[SheepAgent], cfg: FlockConfig) -> np.ndarray:
    return state_social_force(sheep, flock, cfg)


def memory_vector(sheep: SheepAgent, ctx: BehaviourContext) -> np.ndarray:
    return safe_unit(ctx.memory_target - sheep.position)


def heading_to_velocity(heading: float, speed: float) -> np.ndarray:
    return np.array([np.cos(heading) * speed, np.sin(heading) * speed], dtype=float)
