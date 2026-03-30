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
    sensory_gradient: np.ndarray
    flock_centroid: np.ndarray
    flock_spread: float
    neighbour_count: int
    active_factor: float
    memory_target: np.ndarray
    shade_value: float
    terrain_value: float


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
            speed = rng.uniform(0.03, 0.16)
            state = BehaviourState.GRAZING
        else:
            speed = rng.uniform(0.10, 0.28)
            state = BehaviourState.TRAVELLING

        vel = heading_to_velocity(heading, speed)
        memory_map = np.zeros((cfg.field.grid_rows, cfg.field.grid_cols), dtype=float)

        flock.append(
            SheepAgent(
                sheep_id=i,
                position=pos.astype(float),
                velocity=vel,
                heading=heading,
                state=state,
                memory_map=memory_map,
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

    sheep_bias = 1.0 + 0.10 * np.sin(0.73 * sheep.sheep_id + 0.11)
    local_food_enter = tcfg.local_food_enter_graze * sheep_bias
    local_food_exit = tcfg.local_food_exit_graze * sheep_bias

    if isolated or far_from_group:
        sheep.separation_steps += 1
    else:
        sheep.separation_steps = max(0, sheep.separation_steps - 1)

    if ctx.active_factor < cfg.circadian.resting_threshold:
        if sheep.state != BehaviourState.REGROUPING:
            sheep.state = BehaviourState.RESTING
            sheep.state_age = 0
            return

    # Regrouping must be brief and must force an exit.
    if sheep.state == BehaviourState.REGROUPING:
        if cfg.scenario == "abundant":
            min_regroup_steps = 4
            must_exit_steps = 10
        else:
            min_regroup_steps = 8
            must_exit_steps = 16

        ready_to_exit = (
            sheep.state_age >= min_regroup_steps
            and close_enough
            and socially_connected
            and flock_compact
        )

        timed_out = sheep.state_age >= must_exit_steps

        if ready_to_exit or timed_out:
            sheep.separation_steps = 0
            sheep.regroup_cooldown = 24 if cfg.scenario == "abundant" else 12

            if cfg.scenario == "abundant":
                # After regrouping in rich pasture, prefer returning to grazing,
                # not orbiting around the centroid.
                if ctx.local_food >= local_food_exit * 0.85 and ctx.local_health > 0.24:
                    sheep.state = BehaviourState.GRAZING
                else:
                    sheep.state = BehaviourState.TRAVELLING
            else:
                if ctx.local_food >= local_food_enter and ctx.local_health > 0.30:
                    sheep.state = BehaviourState.GRAZING
                else:
                    sheep.state = BehaviourState.TRAVELLING

            sheep.state_age = 0
            return

        return

    if sheep.state == BehaviourState.RESTING:
        if ctx.active_factor >= cfg.circadian.resting_threshold:
            if ctx.local_food >= local_food_enter and ctx.local_health > 0.30:
                sheep.state = BehaviourState.GRAZING
            else:
                sheep.state = BehaviourState.TRAVELLING
            sheep.state_age = 0
        return

    # In abundant fields, regroup only for true sustained stragglers, and never
    # immediately after leaving regrouping.
    if sheep.regroup_cooldown == 0:
        if cfg.scenario == "abundant":
            if (
                sheep.separation_steps >= 20
                and isolated
                and dist_to_centroid > cfg.flock.neighbour_radius * 2.3
                and ctx.local_food < local_food_exit * 0.75
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
        should_graze = (
            ctx.local_food >= local_food_exit
            and ctx.local_health > 0.24
            and sheep.energy > 0.20
        )

        should_travel = (
            ctx.local_food < local_food_exit * 0.80
            or ctx.local_health < 0.20
            or sheep.state_age > 130
        )

        if sheep.state == BehaviourState.GRAZING:
            if should_travel:
                sheep.state = BehaviourState.TRAVELLING
                sheep.state_age = 0
                return
            return

        if sheep.state == BehaviourState.TRAVELLING:
            min_travel_steps = 40
            if sheep.state_age < min_travel_steps:
                return

            if should_graze:
                sheep.state = BehaviourState.GRAZING
                sheep.state_age = 0
                return
            return

        sheep.state = BehaviourState.GRAZING if should_graze else BehaviourState.TRAVELLING
        sheep.state_age = 0
        return

    should_graze = (
        ctx.local_food >= local_food_enter
        and ctx.local_health > 0.34
        and sheep.energy > 0.22
    )

    should_travel = (
        ctx.local_food < local_food_enter
        or ctx.local_health < 0.30
        or sheep.energy < 0.38
        or sheep.state_age > 30
    )

    if sheep.state == BehaviourState.GRAZING:
        if should_travel:
            sheep.state = BehaviourState.TRAVELLING
            sheep.state_age = 0
            return
        return

    if sheep.state == BehaviourState.TRAVELLING:
        min_travel_steps = 24
        if sheep.state_age < min_travel_steps:
            return

        if should_graze:
            sheep.state = BehaviourState.GRAZING
            sheep.state_age = 0
            return
        return

    sheep.state = BehaviourState.GRAZING if should_graze else BehaviourState.TRAVELLING
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

    if sheep.state == BehaviourState.GRAZING:
        social = state_social_force(
            sheep,
            flock,
            cfg.flock,
            attraction_scale=0.24 if cfg.scenario == "abundant" else 0.42,
            alignment_scale=0.14 if cfg.scenario == "abundant" else 0.24,
            repulsion_scale=1.25,
        )

        raw_dir = (
            1.04 * ctx.sensory_gradient
            + 0.06 * mem_dir
            + 0.10 * social
            + 0.08 * prev_dir
        )
        base_dir = safe_unit(raw_dir if np.linalg.norm(raw_dir) > 1e-9 else prev_dir)
        speed = cfg.flock.grazing_speed * (0.52 + 0.50 * ctx.active_factor)
        turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 1.25)
        inertia = 0.22

    elif sheep.state == BehaviourState.TRAVELLING:
        if cfg.scenario == "abundant":
            social = state_social_force(
                sheep,
                flock,
                cfg.flock,
                attraction_scale=0.01,
                alignment_scale=0.01,
                repulsion_scale=0.75,
            )
            raw_dir = (
                0.92 * prev_dir
                + 0.12 * mem_dir
                + 0.04 * ctx.sensory_gradient
                + 0.00 * social
            )
            speed = cfg.flock.travel_speed * (0.98 + 0.06 * ctx.active_factor)
            turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.18)
            inertia = 0.96
        else:
            social = state_social_force(
                sheep,
                flock,
                cfg.flock,
                attraction_scale=0.06,
                alignment_scale=0.06,
                repulsion_scale=0.90,
            )
            raw_dir = (
                0.82 * prev_dir
                + 0.46 * mem_dir
                + 0.18 * ctx.sensory_gradient
                + 0.04 * social
            )
            speed = cfg.flock.travel_speed * (1.00 + 0.24 * ctx.active_factor)
            turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.26)
            inertia = 0.92

        base_dir = safe_unit(raw_dir if np.linalg.norm(raw_dir) > 1e-9 else prev_dir)

    elif sheep.state == BehaviourState.REGROUPING:
        to_centroid = safe_unit(ctx.flock_centroid - sheep.position)

        if cfg.scenario == "abundant":
            # Very light regrouping: enough to reconnect, not enough to pin them at the centre.
            social = state_social_force(
                sheep,
                flock,
                cfg.flock,
                attraction_scale=0.16,
                alignment_scale=0.06,
                repulsion_scale=1.05,
            )
            raw_dir = (
                0.18 * to_centroid
                + 0.10 * social
                + 0.28 * ctx.sensory_gradient
                + 0.24 * prev_dir
            )
            speed = cfg.flock.regroup_speed * 0.62
            turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.30)
            inertia = 0.44
        else:
            social = state_social_force(
                sheep,
                flock,
                cfg.flock,
                attraction_scale=0.55,
                alignment_scale=0.20,
                repulsion_scale=1.00,
            )
            raw_dir = (
                0.46 * to_centroid
                + 0.24 * social
                + 0.14 * ctx.sensory_gradient
                + 0.16 * prev_dir
            )
            speed = cfg.flock.regroup_speed * (0.72 + 0.18 * ctx.active_factor)
            turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.38)
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
) -> np.ndarray:
    repulsion = np.zeros(2, dtype=float)
    attraction = np.zeros(2, dtype=float)
    alignment = np.zeros(2, dtype=float)
    count = 0

    for other in flock:
        if other.sheep_id == sheep.sheep_id:
            continue

        offset = other.position - sheep.position
        dist = np.linalg.norm(offset)
        if dist < 1e-9 or dist > cfg.neighbour_radius:
            continue

        count += 1

        if dist < cfg.repulsion_radius:
            repulsion -= safe_unit(offset) * (cfg.repulsion_radius - dist) / cfg.repulsion_radius

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