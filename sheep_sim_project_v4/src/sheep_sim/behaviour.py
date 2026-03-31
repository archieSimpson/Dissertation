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
            speed = rng.uniform(0.01, 0.08)
            state = BehaviourState.GRAZING
            sociability = float(rng.uniform(0.90, 1.18))
            boldness = float(rng.uniform(0.86, 1.12))
            movement_vigor = float(rng.uniform(0.90, 1.10))
            turning_bias = float(rng.uniform(1.00, 1.35))
            patch_leave_bias = float(rng.uniform(0.88, 1.08))
            preferred_spacing = float(rng.uniform(0.95, 1.18))
        else:
            speed = rng.uniform(0.08, 0.24)
            state = BehaviourState.TRAVELLING
            sociability = float(rng.uniform(0.75, 1.15))
            boldness = float(rng.uniform(0.80, 1.20))
            movement_vigor = float(rng.uniform(0.85, 1.15))
            turning_bias = float(rng.uniform(0.85, 1.20))
            patch_leave_bias = float(rng.uniform(0.85, 1.15))
            preferred_spacing = float(rng.uniform(0.85, 1.20))

        flock.append(
            SheepAgent(
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
    close_enough = dist_to_centroid < cfg.flock.neighbour_radius * 1.05
    far_from_group = dist_to_centroid > cfg.flock.neighbour_radius * 2.0
    isolated = ctx.neighbour_count <= 0
    socially_connected = ctx.neighbour_count >= 2
    flock_compact = ctx.flock_spread < 0.14 * field_scale

    local_food_enter = tcfg.local_food_enter_graze * sheep.site_fidelity
    local_food_exit = tcfg.local_food_exit_graze * sheep.site_fidelity

    if isolated or far_from_group:
        sheep.separation_steps += 1
    else:
        sheep.separation_steps = max(0, sheep.separation_steps - 1)

    # Softer abundant resting: not a hard rectangular block.
    if cfg.scenario == "abundant":
        if ctx.active_factor < cfg.circadian.resting_threshold:
            if sheep.state != BehaviourState.REGROUPING:
                rest_p = 0.18 + 0.30 * (cfg.circadian.resting_threshold - ctx.active_factor) / max(cfg.circadian.resting_threshold, 1e-9)
                if sheep.state == BehaviourState.GRAZING:
                    rest_p += 0.12
                if rng.random() < clamp(rest_p, 0.0, 0.72):
                    sheep.state = BehaviourState.RESTING
                    sheep.state_age = 0
                    return
        elif sheep.state == BehaviourState.RESTING:
            wake_p = clamp(0.22 + 0.90 * (ctx.active_factor - cfg.circadian.resting_threshold), 0.10, 0.95)
            if rng.random() < wake_p:
                sheep.state = (
                    BehaviourState.GRAZING
                    if ctx.local_quality >= local_food_enter * 0.82 and ctx.cell_quality > 0.07
                    else BehaviourState.WALKING
                )
                sheep.state_age = 0
                return
    else:
        if ctx.active_factor < cfg.circadian.resting_threshold:
            if sheep.state != BehaviourState.REGROUPING:
                sheep.state = BehaviourState.RESTING
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

    if sheep.state == BehaviourState.REGROUPING:
        min_regroup_steps = 5 if cfg.scenario == "abundant" else 8
        max_regroup_steps = 12 if cfg.scenario == "abundant" else 16
        if (
            (sheep.state_age >= min_regroup_steps and close_enough and socially_connected and flock_compact)
            or sheep.state_age >= max_regroup_steps
        ):
            sheep.separation_steps = 0
            sheep.regroup_cooldown = 16 if cfg.scenario == "abundant" else 12
            sheep.state = (
                BehaviourState.GRAZING
                if cfg.scenario == "abundant" and ctx.local_quality >= local_food_exit * 0.72 and ctx.cell_quality > 0.07
                else BehaviourState.WALKING if cfg.scenario == "abundant" else BehaviourState.TRAVELLING
            )
            sheep.state_age = 0
        return

    if sheep.regroup_cooldown == 0:
        if cfg.scenario == "abundant":
            if (
                sheep.separation_steps >= int(12 * sheep.boldness)
                and (isolated or far_from_group)
                and ctx.local_quality < local_food_exit * 0.88
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
        stale_steps = int(max(18, tcfg.stale_patch_steps_abundant * sheep.patch_leave_bias * 1.25))
        stale_patch = ctx.cell_quality < tcfg.stale_patch_quality_threshold and sheep.state_age >= stale_steps
        low_gain = ctx.recent_intake_rate < 0.0020 and sheep.state_age >= stale_steps

        if sheep.state == BehaviourState.GRAZING:
            keep = (
                ctx.local_quality >= local_food_exit * 0.72
                and ctx.local_health > 0.14
                and ctx.cell_quality > 0.07
                and sheep.energy > 0.12
            )
            should_walk = (
                stale_patch
                or low_gain
                or ctx.cell_quality < 0.05
                or (ctx.local_quality < local_food_exit * 0.62 and sheep.state_age > 34)
            )
            should_travel = (
                ctx.local_quality < local_food_exit * 0.42
                or ctx.local_health < 0.07
                or sheep.state_age > 420
            )

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
            if sheep.state_age < 8:
                return
            if ctx.local_quality >= local_food_exit * 0.68 and ctx.cell_quality > 0.07 and ctx.local_health > 0.14:
                sheep.state = BehaviourState.GRAZING
                sheep.state_age = 0
                return
            if ctx.local_quality < local_food_exit * 0.46 and sheep.state_age >= 24:
                sheep.state = BehaviourState.TRAVELLING
                sheep.state_age = 0
            return

        if sheep.state == BehaviourState.TRAVELLING:
            if sheep.state_age < 14:
                return
            if ctx.local_quality >= local_food_exit * 0.58:
                sheep.state = BehaviourState.WALKING
                sheep.state_age = 0
            return

        if sheep.state == BehaviourState.RESTING:
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

    if sheep.state == BehaviourState.RESTING:
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
    shade_force = env.shade_gradient(*sheep.position) * cfg.field.shade_strength * max(0.0, 0.40 - ctx.active_factor)

    prev_dir = safe_unit(sheep.velocity) if np.linalg.norm(sheep.velocity) > 1e-9 else sheep.heading_vector()
    mem_dir = memory_vector(sheep, ctx)
    home_dir = safe_unit(ctx.home_target - sheep.position)
    grad_dir = safe_unit(ctx.sensory_gradient)

    prospect_dir = prospect_direction(sheep, food, cfg, rng)
    residence = patch_residence_weight(ctx)
    exit_pressure = grazing_exit_pressure(ctx)

    if sheep.state == BehaviourState.GRAZING:
        social = state_social_force(
            sheep,
            flock,
            cfg.flock,
            attraction_scale=0.28 * sheep.sociability if cfg.scenario == "abundant" else 0.34 * sheep.sociability,
            alignment_scale=0.10 * sheep.sociability if cfg.scenario == "abundant" else 0.18 * sheep.sociability,
            repulsion_scale=1.08 if cfg.scenario == "abundant" else 1.30,
            spacing_scale=1.06 * sheep.preferred_spacing if cfg.scenario == "abundant" else sheep.preferred_spacing,
        )

        tangent = rotate(grad_dir if np.linalg.norm(grad_dir) > 1e-9 else prev_dir, np.pi / 2.0)
        exploratory = soft_jitter(prev_dir, 0.65 * cfg.flock.stochastic_turn_std * sheep.turning_bias, rng)

        raw_dir = (
            0.28 * grad_dir +
            0.18 * tangent * residence +
            0.18 * social +
            0.10 * exploratory * residence +
            0.10 * mem_dir * exit_pressure +
            0.08 * prospect_dir * exit_pressure +
            0.05 * prev_dir +
            0.03 * home_dir * sheep.site_fidelity
        )

        base_dir = safe_unit(raw_dir if np.linalg.norm(raw_dir) > 1e-9 else exploratory)
        base_dir = soft_jitter(
            base_dir,
            1.35 * cfg.flock.stochastic_turn_std * sheep.turning_bias,
            rng,
        )

        speed = cfg.flock.grazing_speed * sheep.movement_vigor * (
            0.18 + 0.22 * ctx.active_factor + 0.14 * exit_pressure
        )
        inertia = 0.03 if cfg.scenario == "abundant" else 0.12

    elif sheep.state == BehaviourState.WALKING:
        social = state_social_force(
            sheep,
            flock,
            cfg.flock,
            attraction_scale=0.20 * sheep.sociability if cfg.scenario == "abundant" else 0.12 * sheep.sociability,
            alignment_scale=0.08 * sheep.sociability if cfg.scenario == "abundant" else 0.08 * sheep.sociability,
            repulsion_scale=0.96,
            spacing_scale=1.00 * sheep.preferred_spacing,
        )

        exploratory = soft_jitter(prev_dir, 0.48 * cfg.flock.stochastic_turn_std * sheep.turning_bias, rng)

        raw_dir = (
            0.18 * prev_dir +
            0.16 * grad_dir +
            0.16 * prospect_dir +
            0.14 * mem_dir +
            0.16 * social +
            0.10 * exploratory +
            0.06 * safe_unit(ctx.flock_centroid - sheep.position) +
            0.04 * home_dir * sheep.site_fidelity
        )

        base_dir = safe_unit(raw_dir if np.linalg.norm(raw_dir) > 1e-9 else exploratory)
        base_dir = soft_jitter(
            base_dir,
            0.72 * cfg.flock.stochastic_turn_std * sheep.turning_bias,
            rng,
        )

        speed = cfg.flock.walking_speed * sheep.movement_vigor * (
            0.38 + 0.12 * ctx.active_factor + 0.10 * exit_pressure
        )
        inertia = 0.18 if cfg.scenario == "abundant" else 0.56

    elif sheep.state == BehaviourState.TRAVELLING:
        social = state_social_force(
            sheep,
            flock,
            cfg.flock,
            attraction_scale=0.18 * sheep.sociability if cfg.scenario == "abundant" else 0.06 * sheep.sociability,
            alignment_scale=0.12 * sheep.sociability if cfg.scenario == "abundant" else 0.06 * sheep.sociability,
            repulsion_scale=0.82 if cfg.scenario == "abundant" else 0.90,
            spacing_scale=0.96 * sheep.preferred_spacing,
        )

        raw_dir = (
            0.30 * prev_dir +
            0.22 * mem_dir +
            0.20 * prospect_dir +
            0.08 * grad_dir +
            0.14 * social +
            0.04 * safe_unit(ctx.flock_centroid - sheep.position) +
            0.02 * home_dir * sheep.site_fidelity
        )

        base_dir = safe_unit(raw_dir if np.linalg.norm(raw_dir) > 1e-9 else prev_dir)
        base_dir = soft_jitter(
            base_dir,
            0.34 * cfg.flock.stochastic_turn_std * sheep.turning_bias,
            rng,
        )

        speed = cfg.flock.travel_speed * sheep.movement_vigor * (
            0.54 + 0.12 * ctx.active_factor
        )
        inertia = 0.42 if cfg.scenario == "abundant" else 0.90

    elif sheep.state == BehaviourState.REGROUPING:
        to_centroid = safe_unit(ctx.flock_centroid - sheep.position)
        social = state_social_force(
            sheep,
            flock,
            cfg.flock,
            attraction_scale=0.44 * sheep.sociability if cfg.scenario == "abundant" else 0.55 * sheep.sociability,
            alignment_scale=0.14 * sheep.sociability if cfg.scenario == "abundant" else 0.20 * sheep.sociability,
            repulsion_scale=0.94,
            spacing_scale=1.04 * sheep.preferred_spacing,
        )

        raw_dir = (
            0.44 * to_centroid +
            0.22 * social +
            0.10 * prev_dir +
            0.10 * grad_dir +
            0.08 * mem_dir
        )

        base_dir = safe_unit(raw_dir if np.linalg.norm(raw_dir) > 1e-9 else to_centroid)
        base_dir = soft_jitter(
            base_dir,
            0.38 * cfg.flock.stochastic_turn_std * sheep.turning_bias,
            rng,
        )

        speed = cfg.flock.regroup_speed * sheep.movement_vigor * (
            0.64 + 0.10 * ctx.active_factor
        )
        inertia = 0.22 if cfg.scenario == "abundant" else 0.52

    else:  # RESTING
        micro_dir = soft_jitter(prev_dir, 0.90 * cfg.flock.stochastic_turn_std, rng)
        base_dir = safe_unit(0.55 * micro_dir + 0.25 * grad_dir + 0.20 * social_force(sheep, flock, cfg.flock))
        speed = cfg.flock.resting_speed * (0.24 + 0.18 * rng.random())
        inertia = 0.02

    env_dir = safe_unit(base_dir + boundary + terrain_force + shade_force)
    if np.linalg.norm(env_dir) < 1e-9:
        env_dir = prev_dir

    desired = safe_unit(env_dir) * clamp(speed, 0.0, cfg.flock.max_speed)
    blended = inertia * sheep.velocity + (1.0 - inertia) * desired

    if np.linalg.norm(blended) < 1e-9:
        blended = desired

    final_speed = float(np.linalg.norm(blended))
    if cfg.scenario == "abundant" and sheep.state in {BehaviourState.GRAZING, BehaviourState.RESTING}:
        final_speed = max(final_speed, 0.008 if sheep.state == BehaviourState.GRAZING else 0.002)

    final_speed = clamp(final_speed, 0.0, cfg.flock.max_speed)
    return safe_unit(blended) * final_speed


def prospect_direction(
    sheep: SheepAgent,
    food: FoodLandscape,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    radii = (10.0, 18.0, 28.0)
    angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    best_score = -np.inf
    best_dir = np.zeros(2, dtype=float)

    for r in radii:
        for a in angles:
            d = np.array([np.cos(a), np.sin(a)], dtype=float)
            probe = sheep.position + d * r
            probe[0] = clamp(float(probe[0]), 0.0, cfg.field.width)
            probe[1] = clamp(float(probe[1]), 0.0, cfg.field.height)

            q = food.local_quality(
                float(probe[0]),
                float(probe[1]),
                cfg.food.sensory_radius,
                cfg.field,
            )
            score = q - 0.010 * r
            if score > best_score:
                best_score = score
                best_dir = d

    return safe_unit(best_dir)


def patch_residence_weight(ctx: BehaviourContext) -> float:
    return clamp(
        0.55 * ctx.local_quality + 0.45 * ctx.cell_quality + 0.20 * ctx.local_health,
        0.0,
        1.0,
    )


def grazing_exit_pressure(ctx: BehaviourContext) -> float:
    poor_quality = max(0.0, 0.22 - ctx.cell_quality)
    poor_intake = max(0.0, 0.0035 - ctx.recent_intake_rate)
    return clamp(1.8 * poor_quality + 80.0 * poor_intake, 0.0, 1.0)


def soft_jitter(
    base_dir: np.ndarray,
    angle_std: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if np.linalg.norm(base_dir) < 1e-9:
        base_dir = np.array([1.0, 0.0], dtype=float)
    return safe_unit(rotate(base_dir, float(rng.normal(0.0, angle_std))))


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