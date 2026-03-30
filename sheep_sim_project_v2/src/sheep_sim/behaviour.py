from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sheep_sim.agents import BehaviourState, SheepAgent
from sheep_sim.config import FlockConfig, FoodConfig, SimulationConfig, StateTransitionConfig
from sheep_sim.environment import FieldEnvironment
from sheep_sim.food import FoodLandscape
from sheep_sim.utils import clamp, rotate, safe_unit


@dataclass(slots=True)
class BehaviourContext:
    local_food: float
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
    spread = 8.0
    for i in range(cfg.flock.n_sheep):
        pos = centre + rng.normal(0.0, spread, size=2)
        pos[0] = clamp(float(pos[0]), 0.0, cfg.field.width)
        pos[1] = clamp(float(pos[1]), 0.0, cfg.field.height)
        heading = float(rng.uniform(-np.pi, np.pi))
        speed = rng.uniform(0.05, 0.35)
        vel = heading_to_velocity(heading, speed)
        state = BehaviourState.GRAZING if cfg.scenario == "abundant" else BehaviourState.TRAVELLING
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
    scenario = cfg.scenario

    sheep.state_age += 1

    if ctx.active_factor < cfg.circadian.resting_threshold and sheep.state != BehaviourState.REGROUPING:
        sheep.state = BehaviourState.RESTING
        sheep.state_age = 0
        return

    dispersed = ctx.flock_spread > tcfg.dispersion_regroup_threshold * max(cfg.field.width, cfg.field.height)
    isolated = ctx.neighbour_count <= 2
    if dispersed or isolated:
        sheep.state = BehaviourState.REGROUPING
        sheep.state_age = 0
        return

    if sheep.state == BehaviourState.RESTING and ctx.active_factor >= cfg.circadian.resting_threshold:
        sheep.state = BehaviourState.GRAZING if ctx.local_food >= tcfg.local_food_exit_graze else BehaviourState.TRAVELLING
        sheep.state_age = 0
        return

    if scenario == "abundant":
        if ctx.local_food >= tcfg.local_food_exit_graze:
            sheep.state = BehaviourState.GRAZING
        elif sheep.state_age > 40:
            sheep.state = BehaviourState.TRAVELLING
    else:
        if ctx.local_food >= tcfg.local_food_enter_graze:
            sheep.state = BehaviourState.GRAZING
        elif sheep.state_age > 25 or sheep.energy < 0.35:
            sheep.state = BehaviourState.TRAVELLING

    if ctx.neighbour_count >= max(6, cfg.flock.n_sheep // 3) and ctx.local_food < tcfg.crowding_regroup_threshold:
        sheep.state = BehaviourState.TRAVELLING



def desired_velocity(
    sheep: SheepAgent,
    flock: list[SheepAgent],
    env: FieldEnvironment,
    food: FoodLandscape,
    ctx: BehaviourContext,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    social = social_force(sheep, flock, cfg.flock)
    boundary = env.boundary_force(sheep.position) * cfg.field.boundary_turn_strength
    terrain_force = -env.terrain_gradient(*sheep.position) * cfg.field.terrain_strength
    shade_force = env.shade_gradient(*sheep.position) * cfg.field.shade_strength * max(0.0, 0.4 - ctx.active_factor)

    if sheep.state == BehaviourState.GRAZING:
        base_dir = safe_unit(0.85 * ctx.sensory_gradient + 0.30 * social + 0.35 * memory_vector(sheep, ctx))
        speed = cfg.flock.grazing_speed * (0.65 + 0.7 * ctx.active_factor)
        turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 1.25)
    elif sheep.state == BehaviourState.TRAVELLING:
        explore_dir = safe_unit(
            0.65 * ctx.sensory_gradient
            + cfg.transitions.memory_bias * memory_vector(sheep, ctx)
            + 0.20 * sheep.heading_vector()
            + 0.22 * social
        )
        base_dir = explore_dir if np.linalg.norm(explore_dir) > 1e-9 else sheep.heading_vector()
        speed = cfg.flock.travel_speed * (0.85 + 0.35 * ctx.active_factor)
        turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.70)
    elif sheep.state == BehaviourState.REGROUPING:
        to_centroid = safe_unit(ctx.flock_centroid - sheep.position)
        base_dir = safe_unit(0.80 * to_centroid + 0.30 * social + 0.10 * ctx.sensory_gradient)
        speed = cfg.flock.regroup_speed * (0.85 + 0.30 * ctx.active_factor)
        turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.55)
    else:
        base_dir = sheep.heading_vector()
        speed = cfg.flock.resting_speed
        turn_noise = rng.normal(0.0, cfg.flock.stochastic_turn_std * 0.20)

    total_dir = safe_unit(base_dir + boundary + terrain_force + shade_force)
    if np.linalg.norm(total_dir) < 1e-9:
        total_dir = sheep.heading_vector()
    total_dir = rotate(total_dir, turn_noise)
    return safe_unit(total_dir) * clamp(speed, 0.0, cfg.flock.max_speed)



def social_force(sheep: SheepAgent, flock: list[SheepAgent], cfg: FlockConfig) -> np.ndarray:
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
        alignment += safe_unit(other.velocity)
    if count == 0:
        return np.zeros(2, dtype=float)
    attraction = attraction / count
    alignment = alignment / count
    social = (
        cfg.repulsion_weight * repulsion
        + cfg.attraction_weight * attraction
        + cfg.alignment_weight * alignment
    )
    return social



def memory_vector(sheep: SheepAgent, ctx: BehaviourContext) -> np.ndarray:
    return safe_unit(ctx.memory_target - sheep.position)



def heading_to_velocity(heading: float, speed: float) -> np.ndarray:
    return np.array([np.cos(heading) * speed, np.sin(heading) * speed], dtype=float)
