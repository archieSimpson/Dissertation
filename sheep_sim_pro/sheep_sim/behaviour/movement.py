from __future__ import annotations

import numpy as np

from sheep_sim.core.agents import BehaviourState, SheepAgent
from sheep_sim.core.config import FlockConfig, SimulationConfig
from sheep_sim.core.utils import clamp, rotate, safe_unit
from sheep_sim.behaviour.states import BehaviourContext
from sheep_sim.behaviour.stochastic import get_effective_weights
from sheep_sim.environment.field import FieldEnvironment
from sheep_sim.environment.food import FoodLandscape

def desired_velocity(
    sheep: SheepAgent,
    flock: list[SheepAgent],
    env: FieldEnvironment,
    food: FoodLandscape,
    ctx: BehaviourContext,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    boundary  = env.boundary_force(sheep.position) * cfg.field.boundary_turn_strength
    terrain_f = -env.terrain_gradient(*sheep.position) * cfg.field.terrain_strength
    shade_f   = (
        env.shade_gradient(*sheep.position)
        * cfg.field.shade_strength
        * max(0.0, 0.40 - ctx.active_factor)
    )
    env_force = boundary + terrain_f + shade_f

    prev_dir     = (
        safe_unit(sheep.velocity)
        if np.linalg.norm(sheep.velocity) > 1e-9
        else sheep.heading_vector()
    )
    mem_dir      = safe_unit(ctx.memory_target - sheep.position)
    home_dir     = safe_unit(ctx.home_target - sheep.position)
    grad_dir     = safe_unit(ctx.sensory_gradient)
    prospect_dir = _prospect_direction(sheep, food, cfg, rng)
    residence    = _patch_residence_weight(ctx)
    exit_p       = _grazing_exit_pressure(ctx)

    if sheep.state == BehaviourState.GRAZING:
        base_dir, speed, inertia = _compute_grazing(
            sheep, flock, cfg, ctx, rng,
            prev_dir, grad_dir, mem_dir, home_dir, prospect_dir, residence, exit_p,
        )
    elif sheep.state == BehaviourState.WALKING:
        base_dir, speed, inertia = _compute_walking(
            sheep, flock, cfg, ctx, rng,
            prev_dir, grad_dir, mem_dir, home_dir, prospect_dir, exit_p,
        )
    elif sheep.state == BehaviourState.TRAVELLING:
        base_dir, speed, inertia = _compute_travelling(
            sheep, flock, cfg, ctx, rng,
            prev_dir, grad_dir, mem_dir, home_dir, prospect_dir,
        )
    elif sheep.state == BehaviourState.REGROUPING:
        base_dir, speed, inertia = _compute_regrouping(
            sheep, flock, cfg, ctx, rng,
            prev_dir, grad_dir, mem_dir,
        )
    else:
        base_dir, speed, inertia = _compute_resting(
            sheep, flock, cfg, ctx, rng, prev_dir, grad_dir
        )

    env_dir = safe_unit(base_dir + env_force)
    if np.linalg.norm(env_dir) < 1e-9:
        env_dir = prev_dir

    desired = safe_unit(env_dir) * clamp(speed, 0.0, cfg.flock.max_speed)
    blended = inertia * sheep.velocity + (1.0 - inertia) * desired
    if np.linalg.norm(blended) < 1e-9:
        blended = desired

    final_speed = float(np.linalg.norm(blended))
    if cfg.scenario == "abundant" and sheep.state in {
        BehaviourState.GRAZING, BehaviourState.RESTING
    }:
        min_sp = 0.008 if sheep.state == BehaviourState.GRAZING else 0.002
        final_speed = max(final_speed, min_sp)

    final_speed = clamp(final_speed, 0.0, cfg.flock.max_speed)
    return safe_unit(blended) * final_speed

def _compute_grazing(
    sheep, flock, cfg, ctx, rng,
    prev_dir, grad_dir, mem_dir, home_dir, prospect_dir, residence, exit_p,
) -> tuple[np.ndarray, float, float]:
    base_attr = 0.28 * sheep.sociability if cfg.scenario == "abundant" else 0.34 * sheep.sociability
    base_aln  = 0.10 * sheep.sociability if cfg.scenario == "abundant" else 0.18 * sheep.sociability
    base_rep  = 1.08 if cfg.scenario == "abundant" else 1.30
    attr, aln, rep = get_effective_weights(sheep, base_attr, base_aln, base_rep)

    social = _social_force(
        sheep, flock, cfg.flock,
        attr_scale=attr, aln_scale=aln, rep_scale=rep,
        spc_scale=1.06 * sheep.preferred_spacing if cfg.scenario == "abundant" else sheep.preferred_spacing,
    )
    tangent     = rotate(grad_dir if np.linalg.norm(grad_dir) > 1e-9 else prev_dir, np.pi / 2.0)
    exploratory = _jitter(prev_dir, 0.65 * cfg.flock.stochastic_turn_std * sheep.turning_bias, rng)

    raw = (
        0.28 * grad_dir
        + 0.18 * tangent * residence
        + 0.18 * social
        + 0.10 * exploratory * residence
        + 0.10 * mem_dir * exit_p
        + 0.08 * prospect_dir * exit_p
        + 0.05 * prev_dir
        + 0.03 * home_dir * sheep.site_fidelity
    )
    base = _jitter(
        safe_unit(raw if np.linalg.norm(raw) > 1e-9 else exploratory),
        1.35 * cfg.flock.stochastic_turn_std * sheep.turning_bias,
        rng,
    )
    speed   = cfg.flock.grazing_speed * sheep.movement_vigor * (
        0.18 + 0.22 * ctx.active_factor + 0.14 * exit_p
    )
    inertia = 0.03 if cfg.scenario == "abundant" else 0.12
    return base, speed, inertia

def _compute_walking(
    sheep, flock, cfg, ctx, rng,
    prev_dir, grad_dir, mem_dir, home_dir, prospect_dir, exit_p,
) -> tuple[np.ndarray, float, float]:
    base_attr = 0.20 * sheep.sociability if cfg.scenario == "abundant" else 0.12 * sheep.sociability
    base_aln  = 0.08 * sheep.sociability
    base_rep  = 0.96
    attr, aln, rep = get_effective_weights(sheep, base_attr, base_aln, base_rep)

    social = _social_force(
        sheep, flock, cfg.flock,
        attr_scale=attr, aln_scale=aln, rep_scale=rep,
        spc_scale=sheep.preferred_spacing,
    )
    exploratory = _jitter(prev_dir, 0.48 * cfg.flock.stochastic_turn_std * sheep.turning_bias, rng)
    raw = (
        0.18 * prev_dir
        + 0.16 * grad_dir
        + 0.16 * prospect_dir
        + 0.14 * mem_dir
        + 0.16 * social
        + 0.10 * exploratory
        + 0.06 * safe_unit(ctx.flock_centroid - sheep.position)
        + 0.04 * home_dir * sheep.site_fidelity
    )
    base = _jitter(
        safe_unit(raw if np.linalg.norm(raw) > 1e-9 else exploratory),
        0.72 * cfg.flock.stochastic_turn_std * sheep.turning_bias,
        rng,
    )
    speed   = cfg.flock.walking_speed * sheep.movement_vigor * (
        0.38 + 0.12 * ctx.active_factor + 0.10 * exit_p
    )
    inertia = 0.18 if cfg.scenario == "abundant" else 0.56
    return base, speed, inertia

def _compute_travelling(
    sheep, flock, cfg, ctx, rng,
    prev_dir, grad_dir, mem_dir, home_dir, prospect_dir,
) -> tuple[np.ndarray, float, float]:
    base_attr = 0.18 * sheep.sociability if cfg.scenario == "abundant" else 0.06 * sheep.sociability
    base_aln  = 0.12 * sheep.sociability if cfg.scenario == "abundant" else 0.06 * sheep.sociability
    base_rep  = 0.82 if cfg.scenario == "abundant" else 0.90
    attr, aln, rep = get_effective_weights(sheep, base_attr, base_aln, base_rep)

    social = _social_force(
        sheep, flock, cfg.flock,
        attr_scale=attr, aln_scale=aln, rep_scale=rep,
        spc_scale=0.96 * sheep.preferred_spacing,
    )
    raw = (
        0.30 * prev_dir
        + 0.22 * mem_dir
        + 0.20 * prospect_dir
        + 0.08 * grad_dir
        + 0.14 * social
        + 0.04 * safe_unit(ctx.flock_centroid - sheep.position)
        + 0.02 * home_dir * sheep.site_fidelity
    )
    base = _jitter(
        safe_unit(raw if np.linalg.norm(raw) > 1e-9 else prev_dir),
        0.34 * cfg.flock.stochastic_turn_std * sheep.turning_bias,
        rng,
    )
    speed   = cfg.flock.travel_speed * sheep.movement_vigor * (0.54 + 0.12 * ctx.active_factor)
    inertia = 0.42 if cfg.scenario == "abundant" else 0.90
    return base, speed, inertia

def _compute_regrouping(
    sheep, flock, cfg, ctx, rng,
    prev_dir, grad_dir, mem_dir,
) -> tuple[np.ndarray, float, float]:
    to_centroid = safe_unit(ctx.flock_centroid - sheep.position)
    base_attr   = 0.44 * sheep.sociability if cfg.scenario == "abundant" else 0.55 * sheep.sociability
    base_aln    = 0.14 * sheep.sociability if cfg.scenario == "abundant" else 0.20 * sheep.sociability
    base_rep    = 0.94
    attr, aln, rep = get_effective_weights(sheep, base_attr, base_aln, base_rep)

    social = _social_force(
        sheep, flock, cfg.flock,
        attr_scale=attr, aln_scale=aln, rep_scale=rep,
        spc_scale=1.04 * sheep.preferred_spacing,
    )
    raw = (
        0.44 * to_centroid
        + 0.22 * social
        + 0.10 * prev_dir
        + 0.10 * grad_dir
        + 0.08 * mem_dir
    )
    base = _jitter(
        safe_unit(raw if np.linalg.norm(raw) > 1e-9 else to_centroid),
        0.38 * cfg.flock.stochastic_turn_std * sheep.turning_bias,
        rng,
    )
    speed   = cfg.flock.regroup_speed * sheep.movement_vigor * (0.64 + 0.10 * ctx.active_factor)
    inertia = 0.22 if cfg.scenario == "abundant" else 0.52
    return base, speed, inertia

def _compute_resting(
    sheep, flock, cfg, ctx, rng, prev_dir, grad_dir,
) -> tuple[np.ndarray, float, float]:
    micro = _jitter(prev_dir, 0.90 * cfg.flock.stochastic_turn_std, rng)
    base  = safe_unit(
        0.55 * micro
        + 0.25 * grad_dir
        + 0.20 * _social_force(sheep, flock, cfg.flock)
    )
    speed   = cfg.flock.resting_speed * (0.24 + 0.18 * rng.random())
    inertia = 0.02
    return base, speed, inertia

def _social_force(
    sheep: SheepAgent,
    flock: list[SheepAgent],
    cfg: FlockConfig,
    *,
    attr_scale: float = 1.0,
    aln_scale:  float = 1.0,
    rep_scale:  float = 1.0,
    spc_scale:  float = 1.0,
) -> np.ndarray:
    repulsion  = np.zeros(2, dtype=float)
    attraction = np.zeros(2, dtype=float)
    alignment  = np.zeros(2, dtype=float)
    count = 0
    eff_rep_r = cfg.repulsion_radius * spc_scale

    for other in flock:
        if other.sheep_id == sheep.sheep_id:
            continue
        offset = other.position - sheep.position
        dist   = np.linalg.norm(offset)
        if dist < 1e-9 or dist > cfg.neighbour_radius:
            continue
        count += 1
        if dist < eff_rep_r:
            repulsion -= safe_unit(offset) * (eff_rep_r - dist) / max(eff_rep_r, 1e-9)
        attraction += safe_unit(offset)
        if np.linalg.norm(other.velocity) > 1e-9:
            alignment += safe_unit(other.velocity)

    if count == 0:
        return np.zeros(2, dtype=float)

    attraction /= count
    alignment  /= count
    return (
        cfg.repulsion_weight  * rep_scale  * repulsion
        + cfg.attraction_weight * attr_scale * attraction
        + cfg.alignment_weight  * aln_scale  * alignment
    )

def _jitter(base_dir: np.ndarray, angle_std: float, rng: np.random.Generator) -> np.ndarray:
    if np.linalg.norm(base_dir) < 1e-9:
        base_dir = np.array([1.0, 0.0], dtype=float)
    return safe_unit(rotate(base_dir, float(rng.normal(0.0, angle_std))))

def _prospect_direction(
    sheep: SheepAgent,
    food: FoodLandscape,
    cfg: SimulationConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    radii  = (10.0, 18.0, 28.0)
    angles = np.linspace(-np.pi, np.pi, 8, endpoint=False)
    best_score = -np.inf
    best_dir   = np.zeros(2, dtype=float)

    for r in radii:
        for a in angles:
            d = np.array([np.cos(a), np.sin(a)], dtype=float)
            probe = sheep.position + d * r
            probe[0] = clamp(float(probe[0]), 0.0, cfg.field.width)
            probe[1] = clamp(float(probe[1]), 0.0, cfg.field.height)
            q = food.local_quality(
                float(probe[0]), float(probe[1]), cfg.food.sensory_radius, cfg.field
            )
            score = q - 0.010 * r
            if score > best_score:
                best_score = score
                best_dir   = d

    return safe_unit(best_dir)

def _patch_residence_weight(ctx: BehaviourContext) -> float:
    return clamp(
        0.55 * ctx.local_quality + 0.45 * ctx.cell_quality + 0.20 * ctx.local_health,
        0.0, 1.0,
    )

def _grazing_exit_pressure(ctx: BehaviourContext) -> float:
    poor_quality = max(0.0, 0.22 - ctx.cell_quality)
    poor_intake  = max(0.0, 0.0035 - ctx.recent_intake_rate)
    return clamp(1.8 * poor_quality + 80.0 * poor_intake, 0.0, 1.0)

def heading_to_velocity(heading: float, speed: float) -> np.ndarray:
    return np.array([np.cos(heading) * speed, np.sin(heading) * speed], dtype=float)
