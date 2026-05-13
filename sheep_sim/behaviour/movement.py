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
    boundary  = env.boundary_force(sheep.position, cfg.field.boundary_margin) * cfg.field.boundary_turn_strength
    if cfg.features.terrain:
        terrain_f = -env.terrain_gradient(*sheep.position) * cfg.field.terrain_strength
    else:
        terrain_f = np.zeros(2, dtype=float)
    env_force = boundary + terrain_f
 
    prev_dir     = (
        safe_unit(sheep.velocity)
        if np.linalg.norm(sheep.velocity) > 1e-9
        else sheep.heading_vector()
    )
    if cfg.features.memory:
        mem_dir  = safe_unit(ctx.memory_target - sheep.position)
    else:
        mem_dir  = np.zeros(2, dtype=float)
    if cfg.features.foraging:
        grad_dir     = safe_unit(ctx.sensory_gradient)
        prospect_dir = _prospect_direction(sheep, food, cfg, rng)
        residence    = _patch_residence_weight(ctx)
        exit_p       = _grazing_exit_pressure(ctx)
    else:
        grad_dir     = np.zeros(2, dtype=float)
        prospect_dir = np.zeros(2, dtype=float)
        residence    = 0.0
        exit_p       = 0.0
 
    if sheep.state == BehaviourState.GRAZING:
        base_dir, speed, inertia = _compute_grazing(
            sheep, flock, cfg, ctx, rng,
            prev_dir, grad_dir, mem_dir, prospect_dir, residence, exit_p,
        )
    elif sheep.state == BehaviourState.WALKING:
        base_dir, speed, inertia = _compute_walking(
            sheep, flock, cfg, ctx, rng,
            prev_dir, grad_dir, mem_dir, prospect_dir, exit_p,
        )
    elif sheep.state == BehaviourState.TRAVELLING:
        base_dir, speed, inertia = _compute_travelling(
            sheep, flock, cfg, ctx, rng,
            prev_dir, grad_dir, mem_dir, prospect_dir,
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
    if cfg.scenario in {"abundant", "uniform_high", "radial_increase", "ring", "corridors"} and sheep.state in {BehaviourState.GRAZING, BehaviourState.RESTING}:
        min_sp = 0.3 if sheep.state == BehaviourState.GRAZING else 0.05
        final_speed = max(final_speed, min_sp)
 
    final_speed = clamp(final_speed, 0.0, cfg.flock.max_speed)
    return safe_unit(blended) * final_speed
 
 
def _compute_grazing(
    sheep, flock, cfg, ctx, rng,
    prev_dir, grad_dir, mem_dir, prospect_dir, residence, exit_p,
) -> tuple[np.ndarray, float, float]:
    base_attr = 0.28 * sheep.sociability if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 0.34 * sheep.sociability
    base_aln  = 0.10 * sheep.sociability if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 0.18 * sheep.sociability
    base_rep  = 1.08 if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 1.30
    attr, aln, rep = get_effective_weights(sheep, base_attr, base_aln, base_rep)
 
    social = _social_force(
        sheep, flock, cfg.flock,
        attr_scale=attr, aln_scale=aln, rep_scale=rep,
        spc_scale=1.06 * sheep.preferred_spacing if cfg.scenario in {"abundant","uniform_high"} else sheep.preferred_spacing,
        enabled=cfg.features.social,
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
    )
    base = _jitter(
        safe_unit(raw if np.linalg.norm(raw) > 1e-9 else exploratory),
        1.35 * cfg.flock.stochastic_turn_std * sheep.turning_bias,
        rng,
    )
    speed   = cfg.flock.grazing_speed * sheep.movement_vigor * (0.18 + 0.22 * ctx.active_factor + 0.14 * exit_p)
    inertia = 0.08
    return base, speed, inertia
 
 
def _compute_walking(
    sheep, flock, cfg, ctx, rng,
    prev_dir, grad_dir, mem_dir, prospect_dir, exit_p,
) -> tuple[np.ndarray, float, float]:
    base_attr = 0.20 * sheep.sociability if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 0.12 * sheep.sociability
    base_aln  = 0.08 * sheep.sociability
    base_rep  = 0.96
    attr, aln, rep = get_effective_weights(sheep, base_attr, base_aln, base_rep)
 
    social = _social_force(
        sheep, flock, cfg.flock,
        attr_scale=attr, aln_scale=aln, rep_scale=rep,
        spc_scale=sheep.preferred_spacing,
        enabled=cfg.features.social,
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
    )
    base = _jitter(
        safe_unit(raw if np.linalg.norm(raw) > 1e-9 else exploratory),
        0.55 * cfg.flock.stochastic_turn_std * sheep.turning_bias,
        rng,
    )
    speed   = cfg.flock.walking_speed * sheep.movement_vigor * (0.38 + 0.12 * ctx.active_factor + 0.10 * exit_p)
    inertia = 0.25 if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 0.40
    return base, speed, inertia
 
 
def _compute_travelling(
    sheep, flock, cfg, ctx, rng,
    prev_dir, grad_dir, mem_dir, prospect_dir,
) -> tuple[np.ndarray, float, float]:
    base_attr = 0.18 * sheep.sociability if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 0.06 * sheep.sociability
    base_aln  = 0.12 * sheep.sociability if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 0.06 * sheep.sociability
    base_rep  = 0.82 if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 0.90
    attr, aln, rep = get_effective_weights(sheep, base_attr, base_aln, base_rep)
 
    social = _social_force(
        sheep, flock, cfg.flock,
        attr_scale=attr, aln_scale=aln, rep_scale=rep,
        spc_scale=0.96 * sheep.preferred_spacing,
        enabled=cfg.features.social,
    )
    raw = (
        0.35 * prev_dir
        + 0.22 * mem_dir
        + 0.20 * prospect_dir
        + 0.06 * grad_dir
        + 0.12 * social
        + 0.03 * safe_unit(ctx.flock_centroid - sheep.position)
    )
    base = _jitter(
        safe_unit(raw if np.linalg.norm(raw) > 1e-9 else prev_dir),
        0.22 * cfg.flock.stochastic_turn_std * sheep.turning_bias,
        rng,
    )
    speed   = cfg.flock.travel_speed * sheep.movement_vigor * (0.54 + 0.12 * ctx.active_factor)
    inertia = 0.50 if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 0.52
    return base, speed, inertia
 
 
def _compute_regrouping(
    sheep, flock, cfg, ctx, rng,
    prev_dir, grad_dir, mem_dir,
) -> tuple[np.ndarray, float, float]:
    to_centroid = safe_unit(ctx.flock_centroid - sheep.position)
    base_attr   = 0.44 * sheep.sociability if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 0.55 * sheep.sociability
    base_aln    = 0.14 * sheep.sociability if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 0.20 * sheep.sociability
    base_rep    = 0.94
    attr, aln, rep = get_effective_weights(sheep, base_attr, base_aln, base_rep)
 
    social = _social_force(
        sheep, flock, cfg.flock,
        attr_scale=attr, aln_scale=aln, rep_scale=rep,
        spc_scale=1.04 * sheep.preferred_spacing,
        enabled=cfg.features.social,
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
    inertia = 0.30 if cfg.scenario in {"abundant","uniform_high","radial_increase","ring","corridors"} else 0.45
    return base, speed, inertia
 
 
def _compute_resting(
    sheep, flock, cfg, ctx, rng, prev_dir, grad_dir,
) -> tuple[np.ndarray, float, float]:
    micro = _jitter(prev_dir, 0.90 * cfg.flock.stochastic_turn_std, rng)
    base  = safe_unit(
        0.55 * micro
        + 0.25 * grad_dir
        + 0.20 * _social_force(sheep, flock, cfg.flock, enabled=cfg.features.social)
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
    enabled: bool = True,
) -> np.ndarray:
    if not enabled:
        return np.zeros(2, dtype=float)
    repulsion  = np.zeros(2, dtype=float)
    attraction = np.zeros(2, dtype=float)
    alignment  = np.zeros(2, dtype=float)
    n_attract = n_align = 0
    eff_rep_r = cfg.repulsion_radius * spc_scale
    eff_aln_r = max(cfg.alignment_radius * spc_scale, eff_rep_r + 1e-6)
    eff_att_r = max(cfg.neighbour_radius * spc_scale, eff_aln_r + 1e-6)
 
    for other in flock:
        if other.sheep_id == sheep.sheep_id:
            continue
        offset = other.position - sheep.position
        dist   = np.linalg.norm(offset)
        if dist < 1e-9 or dist > eff_att_r:
            continue
        if dist < eff_rep_r:
            repulsion -= safe_unit(offset) * (eff_rep_r - dist) / max(eff_rep_r, 1e-9)
        elif dist < eff_aln_r:
            if np.linalg.norm(other.velocity) > 1e-9:
                alignment += safe_unit(other.velocity)
                n_align += 1
        else:
            attraction += safe_unit(offset)
            n_attract += 1
 
    if n_attract > 0:
        attraction /= n_attract
    if n_align > 0:
        alignment /= n_align
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
    radii  = (15.0, 35.0, 60.0)
    angles = np.linspace(-np.pi, np.pi, 12, endpoint=False)
    best_score = -np.inf
    best_dir   = np.zeros(2, dtype=float)
 
    for r in radii:
        for a in angles:
            d = np.array([np.cos(a), np.sin(a)], dtype=float)
            probe = sheep.position + d * r
            probe[0] = clamp(float(probe[0]), 0.0, cfg.field.width)
            probe[1] = clamp(float(probe[1]), 0.0, cfg.field.height)
            q = food.local_quality(float(probe[0]), float(probe[1]), cfg.food.sensory_radius, cfg.field)
            score = q - 0.001 * r
            if score > best_score:
                best_score = score
                best_dir   = d
 
    return safe_unit(best_dir)
 
 
def _patch_residence_weight(ctx: BehaviourContext) -> float:
    return clamp(0.55 * ctx.local_quality + 0.45 * ctx.cell_quality + 0.20 * ctx.local_health, 0.0, 1.0)
 
 
def _grazing_exit_pressure(ctx: BehaviourContext) -> float:
    poor_quality = max(0.0, 0.22 - ctx.cell_quality)
    poor_intake  = max(0.0, 0.0035 - ctx.recent_intake_rate)
    return clamp(1.8 * poor_quality + 80.0 * poor_intake, 0.0, 1.0)
 
 
def heading_to_velocity(heading: float, speed: float) -> np.ndarray:
    return np.array([np.cos(heading) * speed, np.sin(heading) * speed], dtype=float)