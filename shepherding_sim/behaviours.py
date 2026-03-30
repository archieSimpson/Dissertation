import math
import numpy as np

from agents import Sheep, Dog
from config import SimulationConfig
from flock import neighbours_of
from math2d import distance, limit, norm, safe_normalize, vec


def wall_force(pos: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    f = vec(0, 0)

    if pos[0] < cfg.world_margin + cfg.boundary_repulsion_distance:
        strength = (cfg.world_margin + cfg.boundary_repulsion_distance - pos[0]) / cfg.boundary_repulsion_distance
        f += vec(1, 0) * strength

    if pos[0] > cfg.width - cfg.world_margin - cfg.boundary_repulsion_distance:
        strength = (pos[0] - (cfg.width - cfg.world_margin - cfg.boundary_repulsion_distance)) / cfg.boundary_repulsion_distance
        f += vec(-1, 0) * strength

    if pos[1] < cfg.world_margin + cfg.boundary_repulsion_distance:
        strength = (cfg.world_margin + cfg.boundary_repulsion_distance - pos[1]) / cfg.boundary_repulsion_distance
        f += vec(0, 1) * strength

    if pos[1] > cfg.height - cfg.world_margin - cfg.boundary_repulsion_distance:
        strength = (pos[1] - (cfg.height - cfg.world_margin - cfg.boundary_repulsion_distance)) / cfg.boundary_repulsion_distance
        f += vec(0, -1) * strength

    return f


def dog_force(sheep: Sheep, dog: Dog, flock_centroid: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    offset = sheep.pos - dog.pos
    d = norm(offset)

    if d > cfg.dog_influence_radius:
        return vec(0, 0)

    away = safe_normalize(offset)

    short_range = 0.0
    if d < cfg.dog_hard_repulsion_radius:
        short_range = 1.0 + (cfg.dog_hard_repulsion_radius - d) / max(cfg.dog_hard_repulsion_radius, 1.0)

    long_range = max(0.0, 1.0 - d / cfg.dog_influence_radius)
    force = away * (0.8 * long_range + 2.0 * short_range)

    if cfg.use_directional_dog_pressure:
        to_centroid = safe_normalize(flock_centroid - sheep.pos)
        force += 0.22 * to_centroid

    return force * sheep.fear_strength


def update_wander_direction(
    sheep: Sheep,
    cfg: SimulationConfig,
    rng: np.random.Generator,
    py_rng,
) -> np.ndarray:
    sheep.wander_timer -= cfg.dt

    if sheep.wander_timer <= 0.0 or norm(sheep.wander_dir) < 1e-8:
        angle = py_rng.uniform(0.0, 2.0 * math.pi)
        sheep.wander_dir = vec(math.cos(angle), math.sin(angle))
        sheep.wander_timer = py_rng.uniform(
            cfg.wander_change_interval_min,
            cfg.wander_change_interval_max,
        )

    jitter = rng.normal(0.0, cfg.graze_jitter, size=2)
    sheep.wander_dir = safe_normalize(sheep.wander_dir + jitter)

    return sheep.wander_dir


def speed_regulation_force(
    sheep: Sheep,
    desired_dir: np.ndarray,
    alert: bool,
    cfg: SimulationConfig,
) -> np.ndarray:
    desired_dir = safe_normalize(desired_dir)
    if norm(desired_dir) < 1e-8:
        desired_dir = safe_normalize(sheep.vel)

    target_speed = cfg.target_speed_alert if alert else cfg.target_speed_calm
    target_speed *= sheep.max_speed_multiplier

    desired_velocity = desired_dir * target_speed
    return (desired_velocity - sheep.vel) * cfg.speed_relax_gain


def sheep_force(
    sheep: Sheep,
    sheep_list: list[Sheep],
    dog: Dog,
    flock_centroid: np.ndarray,
    cfg: SimulationConfig,
    rng: np.random.Generator,
    py_rng,
) -> np.ndarray:
    neighbours = neighbours_of(
        sheep=sheep,
        sheep_list=sheep_list,
        neighbour_radius=cfg.neighbour_radius,
        use_local_neighbours_only=cfg.use_local_neighbours_only,
        k_neighbours=cfg.k_neighbours,
        vision_angle_deg=cfg.vision_angle_deg,
    )

    sep = vec(0, 0)
    coh = vec(0, 0)
    ali = vec(0, 0)

    count = 0
    for other in neighbours:
        diff = sheep.pos - other.pos
        d = norm(diff)
        if d < 1e-6:
            continue

        if d < cfg.separation_radius:
            sep += safe_normalize(diff) * ((cfg.separation_radius - d) / cfg.separation_radius)

        coh += other.pos
        ali += other.vel
        count += 1

    if count > 0:
        coh = safe_normalize((coh / count) - sheep.pos)
        ali = safe_normalize((ali / count) - sheep.vel)
    else:
        coh = vec(0, 0)
        ali = vec(0, 0)

    d_to_dog = distance(sheep.pos, dog.pos)
    alert = d_to_dog < cfg.dog_influence_radius
    sheep.alert = alert
    sheep.state = "alert" if alert else "calm"

    dog_f = dog_force(sheep, dog, flock_centroid, cfg)
    wall_f = wall_force(sheep.pos, cfg)
    wander_dir = update_wander_direction(sheep, cfg, rng, py_rng)

    if alert:
        graze_f = 0.20 * wander_dir * sheep.graze_bias
        sep_w = cfg.w_sep
        coh_w = cfg.w_coh * 1.35
        ali_w = cfg.w_ali * 1.45
        dog_w = cfg.w_dog
        wall_w = cfg.w_wall

        desired_dir = safe_normalize(
            0.50 * wander_dir +
            0.90 * ali +
            0.35 * coh +
            1.20 * dog_f
        )
    else:
        graze_f = 1.35 * wander_dir * sheep.graze_bias
        sep_w = cfg.w_sep * 0.85
        coh_w = cfg.w_coh * 0.65
        ali_w = cfg.w_ali * 0.50
        dog_w = cfg.w_dog
        wall_w = cfg.w_wall

        desired_dir = safe_normalize(
            1.55 * wander_dir +
            0.35 * ali +
            0.20 * coh
        )

    propulsion_f = speed_regulation_force(sheep, desired_dir, alert, cfg)
    noise = rng.normal(0.0, 1.0, size=2)

    raw_force = (
        sep_w * sep +
        coh_w * coh +
        ali_w * ali +
        cfg.w_graze * graze_f +
        propulsion_f +
        dog_w * dog_f +
        wall_w * wall_f +
        cfg.w_noise * noise
    )

    # Reaction delay / force memory
    sheep.delay_counter += 1
    if sheep.delay_counter >= sheep.reaction_delay_frames:
        smoothed = (cfg.force_memory * sheep.previous_force) + ((1.0 - cfg.force_memory) * raw_force)
        sheep.previous_force = smoothed
        sheep.delay_counter = 0

    return limit(sheep.previous_force, cfg.max_force)