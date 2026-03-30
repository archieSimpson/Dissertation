import numpy as np

from config import SimulationConfig
from math2d import limit, norm, safe_normalize


def clamp_to_world(pos: np.ndarray, cfg: SimulationConfig) -> np.ndarray:
    pos[0] = min(max(cfg.world_margin, pos[0]), cfg.width - cfg.world_margin)
    pos[1] = min(max(cfg.world_margin, pos[1]), cfg.height - cfg.world_margin)
    return pos


def integrate_sheep(pos: np.ndarray, vel: np.ndarray, force: np.ndarray, alert: bool, cfg: SimulationConfig, max_speed_multiplier: float):
    new_vel = (cfg.w_inertia * vel) + (force * cfg.dt)
    max_speed = (cfg.max_speed_alert if alert else cfg.max_speed_calm) * max_speed_multiplier
    new_vel = limit(new_vel, max_speed)
    new_pos = pos + new_vel * cfg.dt
    new_pos = clamp_to_world(new_pos, cfg)
    return new_pos, new_vel


def integrate_dog(pos: np.ndarray, vel: np.ndarray, accel: np.ndarray, cfg: SimulationConfig):
    new_vel = vel + accel * cfg.dt
    new_vel *= cfg.dog_friction
    new_vel = limit(new_vel, cfg.dog_max_speed * cfg.dog_sprint_multiplier)
    new_pos = pos + new_vel * cfg.dt
    new_pos = clamp_to_world(new_pos, cfg)
    return new_pos, new_vel


def resolve_sheep_collisions(sheep_list, cfg: SimulationConfig):
    min_dist = cfg.sheep_radius * 2.2

    for _ in range(2):
        for i in range(len(sheep_list)):
            for j in range(i + 1, len(sheep_list)):
                a = sheep_list[i]
                b = sheep_list[j]

                delta = b.pos - a.pos
                d = norm(delta)

                if d < 1e-8:
                    direction = np.random.normal(0, 1, size=2)
                    direction = safe_normalize(direction)
                    push = direction * (min_dist * 0.5)
                    a.pos -= push
                    b.pos += push
                    a.pos = clamp_to_world(a.pos, cfg)
                    b.pos = clamp_to_world(b.pos, cfg)
                    continue

                if d < min_dist:
                    overlap = min_dist - d
                    direction = delta / d
                    push = direction * (overlap * 0.5)

                    a.pos -= push
                    b.pos += push
                    a.vel -= direction * 8.0
                    b.vel += direction * 8.0

                    a.pos = clamp_to_world(a.pos, cfg)
                    b.pos = clamp_to_world(b.pos, cfg)


def resolve_dog_sheep_collisions(dog, sheep_list, cfg: SimulationConfig):
    min_dist = cfg.dog_radius + cfg.sheep_radius + 2.0

    for sheep in sheep_list:
        delta = sheep.pos - dog.pos
        d = norm(delta)

        if d < 1e-8:
            direction = np.random.normal(0, 1, size=2)
            direction = safe_normalize(direction)
            sheep.pos += direction * (min_dist * 0.5)
            sheep.pos = clamp_to_world(sheep.pos, cfg)
            sheep.vel += direction * cfg.dog_sheep_push_strength
            continue

        if d < min_dist:
            overlap = min_dist - d
            direction = delta / d

            sheep.pos += direction * overlap
            sheep.pos = clamp_to_world(sheep.pos, cfg)
            sheep.vel += direction * cfg.dog_sheep_push_strength