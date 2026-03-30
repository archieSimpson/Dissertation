import pygame

from config import SimulationConfig
from flock import centroid, furthest_sheep_from_centroid, spread
from math2d import safe_normalize, vec


def manual_dog_acceleration(keys: pygame.key.ScancodeWrapper, cfg: SimulationConfig):
    accel = vec(0, 0)

    if keys[pygame.K_w] or keys[pygame.K_UP]:
        accel[1] -= 1
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        accel[1] += 1
    if keys[pygame.K_a] or keys[pygame.K_LEFT]:
        accel[0] -= 1
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]:
        accel[0] += 1

    accel = safe_normalize(accel)

    boost = 1.0
    if keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]:
        boost = cfg.dog_sprint_multiplier

    return accel * cfg.dog_accel * boost


def autonomous_dog_acceleration(dog, sheep_list, goal, cfg: SimulationConfig):
    flock_c = centroid(sheep_list)
    furthest = furthest_sheep_from_centroid(sheep_list)

    if furthest is None:
        return vec(0, 0), "drive"

    flock_spread = spread(sheep_list)
    threshold = cfg.collect_spread_threshold_factor * cfg.separation_radius

    goal_pos = vec(goal.x + goal.w / 2.0, goal.y + goal.h / 2.0)
    goal_dir = safe_normalize(goal_pos - flock_c)

    if flock_spread > threshold:
        # Collect mode: move behind furthest sheep relative to flock centroid
        dog.submode = "collect"
        collect_dir = safe_normalize(furthest.pos - flock_c)
        target = furthest.pos + collect_dir * (cfg.collect_distance_factor * cfg.separation_radius)
    else:
        # Drive mode: move behind flock centroid relative to goal
        dog.submode = "drive"
        target = flock_c - goal_dir * (cfg.drive_distance_factor * cfg.separation_radius * 3.0)

    desired = safe_normalize(target - dog.pos)
    return desired * cfg.dog_accel, dog.submode