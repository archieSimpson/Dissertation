from typing import Tuple
import pygame
import numpy as np

from flock import centroid
from math2d import norm, safe_normalize


BACKGROUND = (245, 245, 240)
FENCE_COLOUR = (70, 70, 70)
GOAL_COLOUR = (210, 240, 210)
TEXT_COLOUR = (20, 20, 20)
DOG_COLOUR = (200, 50, 50)
SHEEP_COLOUR = (240, 240, 240)
SHEEP_EDGE = (80, 80, 80)
CENTROID_COLOUR = (40, 100, 200)


def draw_arrow(surface: pygame.Surface, start: np.ndarray, direction: np.ndarray, length: float, colour: Tuple[int, int, int]) -> None:
    d = safe_normalize(direction)
    if norm(d) < 1e-8:
        return

    end = start + d * length
    pygame.draw.line(surface, colour, start.astype(int), end.astype(int), 2)

    left = end - d * 10 + np.array([-d[1], d[0]]) * 5
    right = end - d * 10 + np.array([d[1], -d[0]]) * 5
    pygame.draw.polygon(surface, colour, [end.astype(int), left.astype(int), right.astype(int)])


def render(screen: pygame.Surface, font: pygame.font.Font, sim) -> None:
    cfg = sim.cfg
    env = sim.env
    dog = sim.dog
    sheep_list = sim.sheep

    screen.fill(BACKGROUND)

    pygame.draw.rect(
        screen,
        FENCE_COLOUR,
        pygame.Rect(cfg.world_margin, cfg.world_margin, cfg.width - 2 * cfg.world_margin, cfg.height - 2 * cfg.world_margin),
        width=3,
    )

    goal_rect = pygame.Rect(env.goal.x, env.goal.y, env.goal.w, env.goal.h)
    pygame.draw.rect(screen, GOAL_COLOUR, goal_rect)
    pygame.draw.rect(screen, (70, 130, 70), goal_rect, width=3)

    for sheep in sheep_list:
        fill = (220, 225, 255) if sheep.alert else SHEEP_COLOUR
        pygame.draw.circle(screen, fill, sheep.pos.astype(int), cfg.sheep_radius)
        pygame.draw.circle(screen, SHEEP_EDGE, sheep.pos.astype(int), cfg.sheep_radius, width=1)
        draw_arrow(screen, sheep.pos, sheep.vel, 12, (120, 120, 120))

    pygame.draw.circle(screen, DOG_COLOUR, dog.pos.astype(int), cfg.dog_radius)
    pygame.draw.circle(screen, (80, 20, 20), dog.pos.astype(int), cfg.dog_radius, width=2)
    draw_arrow(screen, dog.pos, dog.vel, 18, (120, 20, 20))
    pygame.draw.circle(screen, (255, 170, 170), dog.pos.astype(int), int(cfg.dog_influence_radius), width=1)

    c = centroid(sheep_list)
    pygame.draw.circle(screen, CENTROID_COLOUR, c.astype(int), 7)

    lines = [
        f"Time: {sim.time_elapsed:6.1f}s",
        f"Mode: {dog.mode} / {dog.submode}",
        f"Spread: {sim.current_metrics['spread']:6.1f}",
        f"Polarization: {sim.current_metrics['polarization']:.3f}",
        f"Clusters: {sim.current_metrics['num_clusters']}",
        f"Hull area: {sim.current_metrics['convex_hull_area']:.1f}",
        f"Elongation: {sim.current_metrics['elongation_ratio']:.2f}",
        f"In goal: {sim.current_metrics['in_goal']}/{len(sheep_list)}",
        "Controls: WASD / Arrows",
        "Shift = sprint",
        "Tab = toggle manual / auto dog",
    ]

    if sim.success:
        lines.append("SUCCESS: whole flock reached the goal")

    y = 16
    for line in lines:
        text = font.render(line, True, TEXT_COLOUR)
        screen.blit(text, (16, y))
        y += 28