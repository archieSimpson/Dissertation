from collections import deque
from typing import List, Tuple
import numpy as np

from agents import Sheep
from math2d import distance, dot, norm, safe_normalize


def centroid(sheep_list: List[Sheep]) -> np.ndarray:
    if not sheep_list:
        return np.zeros(2, dtype=np.float64)
    return np.mean(np.array([s.pos for s in sheep_list]), axis=0)


def mean_velocity(sheep_list: List[Sheep]) -> np.ndarray:
    if not sheep_list:
        return np.zeros(2, dtype=np.float64)
    return np.mean(np.array([s.vel for s in sheep_list]), axis=0)


def spread(sheep_list: List[Sheep]) -> float:
    if not sheep_list:
        return 0.0
    c = centroid(sheep_list)
    return max(distance(s.pos, c) for s in sheep_list)


def polarization(sheep_list: List[Sheep]) -> float:
    if not sheep_list:
        return 0.0
    headings = np.array([safe_normalize(s.vel) for s in sheep_list])
    return norm(np.mean(headings, axis=0))


def mean_nearest_neighbour_distance(sheep_list: List[Sheep]) -> float:
    if len(sheep_list) < 2:
        return 0.0

    nn_distances = []
    for i, s in enumerate(sheep_list):
        best = float("inf")
        for j, other in enumerate(sheep_list):
            if i == j:
                continue
            d = distance(s.pos, other.pos)
            if d < best:
                best = d
        nn_distances.append(best)

    return float(np.mean(nn_distances))


def _in_vision_cone(sheep: Sheep, other: Sheep, cos_half_angle: float) -> bool:
    to_other = safe_normalize(other.pos - sheep.pos)
    heading = safe_normalize(sheep.vel)

    # If nearly stationary, allow omni-directional awareness
    if norm(heading) < 1e-8:
        return True

    return dot(heading, to_other) > cos_half_angle


def neighbours_of(
    sheep: Sheep,
    sheep_list: List[Sheep],
    neighbour_radius: float,
    use_local_neighbours_only: bool,
    k_neighbours: int,
    vision_angle_deg: float,
) -> List[Sheep]:
    half_angle_rad = np.deg2rad(vision_angle_deg / 2.0)
    cos_half_angle = np.cos(half_angle_rad)

    others: List[Tuple[float, Sheep]] = []
    for s in sheep_list:
        if s.idx == sheep.idx:
            continue
        d = distance(sheep.pos, s.pos)
        if d <= neighbour_radius and _in_vision_cone(sheep, s, cos_half_angle):
            others.append((d, s))

    others.sort(key=lambda x: x[0])

    if use_local_neighbours_only:
        return [s for _, s in others[:k_neighbours]]
    return [s for _, s in others]


def furthest_sheep_from_centroid(sheep_list: List[Sheep]) -> Sheep | None:
    if not sheep_list:
        return None
    c = centroid(sheep_list)
    return max(sheep_list, key=lambda s: distance(s.pos, c))


def count_clusters(sheep_list: List[Sheep], threshold: float) -> int:
    if not sheep_list:
        return 0

    n = len(sheep_list)
    visited = [False] * n

    def close(i: int, j: int) -> bool:
        return distance(sheep_list[i].pos, sheep_list[j].pos) <= threshold

    clusters = 0
    for i in range(n):
        if visited[i]:
            continue
        clusters += 1
        q = deque([i])
        visited[i] = True

        while q:
            u = q.popleft()
            for v in range(n):
                if not visited[v] and close(u, v):
                    visited[v] = True
                    q.append(v)

    return clusters