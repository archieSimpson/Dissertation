from __future__ import annotations

import math
from typing import Iterable

import numpy as np


EPS = 1e-9


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def safe_unit(vec: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(vec)
    if n < EPS:
        return np.zeros_like(vec)
    return vec / n


def rotate(vec: np.ndarray, angle: float) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]], dtype=float)


def angle_of(vec: np.ndarray) -> float:
    return math.atan2(vec[1], vec[0])


def wrap_angle(a: float) -> float:
    while a <= -math.pi:
        a += 2 * math.pi
    while a > math.pi:
        a -= 2 * math.pi
    return a


def heading_vector(angle: float) -> np.ndarray:
    return np.array([math.cos(angle), math.sin(angle)], dtype=float)


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    diffs = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diffs, axis=2)


def convex_hull_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    pts = np.unique(points, axis=0)
    if len(pts) < 3:
        return 0.0
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
        return float((a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]))

    lower: list[np.ndarray] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[np.ndarray] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    x = hull[:, 0]
    y = hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def mean_nearest_neighbour_distance(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    d = pairwise_distances(points)
    np.fill_diagonal(d, np.inf)
    return float(np.mean(np.min(d, axis=1)))


def polarization(velocities: np.ndarray) -> float:
    if len(velocities) == 0:
        return 0.0
    dirs = np.array([safe_unit(v) for v in velocities])
    return float(np.linalg.norm(np.mean(dirs, axis=0)))


def connected_components_from_distance(points: np.ndarray, threshold: float) -> int:
    n = len(points)
    if n == 0:
        return 0
    d = pairwise_distances(points)
    adjacency = d <= threshold
    seen = np.zeros(n, dtype=bool)
    components = 0
    for i in range(n):
        if seen[i]:
            continue
        components += 1
        stack = [i]
        seen[i] = True
        while stack:
            node = stack.pop()
            neighbours = np.where(adjacency[node])[0]
            for nb in neighbours:
                if not seen[nb]:
                    seen[nb] = True
                    stack.append(int(nb))
    return components


def rescale01(arr: np.ndarray) -> np.ndarray:
    lo = float(np.min(arr))
    hi = float(np.max(arr))
    if hi - lo < EPS:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def gaussian_2d(
    xs: np.ndarray,
    ys: np.ndarray,
    cx: float,
    cy: float,
    sigma_x: float,
    sigma_y: float,
    amplitude: float,
) -> np.ndarray:
    return amplitude * np.exp(
        -(((xs - cx) ** 2) / (2 * sigma_x**2) + ((ys - cy) ** 2) / (2 * sigma_y**2))
    )


def circadian_factor(step: int, period: int, phase_shift: float, amplitude: float) -> float:
    theta = 2 * math.pi * ((step / period) + phase_shift)
    return clamp(0.5 + amplitude * math.sin(theta), 0.0, 1.0)


def azimuth_deg(vec: np.ndarray) -> float:
    return (math.degrees(math.atan2(vec[1], vec[0])) + 360.0) % 360.0
