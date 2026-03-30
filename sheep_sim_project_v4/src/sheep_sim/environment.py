from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sheep_sim.config import FieldConfig
from sheep_sim.utils import clamp, gaussian_2d, rescale01, safe_unit


@dataclass(slots=True)
class FieldEnvironment:
    terrain: np.ndarray
    shade: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    width: float
    height: float
    cols: int
    rows: int

    def in_bounds(self, pos: np.ndarray) -> bool:
        return 0.0 <= pos[0] <= self.width and 0.0 <= pos[1] <= self.height

    def clamp_position(self, pos: np.ndarray) -> np.ndarray:
        return np.array([clamp(pos[0], 0.0, self.width), clamp(pos[1], 0.0, self.height)], dtype=float)

    def terrain_gradient(self, x: float, y: float) -> np.ndarray:
        return self._finite_diff(self.terrain, x, y)

    def shade_gradient(self, x: float, y: float) -> np.ndarray:
        return self._finite_diff(self.shade, x, y)

    def boundary_force(self, pos: np.ndarray, margin: float = 6.0) -> np.ndarray:
        x, y = pos
        fx = 0.0
        fy = 0.0
        if x < margin:
            fx += (margin - x) / margin
        elif x > self.width - margin:
            fx -= (x - (self.width - margin)) / margin
        if y < margin:
            fy += (margin - y) / margin
        elif y > self.height - margin:
            fy -= (y - (self.height - margin)) / margin
        return np.array([fx, fy], dtype=float)

    def sample_terrain(self, x: float, y: float) -> float:
        r, c = self._index(x, y)
        return float(self.terrain[r, c])

    def sample_shade(self, x: float, y: float) -> float:
        r, c = self._index(x, y)
        return float(self.shade[r, c])

    def _index(self, x: float, y: float) -> tuple[int, int]:
        col = min(self.cols - 1, max(0, int((x / self.width) * self.cols)))
        row = min(self.rows - 1, max(0, int((y / self.height) * self.rows)))
        return row, col

    def _finite_diff(self, arr: np.ndarray, x: float, y: float) -> np.ndarray:
        eps_x = self.width / self.cols
        eps_y = self.height / self.rows
        left = self._sample(arr, clamp(x - eps_x, 0.0, self.width), y)
        right = self._sample(arr, clamp(x + eps_x, 0.0, self.width), y)
        down = self._sample(arr, x, clamp(y - eps_y, 0.0, self.height))
        up = self._sample(arr, x, clamp(y + eps_y, 0.0, self.height))
        return np.array([(right - left) / eps_x, (up - down) / eps_y], dtype=float)

    def _sample(self, arr: np.ndarray, x: float, y: float) -> float:
        r, c = self._index(x, y)
        return float(arr[r, c])



def build_environment(rng: np.random.Generator, cfg: FieldConfig) -> FieldEnvironment:
    xs = np.linspace(0, cfg.width, cfg.grid_cols)
    ys = np.linspace(0, cfg.height, cfg.grid_rows)
    xx, yy = np.meshgrid(xs, ys)

    terrain = np.zeros((cfg.grid_rows, cfg.grid_cols), dtype=float)
    for _ in range(4):
        cx = rng.uniform(0.0, cfg.width)
        cy = rng.uniform(0.0, cfg.height)
        terrain += gaussian_2d(
            xx,
            yy,
            cx,
            cy,
            rng.uniform(18.0, 45.0),
            rng.uniform(12.0, 35.0),
            rng.uniform(0.2, 0.8),
        )
    terrain += 0.20 * (yy / max(cfg.height, 1e-9))
    terrain = rescale01(terrain)

    shade = np.zeros((cfg.grid_rows, cfg.grid_cols), dtype=float)
    for _ in range(6):
        cx = rng.uniform(0.0, cfg.width)
        cy = rng.uniform(0.0, cfg.height)
        shade += gaussian_2d(
            xx,
            yy,
            cx,
            cy,
            rng.uniform(6.0, 18.0),
            rng.uniform(6.0, 18.0),
            rng.uniform(0.3, 1.0),
        )
    shade = rescale01(shade)

    return FieldEnvironment(
        terrain=terrain,
        shade=shade,
        x_coords=xs,
        y_coords=ys,
        width=cfg.width,
        height=cfg.height,
        cols=cfg.grid_cols,
        rows=cfg.grid_rows,
    )
