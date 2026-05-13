from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sheep_sim.core.config import FieldConfig
from sheep_sim.core.utils import clamp, gaussian_2d


@dataclass(slots=True)
class FieldEnvironment:
    terrain: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray
    width: float
    height: float
    cols: int
    rows: int

    def in_bounds(self, pos: np.ndarray) -> bool:
        return 0.0 <= pos[0] <= self.width and 0.0 <= pos[1] <= self.height

    def clamp_position(self, pos: np.ndarray) -> np.ndarray:
        return np.array(
            [clamp(pos[0], 0.0, self.width), clamp(pos[1], 0.0, self.height)],
            dtype=float,
        )

    def sample_terrain(self, x: float, y: float) -> float:
        r, c = self._cell(x, y)
        return float(self.terrain[r, c])

    def terrain_gradient(self, x: float, y: float) -> np.ndarray:
        return self._grad(self.terrain, x, y)

    def boundary_force(self, pos: np.ndarray, margin: float = 20.0) -> np.ndarray:
        x, y = float(pos[0]), float(pos[1])
        fx = fy = 0.0
        if x < margin:
            strength = ((margin - x) / margin) ** 2
            fx += strength
        elif x > self.width - margin:
            strength = ((x - (self.width - margin)) / margin) ** 2
            fx -= strength
        if y < margin:
            strength = ((margin - y) / margin) ** 2
            fy += strength
        elif y > self.height - margin:
            strength = ((y - (self.height - margin)) / margin) ** 2
            fy -= strength
        return np.array([fx, fy], dtype=float)

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        col = min(self.cols - 1, max(0, int((x / self.width) * self.cols)))
        row = min(self.rows - 1, max(0, int((y / self.height) * self.rows)))
        return row, col

    def _sample_arr(self, arr: np.ndarray, x: float, y: float) -> float:
        r, c = self._cell(x, y)
        return float(arr[r, c])

    def _grad(self, arr: np.ndarray, x: float, y: float) -> np.ndarray:
        ex = self.width / self.cols
        ey = self.height / self.rows
        left  = self._sample_arr(arr, clamp(x - ex, 0.0, self.width), y)
        right = self._sample_arr(arr, clamp(x + ex, 0.0, self.width), y)
        down  = self._sample_arr(arr, x, clamp(y - ey, 0.0, self.height))
        up    = self._sample_arr(arr, x, clamp(y + ey, 0.0, self.height))
        return np.array([(right - left) / max(ex, 1e-9), (up - down) / max(ey, 1e-9)], dtype=float)


def build_environment(rng: np.random.Generator, cfg: FieldConfig) -> FieldEnvironment:
    xs = np.linspace(0, cfg.width, cfg.grid_cols)
    ys = np.linspace(0, cfg.height, cfg.grid_rows)
    xx, yy = np.meshgrid(xs, ys)

    terrain = np.zeros((cfg.grid_rows, cfg.grid_cols), dtype=float)
    for _ in range(4):
        cx = rng.uniform(0.0, cfg.width)
        cy = rng.uniform(0.0, cfg.height)
        terrain += gaussian_2d(xx, yy, cx, cy,
                               rng.uniform(18.0, 45.0), rng.uniform(12.0, 35.0),
                               rng.uniform(0.2, 0.8))

    return FieldEnvironment(
        terrain=terrain,
        x_coords=xs, y_coords=ys,
        width=cfg.width, height=cfg.height,
        cols=cfg.grid_cols, rows=cfg.grid_rows,
    )
