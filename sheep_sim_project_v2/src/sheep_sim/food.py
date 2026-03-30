from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sheep_sim.config import FieldConfig, FoodConfig
from sheep_sim.utils import clamp, gaussian_2d, rescale01


@dataclass(slots=True)
class FoodLandscape:
    values: np.ndarray
    max_values: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray

    def sample(self, x: float, y: float, field: FieldConfig) -> float:
        row, col = self._index(x, y, field)
        return float(self.values[row, col])

    def local_mean(self, x: float, y: float, radius: float, field: FieldConfig) -> float:
        row, col = self._index(x, y, field)
        dx = field.width / field.grid_cols
        dy = field.height / field.grid_rows
        rx = max(1, int(radius / dx))
        ry = max(1, int(radius / dy))
        r0 = max(0, row - ry)
        r1 = min(field.grid_rows, row + ry + 1)
        c0 = max(0, col - rx)
        c1 = min(field.grid_cols, col + rx + 1)
        return float(np.mean(self.values[r0:r1, c0:c1]))

    def deplete(self, x: float, y: float, field: FieldConfig, rate: float) -> float:
        row, col = self._index(x, y, field)
        available = self.values[row, col]
        eaten = min(available, rate)
        self.values[row, col] -= eaten
        return float(eaten)

    def regrow(self, rate: float) -> None:
        self.values = np.minimum(self.max_values, self.values + rate * self.max_values)

    def sensory_gradient(self, x: float, y: float, field: FieldConfig, radius: float) -> np.ndarray:
        eps_x = field.width / field.grid_cols
        eps_y = field.height / field.grid_rows
        left = self.local_mean(clamp(x - eps_x, 0.0, field.width), y, radius, field)
        right = self.local_mean(clamp(x + eps_x, 0.0, field.width), y, radius, field)
        down = self.local_mean(x, clamp(y - eps_y, 0.0, field.height), radius, field)
        up = self.local_mean(x, clamp(y + eps_y, 0.0, field.height), radius, field)
        return np.array([(right - left) / max(eps_x, 1e-9), (up - down) / max(eps_y, 1e-9)], dtype=float)

    def _index(self, x: float, y: float, field: FieldConfig) -> tuple[int, int]:
        col = min(field.grid_cols - 1, max(0, int((x / field.width) * field.grid_cols)))
        row = min(field.grid_rows - 1, max(0, int((y / field.height) * field.grid_rows)))
        return row, col



def build_food_landscape(
    rng: np.random.Generator,
    field: FieldConfig,
    food_cfg: FoodConfig,
    scenario: str,
) -> FoodLandscape:
    xs = np.linspace(0, field.width, field.grid_cols)
    ys = np.linspace(0, field.height, field.grid_rows)
    xx, yy = np.meshgrid(xs, ys)
    values = np.zeros((field.grid_rows, field.grid_cols), dtype=float)

    if scenario == "abundant":
        base = 0.35
        n_patches = food_cfg.n_patches + 6
        amp_scale = 0.50
        sigma_boost = 1.35
    else:
        base = 0.05
        n_patches = max(4, food_cfg.n_patches - 4)
        amp_scale = 0.85
        sigma_boost = 0.75

    values += base

    for _ in range(n_patches):
        cx = rng.uniform(0.08 * field.width, 0.92 * field.width)
        cy = rng.uniform(0.08 * field.height, 0.92 * field.height)
        sigma_x = rng.uniform(food_cfg.patch_sigma_min, food_cfg.patch_sigma_max) * sigma_boost
        sigma_y = rng.uniform(food_cfg.patch_sigma_min, food_cfg.patch_sigma_max) * sigma_boost
        amplitude = rng.uniform(0.18, 1.00) * amp_scale
        values += gaussian_2d(xx, yy, cx, cy, sigma_x, sigma_y, amplitude)

    if scenario == "scarce":
        gap_mask = rng.uniform(0.0, 1.0, size=values.shape)
        values = np.where(gap_mask < 0.12, values * 0.15, values)

    values = rescale01(values) * food_cfg.total_food_scale
    max_values = values.copy()
    return FoodLandscape(values=values, max_values=max_values, x_coords=xs, y_coords=ys)
