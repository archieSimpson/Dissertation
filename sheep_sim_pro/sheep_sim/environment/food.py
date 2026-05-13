from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sheep_sim.core.config import FieldConfig, FoodConfig
from sheep_sim.core.utils import clamp, gaussian_2d, rescale01


@dataclass(slots=True)
class FoodLandscape:
    biomass: np.ndarray
    carrying_capacity: np.ndarray
    ndvi: np.ndarray
    health: np.ndarray
    grazing_pressure: np.ndarray
    x_coords: np.ndarray
    y_coords: np.ndarray

    @property
    def values(self) -> np.ndarray:
        return self.biomass

    @property
    def max_values(self) -> np.ndarray:
        return self.carrying_capacity

    def sample(self, x: float, y: float, field: FieldConfig) -> float:
        r, c = self._cell(x, y, field)
        return float(self.biomass[r, c])

    def sample_health(self, x: float, y: float, field: FieldConfig) -> float:
        r, c = self._cell(x, y, field)
        return float(self.health[r, c])

    def sample_quality(self, x: float, y: float, field: FieldConfig) -> float:
        r, c = self._cell(x, y, field)
        return float(self.biomass[r, c] * self.health[r, c])

    def _local_window(self, x: float, y: float, radius: float, field: FieldConfig) -> tuple[int, int, int, int]:
        row, col = self._cell(x, y, field)
        rx = max(1, int(radius / (field.width / field.grid_cols)))
        ry = max(1, int(radius / (field.height / field.grid_rows)))
        r0 = max(0, row - ry)
        r1 = min(field.grid_rows, row + ry + 1)
        c0 = max(0, col - rx)
        c1 = min(field.grid_cols, col + rx + 1)
        return r0, r1, c0, c1

    def local_mean(self, x: float, y: float, radius: float, field: FieldConfig) -> float:
        r0, r1, c0, c1 = self._local_window(x, y, radius, field)
        return float(np.mean(self.biomass[r0:r1, c0:c1]))

    def local_health(self, x: float, y: float, radius: float, field: FieldConfig) -> float:
        r0, r1, c0, c1 = self._local_window(x, y, radius, field)
        return float(np.mean(self.health[r0:r1, c0:c1]))

    def local_quality(self, x: float, y: float, radius: float, field: FieldConfig) -> float:
        r0, r1, c0, c1 = self._local_window(x, y, radius, field)
        return float(np.mean(self.biomass[r0:r1, c0:c1] * self.health[r0:r1, c0:c1]))

    def deplete(self, x: float, y: float, field: FieldConfig, rate: float, food_cfg: FoodConfig) -> float:
        row, col = self._cell(x, y, field)
        available = self.biomass[row, col]
        eaten = min(available, rate)
        self.biomass[row, col] -= eaten
        self.grazing_pressure[row, col] += eaten * food_cfg.grazing_impact_per_unit
        deficit_ratio = 1.0 - (self.biomass[row, col] / max(self.carrying_capacity[row, col], 1e-9))
        overgraze = max(0.0, deficit_ratio - food_cfg.overgrazing_threshold)
        health_loss = food_cfg.health_damage_scale * eaten + food_cfg.pressure_damage_scale * overgraze
        self.health[row, col] = max(0.05, self.health[row, col] - health_loss)
        return float(eaten)

    def sensory_gradient(self, x: float, y: float, field: FieldConfig, radius: float) -> np.ndarray:
        ex = field.width / field.grid_cols
        ey = field.height / field.grid_rows
        left  = self.local_quality(clamp(x - ex, 0.0, field.width), y, radius, field)
        right = self.local_quality(clamp(x + ex, 0.0, field.width), y, radius, field)
        down  = self.local_quality(x, clamp(y - ey, 0.0, field.height), radius, field)
        up    = self.local_quality(x, clamp(y + ey, 0.0, field.height), radius, field)
        return np.array([(right - left) / max(ex, 1e-9), (up - down) / max(ey, 1e-9)], dtype=float)

    def biomass_percentage(self) -> float:
        return 100.0 * float(np.mean(self.biomass) / max(np.mean(self.carrying_capacity), 1e-9))

    def health_percentage(self) -> float:
        return 100.0 * float(np.mean(self.health))

    def mean_pressure(self) -> float:
        return float(np.mean(self.grazing_pressure))

    def degraded_area_percentage(self, threshold: float = 0.60) -> float:
        return 100.0 * float(np.mean(self.health < threshold))

    def _cell(self, x: float, y: float, field: FieldConfig) -> tuple[int, int]:
        col = min(field.grid_cols - 1, max(0, int((x / field.width) * field.grid_cols)))
        row = min(field.grid_rows - 1, max(0, int((y / field.height) * field.grid_rows)))
        return row, col


def _make_landscape(ndvi: np.ndarray, food_cfg: FoodConfig, xs: np.ndarray, ys: np.ndarray) -> FoodLandscape:
    carrying_capacity = np.clip(ndvi * food_cfg.max_biomass_per_cell * food_cfg.total_food_scale, 0.02, None)
    health = np.ones_like(carrying_capacity, dtype=float)
    biomass = carrying_capacity * food_cfg.initial_biomass_fraction
    grazing_pressure = np.zeros_like(carrying_capacity, dtype=float)
    return FoodLandscape(
        biomass=biomass,
        carrying_capacity=carrying_capacity,
        ndvi=ndvi,
        health=health,
        grazing_pressure=grazing_pressure,
        x_coords=xs,
        y_coords=ys,
    )


def build_food_landscape(
    rng: np.random.Generator,
    field: FieldConfig,
    food_cfg: FoodConfig,
    scenario: str,
) -> FoodLandscape:
    xs = np.linspace(0, field.width, field.grid_cols)
    ys = np.linspace(0, field.height, field.grid_rows)
    xx, yy = np.meshgrid(xs, ys)

    if scenario == "uniform_low":
        ndvi = np.full((field.grid_rows, field.grid_cols), 0.1, dtype=float)
        return _make_landscape(ndvi, food_cfg, xs, ys)

    if scenario == "uniform_high":
        ndvi = np.full((field.grid_rows, field.grid_cols), 1.0, dtype=float)
        return _make_landscape(ndvi, food_cfg, xs, ys)

    if scenario == "radial_increase":
        cx = field.width  / 2.0
        cy = field.height / 2.0
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        ndvi = np.clip(dist / max_dist, 0.05, 1.0)
        return _make_landscape(ndvi, food_cfg, xs, ys)

    if scenario == "ring":
        cx = field.width  / 2.0
        cy = field.height / 2.0
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        max_dist = np.sqrt(cx ** 2 + cy ** 2)
        inner_radius  = max_dist * 0.22
        ring_inner    = max_dist * 0.45
        ring_outer    = max_dist * 0.68

        ndvi = np.where(dist <= inner_radius, 0.5, 0.0)
        gap_mask  = (dist > inner_radius) & (dist <= ring_inner)
        ring_mask = (dist > ring_inner)   & (dist <= ring_outer)
        ndvi = np.where(gap_mask,  0.1, ndvi)
        ndvi = np.where(ring_mask, 1.0, ndvi)
        ndvi = np.where(dist > ring_outer, 0.05, ndvi)
        return _make_landscape(ndvi, food_cfg, xs, ys)

    if scenario == "corridors":








        cx = field.width  / 2.0
        cy = field.height / 2.0
        column_width  = 25.0
        centre_radius = 22.0

        ndvi = np.full((field.grid_rows, field.grid_cols), 0.0, dtype=float)
        rich_corridor_mask = (xx <= column_width) | (xx >= field.width - column_width)
        ndvi = np.where(rich_corridor_mask, 0.90, ndvi)
        dist_centre = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)



        ndvi = np.where(dist_centre <= centre_radius, 0.55, ndvi)
        return _make_landscape(ndvi, food_cfg, xs, ys)

    ndvi = np.full((field.grid_rows, field.grid_cols), 0.42 if scenario == "abundant" else 0.16, dtype=float)
    n_patches   = food_cfg.n_patches + (5 if scenario == "abundant" else 0)
    sigma_boost = 1.30 if scenario == "abundant" else 0.72
    amp_scale   = 0.42 if scenario == "abundant" else 0.90

    for _ in range(n_patches):
        cx = rng.uniform(0.08 * field.width, 0.92 * field.width)
        cy = rng.uniform(0.08 * field.height, 0.92 * field.height)
        sigma_x   = rng.uniform(food_cfg.patch_sigma_min, food_cfg.patch_sigma_max) * sigma_boost
        sigma_y   = rng.uniform(food_cfg.patch_sigma_min, food_cfg.patch_sigma_max) * sigma_boost
        amplitude = rng.uniform(0.18, 1.00) * amp_scale
        ndvi += gaussian_2d(xx, yy, cx, cy, sigma_x, sigma_y, amplitude)

    for _ in range(3):
        cx = rng.uniform(0.0, field.width)
        cy = rng.uniform(0.0, field.height)
        ndvi += gaussian_2d(xx, yy, cx, cy, rng.uniform(26.0, 60.0), rng.uniform(20.0, 50.0), rng.uniform(-0.14, 0.18))

    ndvi += rng.normal(0.0, food_cfg.ndvi_noise_scale, size=ndvi.shape)
    ndvi = np.clip(rescale01(ndvi), 0.0, 1.0)

    if scenario == "scarce":
        gap_mask = rng.uniform(0.0, 1.0, size=ndvi.shape)
        ndvi = np.where(gap_mask < 0.10, ndvi * 0.08, ndvi)

    carrying_capacity = np.clip(ndvi * food_cfg.max_biomass_per_cell * food_cfg.total_food_scale, 0.02, None)
    health = np.ones_like(carrying_capacity, dtype=float)
    biomass = carrying_capacity * food_cfg.initial_biomass_fraction
    grazing_pressure = np.zeros_like(carrying_capacity, dtype=float)

    return FoodLandscape(
        biomass=biomass,
        carrying_capacity=carrying_capacity,
        ndvi=ndvi,
        health=health,
        grazing_pressure=grazing_pressure,
        x_coords=xs,
        y_coords=ys,
    )
