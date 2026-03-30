from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sheep_sim.config import FieldConfig, FoodConfig
from sheep_sim.utils import clamp, gaussian_2d, rescale01


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
        row, col = self._index(x, y, field)
        return float(self.biomass[row, col])

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
        return float(np.mean(self.biomass[r0:r1, c0:c1]))

    def local_health(self, x: float, y: float, radius: float, field: FieldConfig) -> float:
        row, col = self._index(x, y, field)
        dx = field.width / field.grid_cols
        dy = field.height / field.grid_rows
        rx = max(1, int(radius / dx))
        ry = max(1, int(radius / dy))
        r0 = max(0, row - ry)
        r1 = min(field.grid_rows, row + ry + 1)
        c0 = max(0, col - rx)
        c1 = min(field.grid_cols, col + rx + 1)
        return float(np.mean(self.health[r0:r1, c0:c1]))

    def deplete(self, x: float, y: float, field: FieldConfig, rate: float, food_cfg: FoodConfig) -> float:
        row, col = self._index(x, y, field)
        available = self.biomass[row, col]
        eaten = min(available, rate)
        self.biomass[row, col] -= eaten
        self.grazing_pressure[row, col] += eaten * food_cfg.grazing_impact_per_unit
        deficit_ratio = 1.0 - (self.biomass[row, col] / max(self.carrying_capacity[row, col], 1e-9))
        overgraze = max(0.0, deficit_ratio - food_cfg.overgrazing_threshold)
        health_loss = food_cfg.health_damage_scale * eaten + food_cfg.pressure_damage_scale * overgraze
        self.health[row, col] = max(0.05, self.health[row, col] - health_loss)
        return float(eaten)

    def regrow(self, dt_days: float, food_cfg: FoodConfig) -> None:
        pressure_decay = np.exp(-food_cfg.daily_grazing_pressure_decay * dt_days)
        self.grazing_pressure *= pressure_decay

        health_recovery = food_cfg.daily_health_recovery * dt_days * (1.0 - self.grazing_pressure)
        self.health = np.clip(self.health + health_recovery, 0.05, 1.0)

        effective_capacity = self.carrying_capacity * self.health
        growth_gap = np.maximum(effective_capacity - self.biomass, 0.0)
        pressure_penalty = np.clip(1.0 - 0.65 * self.grazing_pressure, 0.10, 1.0)
        growth = food_cfg.daily_regrowth_rate * dt_days * self.ndvi * pressure_penalty * growth_gap
        self.biomass = np.minimum(effective_capacity, self.biomass + growth)

    def sensory_gradient(self, x: float, y: float, field: FieldConfig, radius: float) -> np.ndarray:
        eps_x = field.width / field.grid_cols
        eps_y = field.height / field.grid_rows
        left = self.local_mean(clamp(x - eps_x, 0.0, field.width), y, radius, field)
        right = self.local_mean(clamp(x + eps_x, 0.0, field.width), y, radius, field)
        down = self.local_mean(x, clamp(y - eps_y, 0.0, field.height), radius, field)
        up = self.local_mean(x, clamp(y + eps_y, 0.0, field.height), radius, field)
        return np.array([(right - left) / max(eps_x, 1e-9), (up - down) / max(eps_y, 1e-9)], dtype=float)

    def biomass_percentage(self) -> float:
        return 100.0 * float(np.mean(self.biomass) / max(np.mean(self.carrying_capacity), 1e-9))

    def health_percentage(self) -> float:
        return 100.0 * float(np.mean(self.health))

    def mean_pressure(self) -> float:
        return float(np.mean(self.grazing_pressure))

    def degraded_area_percentage(self, threshold: float = 0.60) -> float:
        return 100.0 * float(np.mean(self.health < threshold))

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

    ndvi = np.zeros((field.grid_rows, field.grid_cols), dtype=float)
    base = 0.42 if scenario == "abundant" else 0.16
    ndvi += base

    n_patches = food_cfg.n_patches + (5 if scenario == "abundant" else 0)
    sigma_boost = 1.30 if scenario == "abundant" else 0.72
    amp_scale = 0.42 if scenario == "abundant" else 0.90

    for _ in range(n_patches):
        cx = rng.uniform(0.08 * field.width, 0.92 * field.width)
        cy = rng.uniform(0.08 * field.height, 0.92 * field.height)
        sigma_x = rng.uniform(food_cfg.patch_sigma_min, food_cfg.patch_sigma_max) * sigma_boost
        sigma_y = rng.uniform(food_cfg.patch_sigma_min, food_cfg.patch_sigma_max) * sigma_boost
        amplitude = rng.uniform(0.18, 1.00) * amp_scale
        ndvi += gaussian_2d(xx, yy, cx, cy, sigma_x, sigma_y, amplitude)

    # Low-frequency spatial heterogeneity to mimic pasture quality gradients.
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
