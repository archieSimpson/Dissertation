from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from sheep_sim.agents import BehaviourState, SheepAgent
from sheep_sim.food import FoodLandscape
from sheep_sim.utils import azimuth_deg, connected_components_from_distance, convex_hull_area, mean_nearest_neighbour_distance, polarization


@dataclass(slots=True)
class MetricsRecorder:
    group_rows: list[dict[str, Any]]
    position_rows: list[dict[str, Any]]
    field_rows: list[dict[str, Any]]

    def __init__(self) -> None:
        self.group_rows = []
        self.position_rows = []
        self.field_rows = []

    def record(self, step: int, flock: list[SheepAgent], scenario: str, food: FoodLandscape) -> None:
        positions = np.array([s.position for s in flock], dtype=float)
        velocities = np.array([s.velocity for s in flock], dtype=float)
        centroid = positions.mean(axis=0)
        dists = np.linalg.norm(positions - centroid, axis=1)
        spread = float(np.mean(dists))
        hull_area = convex_hull_area(positions)
        nn = mean_nearest_neighbour_distance(positions)
        pol = polarization(velocities)
        comps = connected_components_from_distance(positions, threshold=10.0)
        xs = positions[:, 0]
        ys = positions[:, 1]
        elongation = float((xs.max() - xs.min()) / max(ys.max() - ys.min(), 1e-9))
        speed_mean = float(np.mean(np.linalg.norm(velocities, axis=1)))
        counts = {state.value: int(sum(1 for s in flock if s.state == state)) for state in BehaviourState}
        mean_energy = float(np.mean([s.energy for s in flock]))
        mean_food_intake = float(np.mean([s.last_food_intake for s in flock]))
        azimuth = float(np.mean([azimuth_deg(v) for v in velocities if np.linalg.norm(v) > 1e-6])) if speed_mean > 0 else 0.0
        self.group_rows.append({
            'step': step, 'scenario': scenario, 'centroid_x': centroid[0], 'centroid_y': centroid[1],
            'spread': spread, 'polarization': pol, 'mean_nearest_neighbour_distance': nn,
            'convex_hull_area': hull_area, 'elongation_ratio': elongation, 'number_of_clusters': comps,
            'grazing_count': counts.get('grazing', 0), 'walking_count': counts.get('walking', 0),
            'travelling_count': counts.get('travelling', 0), 'regrouping_count': counts.get('regrouping', 0),
            'resting_count': counts.get('resting', 0), 'mean_speed': speed_mean, 'mean_energy': mean_energy,
            'mean_food_intake': mean_food_intake, 'field_mean_food': float(np.mean(food.values)),
            'field_health_pct': food.health_percentage(), 'field_biomass_pct': food.biomass_percentage(),
            'mean_grazing_pressure': food.mean_pressure(), 'degraded_area_pct': food.degraded_area_percentage(),
            'mean_ndvi_pct': 100.0 * float(np.mean(food.ndvi)), 'mean_azimuth_deg': azimuth,
        })
        self.field_rows.append({
            'step': step, 'scenario': scenario, 'field_health_pct': food.health_percentage(),
            'field_biomass_pct': food.biomass_percentage(), 'mean_grazing_pressure': food.mean_pressure(),
            'degraded_area_pct': food.degraded_area_percentage(), 'mean_ndvi_pct': 100.0 * float(np.mean(food.ndvi)),
        })
        for s in flock:
            self.position_rows.append({
                'step': step, 'scenario': scenario, 'sheep_id': s.sheep_id, 'x': s.position[0], 'y': s.position[1],
                'vx': s.velocity[0], 'vy': s.velocity[1], 'speed': s.speed(), 'state': s.state.value,
                'energy': s.energy, 'last_food_intake': s.last_food_intake, 'cumulative_food': s.cumulative_food,
                'path_length': s.path_length,
            })

    def group_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.group_rows)

    def position_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.position_rows)

    def field_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.field_rows)
