from __future__ import annotations
 
from dataclasses import dataclass, field
from typing import Any
 
import numpy as np
import pandas as pd
 
from sheep_sim.core.agents import BehaviourState, SheepAgent
from sheep_sim.core.utils import (
    azimuth_deg,
    connected_components_from_distance,
    convex_hull_area,
    mean_nearest_neighbour_distance,
    polarization,
)
from sheep_sim.environment.food import FoodLandscape
 
 
@dataclass
class MetricsRecorder:
    _group: list[dict[str, Any]] = field(default_factory=list)
    _position: list[dict[str, Any]] = field(default_factory=list)
    _field: list[dict[str, Any]] = field(default_factory=list)
 
    def record(
        self,
        step: int,
        flock: list[SheepAgent],
        scenario: str,
        food: FoodLandscape,
    ) -> None:
        positions  = np.array([s.position for s in flock], dtype=float)
        velocities = np.array([s.velocity for s in flock], dtype=float)
        speeds     = np.linalg.norm(velocities, axis=1)
        centroid   = positions.mean(axis=0)
        dists      = np.linalg.norm(positions - centroid, axis=1)
        spread     = float(np.mean(dists))
 
        hull_area  = convex_hull_area(positions)
        nn_dist    = mean_nearest_neighbour_distance(positions)
        pol        = polarization(velocities)
        clusters   = connected_components_from_distance(positions, threshold=15.0)
        xs, ys     = positions[:, 0], positions[:, 1]
        elongation = float((xs.max() - xs.min()) / max(ys.max() - ys.min(), 1e-9))
 
        speed_mean = float(np.mean(speeds))
        counts     = {st.value: int(sum(1 for s in flock if s.state == st)) for st in BehaviourState}
        mean_intake  = float(np.mean([s.last_food_intake for s in flock]))
        azimuth = (
            float(np.mean([azimuth_deg(v) for v in velocities if np.linalg.norm(v) > 1e-6]))
            if speed_mean > 0 else 0.0
        )
 


        nnd_per_agent = []
        for s in flock:
            dists_to_others = [
                float(np.linalg.norm(s.position - o.position))
                for o in flock if o.sheep_id != s.sheep_id
            ]
            nnd_per_agent.append(min(dists_to_others))
        nnd_std = float(np.std(nnd_per_agent))
 

        flock_compactness = float(np.mean(dists < 19.0))
 

        within_radius = 0
        total_pairs = 0
        for i, s in enumerate(flock):
            for o in flock[i+1:]:
                total_pairs += 1
                if np.linalg.norm(s.position - o.position) <= 19.0:
                    within_radius += 1
        pair_cohesion = float(within_radius / max(total_pairs, 1))
 

        contagion_active = int(sum(1 for s in flock if s.contagion_steps > 0))
 


        speed_std = float(np.std(speeds))
 

        resting_sustained = int(sum(
            1 for s in flock
            if s.state == BehaviourState.RESTING and s.state_age > 1
        ))
 

        mean_path_length     = float(np.mean([s.path_length for s in flock]))
        mean_cumulative_food = float(np.mean([s.cumulative_food for s in flock]))
 

        memory_density = 0.0
        memory_mean_value = 0.0
        if flock[0].memory_map is not None:
            all_maps = np.array([s.memory_map for s in flock])
            memory_density    = float(np.mean(all_maps > 1e-6))
            memory_mean_value = float(np.mean(all_maps[all_maps > 1e-6])) if np.any(all_maps > 1e-6) else 0.0
 


        speed_variance_agents = float(np.var(speeds))
 

        nnd_variance = float(np.var(nnd_per_agent))
 

        cumfood_std = float(np.std([s.cumulative_food for s in flock]))
 

        path_length_std = float(np.std([s.path_length for s in flock]))
 

        mean_ou_magnitude = float(np.mean([
            abs(s.ou_attraction) + abs(s.ou_alignment) + abs(s.ou_repulsion)
            for s in flock
        ]))
 

        separation_std = float(np.std([s.separation_steps for s in flock]))
 





        mean_displacement = float(np.mean(dists))
 



        fresh_walkers = int(sum(
            1 for s in flock
            if s.state == BehaviourState.WALKING and s.state_age <= 2
        ))
 

        mean_state_age = float(np.mean([s.state_age for s in flock]))
        state_age_std  = float(np.std([s.state_age for s in flock]))
 

        intake_efficiency = float(
            np.mean([s.last_food_intake for s in flock if s.state == BehaviourState.GRAZING])
            if counts.get("grazing", 0) > 0 else 0.0
        )
 

        self._group.append({

            "step": step, "scenario": scenario,
            "centroid_x": centroid[0], "centroid_y": centroid[1],
            "spread": spread, "polarization": pol,
            "mean_nearest_neighbour_distance": nn_dist,
            "convex_hull_area": hull_area,
            "elongation_ratio": elongation,
            "number_of_clusters": clusters,
            "grazing_count":    counts.get("grazing", 0),
            "walking_count":    counts.get("walking", 0),
            "travelling_count": counts.get("travelling", 0),
            "regrouping_count": counts.get("regrouping", 0),
            "resting_count":    counts.get("resting", 0),
            "mean_speed": speed_mean,
            "mean_food_intake": mean_intake,
            "field_mean_food": float(np.mean(food.values)),
            "field_health_pct": food.health_percentage(),
            "field_biomass_pct": food.biomass_percentage(),
            "mean_grazing_pressure": food.mean_pressure(),
            "degraded_area_pct": food.degraded_area_percentage(),
            "mean_ndvi_pct": 100.0 * float(np.mean(food.ndvi)),
            "mean_azimuth_deg": azimuth,

            "nnd_std":            nnd_std,
            "flock_compactness":  flock_compactness,
            "pair_cohesion":      pair_cohesion,
            "contagion_active":   contagion_active,

            "speed_std":          speed_std,
            "resting_sustained":  resting_sustained,

            "mean_path_length":      mean_path_length,
            "mean_cumulative_food":  mean_cumulative_food,
            "memory_density":        memory_density,
            "memory_mean_value":     memory_mean_value,

            "speed_variance_agents": speed_variance_agents,
            "nnd_variance":          nnd_variance,
            "cumfood_std":           cumfood_std,
            "path_length_std":       path_length_std,
            "mean_ou_magnitude":     mean_ou_magnitude,
            "separation_std":        separation_std,

            "mean_displacement":     mean_displacement,
            "fresh_walkers":         fresh_walkers,
            "mean_state_age":        mean_state_age,
            "state_age_std":         state_age_std,
            "intake_efficiency":     intake_efficiency,
        })
 
        self._field.append({
            "step": step, "scenario": scenario,
            "field_health_pct": food.health_percentage(),
            "field_biomass_pct": food.biomass_percentage(),
            "mean_grazing_pressure": food.mean_pressure(),
            "degraded_area_pct": food.degraded_area_percentage(),
            "mean_ndvi_pct": 100.0 * float(np.mean(food.ndvi)),
        })
 
        for s in flock:
            self._position.append({
                "step": step, "scenario": scenario,
                "sheep_id": s.sheep_id,
                "x": s.position[0], "y": s.position[1],
                "vx": s.velocity[0], "vy": s.velocity[1],
                "speed": s.speed(), "state": s.state.value,
                "last_food_intake": s.last_food_intake,
                "cumulative_food": s.cumulative_food,
                "path_length": s.path_length,

                "boldness":         s.boldness,
                "sociability":      s.sociability,
                "movement_vigor":   s.movement_vigor,
                "turning_bias":     s.turning_bias,
                "patch_leave_bias": s.patch_leave_bias,
                "preferred_spacing":s.preferred_spacing,
                "site_fidelity":    s.site_fidelity,
                "separation_steps": s.separation_steps,
                "state_age":        s.state_age,
            })
 
    def group_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._group)
 
    def position_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._position)
 
    def field_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._field)
 
    @property
    def n_steps_recorded(self) -> int:
        return len(self._group)
 
    def last_group_row(self) -> dict[str, Any]:
        return self._group[-1] if self._group else {}