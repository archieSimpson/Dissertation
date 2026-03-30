from __future__ import annotations

from dataclasses import replace

from sheep_sim.config import SimulationConfig


def build_scenario_config(name: str, steps: int, seed: int) -> SimulationConfig:
    cfg = SimulationConfig(seed=seed, scenario=name, steps=steps)
    cfg.flock = replace(cfg.flock, n_sheep=40, neighbour_radius=12.0, repulsion_radius=3.5, attraction_weight=0.45, alignment_weight=0.20, repulsion_weight=1.35, resting_speed=0.02, stochastic_turn_std=0.20)
    if name == 'abundant':
        cfg.food = replace(cfg.food, n_patches=18, total_food_scale=1.0, initial_biomass_fraction=0.82, max_biomass_per_cell=1.15, daily_regrowth_rate=0.040, daily_health_recovery=0.006, daily_grazing_pressure_decay=0.10, depletion_rate=0.014, grazing_impact_per_unit=0.18, health_damage_scale=0.012, pressure_damage_scale=0.006, overgrazing_threshold=0.30, sensory_radius=8.0)
        cfg.transitions = replace(cfg.transitions, local_food_enter_graze=0.48, local_food_exit_graze=0.34, memory_bias=0.06, patch_residence_bias=0.40, explore_bias=0.02, scarce_search_bias=0.04, dispersion_regroup_threshold=0.18, stale_patch_quality_threshold=0.10, stale_patch_steps_abundant=14)
        cfg.flock = replace(cfg.flock, grazing_speed=0.20, walking_speed=0.34, travel_speed=0.55, regroup_speed=0.36, max_speed=0.85, stochastic_turn_std=0.19)
    elif name == 'scarce':
        cfg.food = replace(cfg.food, n_patches=8, total_food_scale=0.58, initial_biomass_fraction=0.46, max_biomass_per_cell=0.78, daily_regrowth_rate=0.014, daily_health_recovery=0.002, daily_grazing_pressure_decay=0.045, depletion_rate=0.011, grazing_impact_per_unit=0.28, health_damage_scale=0.025, pressure_damage_scale=0.014, overgrazing_threshold=0.22, sensory_radius=10.0)
        cfg.transitions = replace(cfg.transitions, local_food_enter_graze=0.44, local_food_exit_graze=0.34, memory_bias=0.34, patch_residence_bias=0.08, explore_bias=0.26, scarce_search_bias=0.42, dispersion_regroup_threshold=0.20, stale_patch_quality_threshold=0.08, stale_patch_steps_scarce=10)
        cfg.flock = replace(cfg.flock, grazing_speed=0.20, walking_speed=0.42, travel_speed=0.80, regroup_speed=0.42, max_speed=1.05, stochastic_turn_std=0.20)
    else:
        raise ValueError(f'Unknown scenario: {name}')
    return cfg
