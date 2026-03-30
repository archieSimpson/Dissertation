from __future__ import annotations

from sheep_sim.config import SimulationConfig



def build_scenario_config(name: str, steps: int, seed: int) -> SimulationConfig:
    cfg = SimulationConfig(seed=seed, scenario=name, steps=steps)

    if name == "abundant":
        # Resource-rich pasture: higher carrying capacity, slow patch depletion,
        # strong patch residence, weaker long-range exploration.
        cfg.food.n_patches = 18
        cfg.food.total_food_scale = 1.0
        cfg.food.initial_biomass_fraction = 0.82
        cfg.food.max_biomass_per_cell = 1.15
        cfg.food.daily_regrowth_rate = 0.040
        cfg.food.daily_health_recovery = 0.006
        cfg.food.daily_grazing_pressure_decay = 0.10
        cfg.food.depletion_rate = 0.014
        cfg.food.grazing_impact_per_unit = 0.18
        cfg.food.health_damage_scale = 0.012
        cfg.food.pressure_damage_scale = 0.006
        cfg.food.overgrazing_threshold = 0.30
        cfg.transitions.local_food_enter_graze = 0.42
        cfg.transitions.local_food_exit_graze = 0.18
        cfg.transitions.memory_bias = 0.14
        cfg.transitions.patch_residence_bias = 0.30
        cfg.transitions.explore_bias = 0.05
        cfg.transitions.scarce_search_bias = 0.08
        cfg.flock.grazing_speed = 0.22
        cfg.flock.travel_speed = 0.90
        cfg.flock.attraction_weight = 0.060
        cfg.flock.alignment_weight = 0.065
    elif name == "scarce":
        # Scarce pasture: lower NDVI/carrying capacity, slower regrowth,
        # stronger search and patch-switching behaviour.
        cfg.food.n_patches = 8
        cfg.food.total_food_scale = 0.58
        cfg.food.initial_biomass_fraction = 0.46
        cfg.food.max_biomass_per_cell = 0.78
        cfg.food.daily_regrowth_rate = 0.014
        cfg.food.daily_health_recovery = 0.002
        cfg.food.daily_grazing_pressure_decay = 0.045
        cfg.food.depletion_rate = 0.011
        cfg.food.grazing_impact_per_unit = 0.28
        cfg.food.health_damage_scale = 0.025
        cfg.food.pressure_damage_scale = 0.014
        cfg.food.overgrazing_threshold = 0.22
        cfg.transitions.local_food_enter_graze = 0.075
        cfg.transitions.local_food_exit_graze = 0.040
        cfg.transitions.memory_bias = 0.34
        cfg.transitions.patch_residence_bias = 0.08
        cfg.transitions.explore_bias = 0.26
        cfg.transitions.scarce_search_bias = 0.42
        cfg.flock.grazing_speed = 0.30
        cfg.flock.travel_speed = 1.34
        cfg.flock.attraction_weight = 0.052
        cfg.flock.alignment_weight = 0.058
    else:
        raise ValueError(f"Unknown scenario: {name}")

    return cfg
