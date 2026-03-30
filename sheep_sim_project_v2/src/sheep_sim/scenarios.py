from __future__ import annotations

from sheep_sim.config import SimulationConfig



def build_scenario_config(name: str, steps: int, seed: int) -> SimulationConfig:
    cfg = SimulationConfig(seed=seed, scenario=name, steps=steps)

    if name == "abundant":
        cfg.food.n_patches = 18
        cfg.food.total_food_scale = 1.0
        cfg.food.regrowth_rate = 0.0013
        cfg.food.depletion_rate = 0.018
        cfg.transitions.local_food_enter_graze = 0.48
        cfg.transitions.local_food_exit_graze = 0.20
        cfg.transitions.memory_bias = 0.16
        cfg.transitions.explore_bias = 0.08
        cfg.flock.grazing_speed = 0.24
        cfg.flock.travel_speed = 0.95
        cfg.flock.attraction_weight = 0.060
        cfg.flock.alignment_weight = 0.065
    elif name == "scarce":
        cfg.food.n_patches = 7
        cfg.food.total_food_scale = 0.68
        cfg.food.regrowth_rate = 0.00018
        cfg.food.depletion_rate = 0.020
        cfg.transitions.local_food_enter_graze = 0.62
        cfg.transitions.local_food_exit_graze = 0.28
        cfg.transitions.memory_bias = 0.34
        cfg.transitions.explore_bias = 0.24
        cfg.flock.grazing_speed = 0.32
        cfg.flock.travel_speed = 1.30
        cfg.flock.attraction_weight = 0.050
        cfg.flock.alignment_weight = 0.055
    else:
        raise ValueError(f"Unknown scenario: {name}")

    return cfg
