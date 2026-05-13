from __future__ import annotations

from dataclasses import replace

from sheep_sim.core.config import SimulationConfig









def _apply_abundant_params(cfg: SimulationConfig) -> SimulationConfig:
    cfg = replace(cfg, food=replace(cfg.food,
        n_patches=18, total_food_scale=1.0, initial_biomass_fraction=0.82,
        max_biomass_per_cell=1.15,
        depletion_rate=0.014, grazing_impact_per_unit=0.18,
        health_damage_scale=0.012, pressure_damage_scale=0.006,
        overgrazing_threshold=0.30, sensory_radius=20.0,
    ))
    cfg = replace(cfg, transitions=replace(cfg.transitions,
        local_food_enter_graze=0.42, local_food_exit_graze=0.31,
        memory_bias=0.08, patch_residence_bias=0.56,
        explore_bias=0.03, scarce_search_bias=0.04,
        dispersion_regroup_threshold=0.16,
        stale_patch_quality_threshold=0.08, stale_patch_steps_abundant=18,
    ))
    cfg = replace(cfg, flock=replace(cfg.flock,
        attraction_weight=0.52, alignment_weight=0.24, repulsion_weight=1.18,
        grazing_speed=2.4, walking_speed=12.0, travel_speed=27.0,
        regroup_speed=18.0, resting_speed=0.3, max_speed=35.0,
        stochastic_turn_std=0.55,
    ))
    cfg = replace(cfg, field=replace(cfg.field,
        boundary_turn_strength=80.0, boundary_margin=8.0,
    ))
    return cfg


def _apply_scarce_params(cfg: SimulationConfig) -> SimulationConfig:
    cfg = replace(cfg, field=replace(cfg.field,
        boundary_turn_strength=80.0, boundary_margin=8.0,
    ))
    cfg = replace(cfg, food=replace(cfg.food,
        n_patches=8, total_food_scale=0.58, initial_biomass_fraction=0.46,
        max_biomass_per_cell=0.78,
        depletion_rate=0.011, grazing_impact_per_unit=0.28,
        health_damage_scale=0.025, pressure_damage_scale=0.014,
        overgrazing_threshold=0.22, sensory_radius=25.0,
    ))
    cfg = replace(cfg, transitions=replace(cfg.transitions,
        local_food_enter_graze=0.04,
        local_food_exit_graze=0.020,
        memory_bias=0.34, patch_residence_bias=0.08,
        explore_bias=0.26, scarce_search_bias=0.55,
        dispersion_regroup_threshold=0.20,
        stale_patch_quality_threshold=0.03, stale_patch_steps_scarce=6,
    ))
    cfg = replace(cfg, flock=replace(cfg.flock,
        grazing_speed=2.4, walking_speed=14.0, travel_speed=32.0,
        regroup_speed=20.0, max_speed=42.0, stochastic_turn_std=0.38,
    ))
    return cfg


def _apply_corridors_params(cfg: SimulationConfig) -> SimulationConfig:
    """Custom corridor parameters separate from abundant/scarce.

    Designed to produce the balanced fission pattern: flock splits roughly
    50/50 between the two rich strips, agents spread along the corridor
    (multiple sub-clusters in each), and grazing intensity is high enough
    that degradation contours develop on both corridors during the day.

    Differences from abundant:
      - attraction_weight 0.52 -> 0.28      (weaker social pull -> non-unanimous fission)
      - alignment_weight  0.24 -> 0.14
      - repulsion_weight  1.18 -> 1.10
      - repulsion_radius  5    -> 8         (wider personal space, sub-clusters)
      - stochastic_turn_std 0.55 -> 0.50
      - ring-style foraging FSM thresholds (low entry, high memory, low residence)
      - pressure_damage_scale 0.006 -> 0.012 (degradation contours visible)
    """
    cfg = replace(cfg, food=replace(cfg.food,
        n_patches=1,
        total_food_scale=1.0, initial_biomass_fraction=0.88,
        max_biomass_per_cell=1.0,
        depletion_rate=0.013, grazing_impact_per_unit=0.20,
        health_damage_scale=0.015, pressure_damage_scale=0.012,
        overgrazing_threshold=0.25, sensory_radius=45.0,
    ))
    cfg = replace(cfg, transitions=replace(cfg.transitions,
        local_food_enter_graze=0.22, local_food_exit_graze=0.14,
        explore_bias=0.32, scarce_search_bias=0.38,
        memory_bias=0.55, patch_residence_bias=0.15,
        stale_patch_quality_threshold=0.06, stale_patch_steps_abundant=6,
    ))
    cfg = replace(cfg, flock=replace(cfg.flock,
        attraction_weight=0.28, alignment_weight=0.14, repulsion_weight=1.10,
        repulsion_radius=8.0,
        grazing_speed=2.4, walking_speed=12.0, travel_speed=27.0,
        regroup_speed=18.0, max_speed=35.0, stochastic_turn_std=0.50,
    ))
    cfg = replace(cfg, field=replace(cfg.field,
        boundary_turn_strength=80.0, boundary_margin=8.0,
    ))
    return cfg




_FROZEN_PARENT = {
    "uniform_high":    "abundant",
    "radial_increase": "abundant",
    "ring":            "abundant",
    "uniform_low":     "scarce",
}


def build_scenario_config(
    name: str,
    steps: int = 2500,
    seed: int = 42,
    landscape_seed: int = 7,
) -> SimulationConfig:
    if name not in {"abundant","scarce","uniform_low","uniform_high","radial_increase","ring","corridors"}:
        raise ValueError(
            f"Unknown scenario '{name}'. "
            "Choose 'abundant', 'scarce', 'uniform_low', 'uniform_high', "
            "'radial_increase', 'ring', or 'corridors'."
        )

    cfg = SimulationConfig(seed=seed, landscape_seed=landscape_seed, scenario=name, steps=steps)

    cfg = replace(cfg, flock=replace(
        cfg.flock,
        n_sheep=40,
        neighbour_radius=19.0,
        alignment_radius=13.0,
        repulsion_radius=5.0,
        attraction_weight=0.46,
        alignment_weight=0.20,
        repulsion_weight=1.30,
        resting_speed=0.3,
        stochastic_turn_std=0.45,
    ))

    if name == "abundant":
        cfg = _apply_abundant_params(cfg)

    elif name == "scarce":
        cfg = _apply_scarce_params(cfg)

    elif name in _FROZEN_PARENT:



        parent = _FROZEN_PARENT[name]
        if parent == "abundant":
            cfg = _apply_abundant_params(cfg)
        else:
            cfg = _apply_scarce_params(cfg)

    elif name == "corridors":
        cfg = _apply_corridors_params(cfg)

    return cfg
