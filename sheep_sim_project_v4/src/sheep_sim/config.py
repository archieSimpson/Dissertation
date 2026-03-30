from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Literal

ScenarioName = Literal["abundant", "scarce"]
RenderMode = Literal["none", "live", "final"]


@dataclass(slots=True)
class FieldConfig:
    width: float = 220.0
    height: float = 140.0
    grid_cols: int = 88
    grid_rows: int = 56
    boundary_turn_strength: float = 2.8
    terrain_strength: float = 0.35
    shade_strength: float = 0.10


@dataclass(slots=True)
class FoodConfig:
    n_patches: int = 12
    patch_sigma_min: float = 5.0
    patch_sigma_max: float = 18.0
    total_food_scale: float = 1.0
    sensory_radius: float = 8.0
    memory_decay: float = 0.0005
    initial_biomass_fraction: float = 0.72
    max_biomass_per_cell: float = 1.0
    daily_regrowth_rate: float = 0.035
    daily_health_recovery: float = 0.004
    daily_grazing_pressure_decay: float = 0.08
    grazing_impact_per_unit: float = 0.22
    health_damage_scale: float = 0.020
    pressure_damage_scale: float = 0.010
    overgrazing_threshold: float = 0.24
    depletion_rate: float = 0.016
    ndvi_noise_scale: float = 0.08


@dataclass(slots=True)
class FlockConfig:
    n_sheep: int = 40
    neighbour_radius: float = 12.0
    repulsion_radius: float = 3.5
    attraction_weight: float = 0.45
    alignment_weight: float = 0.20
    repulsion_weight: float = 1.35
    cohesion_target_weight: float = 0.020
    stochastic_turn_std: float = 0.20
    max_speed: float = 1.10
    grazing_speed: float = 0.20
    walking_speed: float = 0.34
    travel_speed: float = 0.55
    regroup_speed: float = 0.36
    resting_speed: float = 0.02


@dataclass(slots=True)
class CircadianConfig:
    day_length_steps: int = 1200
    active_peak_shift: float = 0.10
    active_amplitude: float = 0.42
    resting_threshold: float = 0.28


@dataclass(slots=True)
class StateTransitionConfig:
    local_food_enter_graze: float = 0.48
    local_food_exit_graze: float = 0.34
    crowding_regroup_threshold: float = 0.42
    dispersion_regroup_threshold: float = 0.72
    energy_loss_per_step: float = 0.0011
    food_gain_scale: float = 0.028
    fatigue_recovery_scale: float = 0.009
    explore_bias: float = 0.15
    memory_bias: float = 0.26
    patch_residence_bias: float = 0.22
    scarce_search_bias: float = 0.35
    stale_patch_quality_threshold: float = 0.10
    stale_patch_steps_abundant: int = 14
    stale_patch_steps_scarce: int = 10


@dataclass(slots=True)
class OutputConfig:
    record_every: int = 1
    live_interval_ms: int = 40
    render_every_n_steps: int = 4


@dataclass(slots=True)
class TimeConfig:
    real_seconds_per_step: float = 30.0
    render_fps: int = 25
    start_hour: int = 6
    start_minute: int = 0


@dataclass(slots=True)
class SimulationConfig:
    seed: int = 7
    scenario: ScenarioName = "abundant"
    steps: int = 2500
    dt: float = 1.0
    field: FieldConfig = dc_field(default_factory=FieldConfig)
    food: FoodConfig = dc_field(default_factory=FoodConfig)
    flock: FlockConfig = dc_field(default_factory=FlockConfig)
    circadian: CircadianConfig = dc_field(default_factory=CircadianConfig)
    transitions: StateTransitionConfig = dc_field(default_factory=StateTransitionConfig)
    output: OutputConfig = dc_field(default_factory=OutputConfig)
    time: TimeConfig = dc_field(default_factory=TimeConfig)
