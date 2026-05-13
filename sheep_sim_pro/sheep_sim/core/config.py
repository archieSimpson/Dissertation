from __future__ import annotations
 
from dataclasses import dataclass, field as dc_field, replace
from typing import Literal
 
ScenarioName = Literal["abundant", "scarce", "uniform_low", "uniform_high", "radial_increase", "ring", "corridors"]
RenderMode   = Literal["none", "live", "final"]
 
 
@dataclass(frozen=True)
class FieldConfig:
    width: float = 220.0
    height: float = 140.0
    grid_cols: int = 88
    grid_rows: int = 56
    boundary_turn_strength: float = 60.0
    boundary_margin: float = 8.0
    terrain_strength: float = 0.35
 
    @property
    def cell_width(self) -> float:
        return self.width / self.grid_cols
 
    @property
    def cell_height(self) -> float:
        return self.height / self.grid_rows
 
 
@dataclass(frozen=True)
class FoodConfig:
    n_patches: int = 12
    patch_sigma_min: float = 5.0
    patch_sigma_max: float = 18.0
    total_food_scale: float = 1.0
    sensory_radius: float = 20.0
    memory_decay: float = 0.0005
    initial_biomass_fraction: float = 0.72
    max_biomass_per_cell: float = 1.0
    grazing_impact_per_unit: float = 0.22
    health_damage_scale: float = 0.020
    pressure_damage_scale: float = 0.010
    overgrazing_threshold: float = 0.24
    depletion_rate: float = 0.016
    ndvi_noise_scale: float = 0.08
 
 
@dataclass(frozen=True)
class FlockConfig:
    n_sheep: int = 40







    neighbour_radius: float = 19.0
    alignment_radius: float = 19.0
    repulsion_radius: float = 8.0
    attraction_weight: float = 0.45
    alignment_weight: float = 0.20
    repulsion_weight: float = 1.35
    cohesion_target_weight: float = 0.020
    stochastic_turn_std: float = 0.45
    max_speed: float = 35.0
    grazing_speed: float = 2.4
    walking_speed: float = 12.0
    travel_speed: float = 27.0
    regroup_speed: float = 18.0
    resting_speed: float = 0.3
 
 
@dataclass(frozen=True)
class CircadianConfig:
    day_length_steps: int = 1920
    active_peak_shift: float = -0.25
    active_amplitude: float = 0.80
    resting_threshold: float = 0.28
 
 
@dataclass(frozen=True)
class StateTransitionConfig:
    local_food_enter_graze: float = 0.48
    local_food_exit_graze: float = 0.34
    crowding_regroup_threshold: float = 0.42
    dispersion_regroup_threshold: float = 0.72
    explore_bias: float = 0.15
    memory_bias: float = 0.26
    patch_residence_bias: float = 0.22
    scarce_search_bias: float = 0.35
    stale_patch_quality_threshold: float = 0.10
    stale_patch_steps_abundant: int = 14
    stale_patch_steps_scarce: int = 10
 
 
@dataclass(frozen=True)
class OutputConfig:
    record_every: int = 1
    live_interval_ms: int = 40
    render_every_n_steps: int = 4
 
 
@dataclass(frozen=True)
class TimeConfig:
    real_seconds_per_step: float = 30.0
    render_fps: int = 25
    start_hour: int = 6
    start_minute: int = 0
 
 
@dataclass(frozen=True)
class FeatureConfig:
    """Independent toggles for each behavioural subsystem.
 
    With all True, the model matches the full six-mechanism setup.
    With all False, agents perform a correlated random walk: boundary force
    and stochastic turn noise only. No food awareness, no patch departure,
    no social cohesion, no circadian rhythm, no memory, no personality
    variation, no terrain response.
    """
    foraging: bool = True
    social: bool = True
    circadian: bool = True
    memory: bool = True
    personality: bool = True
    terrain: bool = True
 
 
@dataclass(frozen=True)
class SimulationConfig:
    seed: int = 42
    landscape_seed: int = 7
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
    features: FeatureConfig = dc_field(default_factory=FeatureConfig)
 
    def with_time(self, **kwargs) -> SimulationConfig:
        return replace(self, time=replace(self.time, **kwargs))
 
    def with_output(self, **kwargs) -> SimulationConfig:
        return replace(self, output=replace(self.output, **kwargs))