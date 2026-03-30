from dataclasses import dataclass


@dataclass
class SimulationConfig:
    # Window / timing
    width: int = 1400
    height: int = 900
    fps: int = 60
    dt: float = 1.0 / 60.0

    # World
    world_margin: int = 40

    # Goal region
    goal_x: int = 1180
    goal_y: int = 80
    goal_w: int = 140
    goal_h: int = 140

    # Agents
    num_sheep: int = 45
    sheep_radius: int = 6
    dog_radius: int = 9

    # Sheep neighbourhood
    neighbour_radius: float = 85.0
    separation_radius: float = 24.0
    use_local_neighbours_only: bool = True
    k_neighbours: int = 8

    # Vision cone
    vision_angle_deg: float = 270.0

    # Cluster metric
    cluster_distance_threshold: float = 70.0

    # Dog influence
    dog_influence_radius: float = 170.0
    dog_hard_repulsion_radius: float = 60.0
    use_directional_dog_pressure: bool = True

    # Walls
    boundary_repulsion_distance: float = 45.0

    # Force weights
    w_sep: float = 2.8
    w_coh: float = 0.55
    w_ali: float = 0.60
    w_dog: float = 3.2
    w_wall: float = 2.0
    w_noise: float = 0.12
    w_inertia: float = 0.992
    w_graze: float = 1.45

    # Grazing behaviour
    wander_change_interval_min: float = 0.8
    wander_change_interval_max: float = 2.2
    graze_jitter: float = 0.28

    # Speed regulation
    target_speed_calm: float = 78.0
    target_speed_alert: float = 150.0
    speed_relax_gain: float = 2.2

    # Sheep speed limits
    max_speed_calm: float = 100.0
    max_speed_alert: float = 170.0

    # Global force cap
    max_force: float = 320.0

    # Heterogeneity ranges
    fear_strength_min: float = 0.85
    fear_strength_max: float = 1.20

    max_speed_mult_min: float = 0.90
    max_speed_mult_max: float = 1.12

    graze_bias_min: float = 0.80
    graze_bias_max: float = 1.25

    reaction_delay_min_frames: int = 1
    reaction_delay_max_frames: int = 4

    # Reaction smoothing
    force_memory: float = 0.70  # sheep.force = 0.7*prev + 0.3*new

    # Dog dynamics
    dog_max_speed: float = 380.0
    dog_accel: float = 900.0
    dog_friction: float = 0.94
    dog_sprint_multiplier: float = 1.8
    dog_sheep_push_strength: float = 16.0

    # Autonomous dog
    dog_mode_default: str = "manual"   # "manual" or "auto"
    collect_distance_factor: float = 1.5
    drive_distance_factor: float = 1.2
    collect_spread_threshold_factor: float = 1.8

    # Spawn
    sheep_spawn_cx: float = 350.0
    sheep_spawn_cy: float = 450.0
    sheep_spawn_std: float = 55.0
    sheep_spawn_speed_min: float = 45.0
    sheep_spawn_speed_max: float = 90.0

    # Rendering
    font_size: int = 22

    # Logging
    enable_logging: bool = True
    log_every_n_frames: int = 1
    output_csv: str = "shepherding_log.csv"
    output_sheep_csv: str = "sheep_positions_log.csv"

    # Seeds
    python_seed: int = 7
    numpy_seed: int = 7