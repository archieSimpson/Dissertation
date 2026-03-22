import matplotlib
matplotlib.use("QtAgg")

from app.config import (
    NUM_DOTS,
    STEP_SIZE,
    TOTAL_STEPS,
    PLOT_LIMIT,
    DEFAULT_FOOD_COUNT,
    MIN_FOOD_COUNT,
    MAX_FOOD_COUNT,
    DEFAULT_FOOD_CLUSTER_SPREAD,
    MIN_FOOD_CLUSTER_SPREAD,
    FOOD_RADIUS,
    AGENT_RADIUS,
    FPS,
    EAT_COOLDOWN_SECONDS,
    BASE_DIRECTION_RESET_STEPS,
    MIN_DIRECTION_CHANGE_DEGREES,
    DIRECTION_NOISE_SCALE,
    LEVY_FLIGHT_PROBABILITY,
    LEVY_FLIGHT_MULTIPLIER_MIN,
    LEVY_FLIGHT_MULTIPLIER_MAX,
    MEMORY_SIZE,
    MEMORY_SCORE_DECAY,
    MEMORY_DISTANCE_MERGE,
    MEMORY_REVISIT_STRENGTH,
    INITIAL_PATCH_RADIUS,
    MIN_PATCH_RADIUS,
    MAX_PATCH_RADIUS,
    PATCH_TIMEOUT_STEPS,
    PATCH_TIMEOUT_GROWTH,
    PATCH_RADIUS_SHRINK_FACTOR,
    PATCH_RADIUS_GROWTH_FACTOR,
    PATCH_CONFIDENCE_GAIN,
    PATCH_CONFIDENCE_DECAY,
    FORAGE_PULL_STRENGTH,
    FORAGE_TANGENT_STRENGTH,
    FORAGE_DIRECTION_BLEND,
    LOCAL_WANDER_SCALE,
    FORAGE_SPEED_REDUCTION,
    SCENT_RADIUS,
    SCENT_STRENGTH,
    INSPECTION_SPEED_REDUCTION,
    SOCIAL_SIGNAL_RADIUS,
    SOCIAL_SIGNAL_STRENGTH,
    SOCIAL_SIGNAL_LIFETIME,
    USE_BOUNDARY_REFLECTION,
)
from app.simulation.random_walk import RandomWalkSimulation
from app.ui.setup_gui import run_setup_gui
from app.visualisation.animation import animate_live
from app.visualisation.plots import plot_bell_curve, plot_gaussian_xy


def main():
    food_patches = run_setup_gui(
        plot_limit=PLOT_LIMIT,
        default_food_count=DEFAULT_FOOD_COUNT,
        min_food_count=MIN_FOOD_COUNT,
        max_food_count=MAX_FOOD_COUNT,
        default_food_cluster_spread=DEFAULT_FOOD_CLUSTER_SPREAD,
        min_food_cluster_spread=MIN_FOOD_CLUSTER_SPREAD,
    )

    simulation = RandomWalkSimulation(
        num_dots=NUM_DOTS,
        step_size=STEP_SIZE,
        food_patches=food_patches,
        food_radius=FOOD_RADIUS,
        agent_radius=AGENT_RADIUS,
        fps=FPS,
        eat_cooldown_seconds=EAT_COOLDOWN_SECONDS,
        base_direction_reset_steps=BASE_DIRECTION_RESET_STEPS,
        min_direction_change_degrees=MIN_DIRECTION_CHANGE_DEGREES,
        direction_noise_scale=DIRECTION_NOISE_SCALE,
        levy_flight_probability=LEVY_FLIGHT_PROBABILITY,
        levy_flight_multiplier_min=LEVY_FLIGHT_MULTIPLIER_MIN,
        levy_flight_multiplier_max=LEVY_FLIGHT_MULTIPLIER_MAX,
        memory_size=MEMORY_SIZE,
        memory_score_decay=MEMORY_SCORE_DECAY,
        memory_distance_merge=MEMORY_DISTANCE_MERGE,
        memory_revisit_strength=MEMORY_REVISIT_STRENGTH,
        initial_patch_radius=INITIAL_PATCH_RADIUS,
        min_patch_radius=MIN_PATCH_RADIUS,
        max_patch_radius=MAX_PATCH_RADIUS,
        patch_timeout_steps=PATCH_TIMEOUT_STEPS,
        patch_timeout_growth=PATCH_TIMEOUT_GROWTH,
        patch_radius_shrink_factor=PATCH_RADIUS_SHRINK_FACTOR,
        patch_radius_growth_factor=PATCH_RADIUS_GROWTH_FACTOR,
        patch_confidence_gain=PATCH_CONFIDENCE_GAIN,
        patch_confidence_decay=PATCH_CONFIDENCE_DECAY,
        forage_pull_strength=FORAGE_PULL_STRENGTH,
        forage_tangent_strength=FORAGE_TANGENT_STRENGTH,
        forage_direction_blend=FORAGE_DIRECTION_BLEND,
        local_wander_scale=LOCAL_WANDER_SCALE,
        forage_speed_reduction=FORAGE_SPEED_REDUCTION,
        scent_radius=SCENT_RADIUS,
        scent_strength=SCENT_STRENGTH,
        inspection_speed_reduction=INSPECTION_SPEED_REDUCTION,
        social_signal_radius=SOCIAL_SIGNAL_RADIUS,
        social_signal_strength=SOCIAL_SIGNAL_STRENGTH,
        social_signal_lifetime=SOCIAL_SIGNAL_LIFETIME,
        use_boundary_reflection=USE_BOUNDARY_REFLECTION,
        plot_limit=PLOT_LIMIT,
    )

    animate_live(
        simulation=simulation,
        total_steps=TOTAL_STEPS,
        limit=PLOT_LIMIT,
        interval=1 / FPS,
    )

    final_positions = simulation.history[-1]
    plot_bell_curve(final_positions)
    plot_gaussian_xy(final_positions)


if __name__ == "__main__":
    main()