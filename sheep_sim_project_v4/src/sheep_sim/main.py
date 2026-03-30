from __future__ import annotations

import argparse
from pathlib import Path

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sheep movement simulation")
    parser.add_argument("--scenario", choices=["abundant", "scarce"], default="abundant")
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render", choices=["none", "live", "final"], default="final")
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--seconds-per-step", type=float, default=30.0, help="How much real-world time one simulation step represents.")
    parser.add_argument("--render-fps", type=int, default=25, help="Live animation frame rate.")
    parser.add_argument("--render-every", type=int, default=4, help="Draw every N simulation steps in live mode.")
    return parser.parse_args()



def main() -> None:
    args = parse_args()
    cfg = build_scenario_config(args.scenario, args.steps, args.seed)
    cfg.time.real_seconds_per_step = args.seconds_per_step
    cfg.time.render_fps = args.render_fps
    cfg.output.render_every_n_steps = max(1, args.render_every)

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / args.scenario
    sim = SheepSimulation(cfg)
    result = sim.run(render=args.render, output_dir=output_dir)

    group_df = result.metrics.group_dataframe()
    final = group_df.iloc[-1]
    speedup = cfg.time.real_seconds_per_step * cfg.time.render_fps / cfg.output.render_every_n_steps
    print("Simulation complete")
    print(f"Scenario: {args.scenario}")
    print(f"Steps: {args.steps}")
    print(f"Output directory: {output_dir}")
    print(f"Time scale: 1 step = {cfg.time.real_seconds_per_step:.1f} real seconds")
    print(f"Approx live-animation speedup: {speedup:.1f}x real time")
    print(
        "Final metrics: "
        f"spread={final['spread']:.2f}, "
        f"polarization={final['polarization']:.2f}, "
        f"mean_speed={final['mean_speed']:.2f}, "
        f"mean_food_intake={final['mean_food_intake']:.4f}, "
        f"field_health_pct={final['field_health_pct']:.1f}, "
        f"field_biomass_pct={final['field_biomass_pct']:.1f}, "
        f"degraded_area_pct={final['degraded_area_pct']:.1f}, "
        f"clusters={int(final['number_of_clusters'])}"
    )


if __name__ == "__main__":
    main()
