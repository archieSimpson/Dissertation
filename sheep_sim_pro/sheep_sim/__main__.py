from __future__ import annotations

import argparse
from pathlib import Path

from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sheep movement simulation — abundant and scarce field scenarios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenario", choices=["abundant", "scarce"], default="abundant",
        help="Field resource scenario.",
    )
    p.add_argument(
        "--steps", type=int, default=2500,
        help="Number of simulation steps.",
    )
    p.add_argument(
        "--landscape-seed", type=int, default=7,
        help="RNG seed for NDVI/terrain — fix this across runs to hold the environment constant.",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for sheep behaviour (personalities, OU drift, Markov matrices) — vary across runs.",
    )
    p.add_argument(
        "--render", choices=["none", "live", "final"], default="final",
        help="Rendering mode.",
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for CSV and PNG files.",
    )
    p.add_argument(
        "--seconds-per-step", type=float, default=30.0,
        help="Real-world seconds represented by each simulation step.",
    )
    p.add_argument(
        "--render-fps", type=int, default=25,
        help="Target frame rate for live animation.",
    )
    p.add_argument(
        "--render-every", type=int, default=4,
        help="Draw every N steps in live mode.",
    )
    return p.parse_args()

def main() -> None:
    args = parse_args()

    cfg = build_scenario_config(
        args.scenario,
        steps=args.steps,
        seed=args.seed,
        landscape_seed=args.landscape_seed,
    )
    cfg = cfg.with_time(
        real_seconds_per_step=args.seconds_per_step,
        render_fps=args.render_fps,
    )
    cfg = cfg.with_output(render_every_n_steps=max(1, args.render_every))

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path("outputs") / args.scenario
    )

    sim    = SheepSimulation(cfg)
    result = sim.run(render=args.render, output_dir=output_dir)

    df    = result.metrics.group_dataframe()
    final = df.iloc[-1]
    speedup = (
        cfg.time.real_seconds_per_step
        * cfg.time.render_fps
        / max(cfg.output.render_every_n_steps, 1)
    )

    print("\n" + "=" * 60)
    print(f"  Simulation complete — {args.scenario} scenario")
    print("=" * 60)
    print(f"  Steps            : {args.steps}")
    print(f"  Landscape seed   : {args.landscape_seed}  (environment fixed)")
    print(f"  Behaviour seed   : {args.seed}  (varies across runs)")
    print(f"  Output dir       : {output_dir}")
    print(f"  Time scale       : 1 step = {cfg.time.real_seconds_per_step:.1f} real seconds")
    print(f"  Animation speed  : ~{speedup:.1f}x real time")
    print("-" * 60)
    print(f"  Final spread         : {final['spread']:.2f}")
    print(f"  Final polarisation   : {final['polarization']:.2f}")
    print(f"  Mean speed           : {final['mean_speed']:.3f}")
    print(f"  Mean food intake     : {final['mean_food_intake']:.5f}")
    print(f"  Field health         : {final['field_health_pct']:.1f}%")
    print(f"  Remaining biomass    : {final['field_biomass_pct']:.1f}%")
    print(f"  Degraded area        : {final['degraded_area_pct']:.1f}%")
    print(f"  Number of clusters   : {int(final['number_of_clusters'])}")
    print("=" * 60)

if __name__ == "__main__":
    main()
