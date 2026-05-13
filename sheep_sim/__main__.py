from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path

from sheep_sim.core.config import FeatureConfig
from sheep_sim.scenarios import build_scenario_config
from sheep_sim.simulation import SheepSimulation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sheep movement simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--scenario",
        choices=["abundant", "scarce", "uniform_low", "uniform_high", "radial_increase", "ring", "corridors"],
        default="abundant",
    )
    p.add_argument("--steps",            type=int,   default=2500)
    p.add_argument("--landscape-seed",   type=int,   default=7)
    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--render",           choices=["none", "live", "final"], default="final")
    p.add_argument("--output-dir",       type=str,   default=None)
    p.add_argument("--seconds-per-step", type=float, default=30.0)
    p.add_argument("--render-fps",       type=int,   default=25)
    p.add_argument("--render-every",     type=int,   default=2)

    p.add_argument("--disable-foraging",    action="store_true", help="Disable sensory gradient, prospecting, and MVT patch departure (correlated random walk base)")
    p.add_argument("--disable-social",      action="store_true", help="Disable Reynolds social force, REGROUPING, departure contagion")
    p.add_argument("--disable-circadian",   action="store_true", help="Disable bimodal circadian and RESTING transitions (active_factor=1.0)")
    p.add_argument("--disable-memory",      action="store_true", help="Disable spatial memory map (§2.7)")
    p.add_argument("--disable-personality", action="store_true", help="Disable per-sheep trait variation, OU drift, Markov bias")
    p.add_argument("--disable-terrain",     action="store_true", help="Disable terrain gradient force")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    cfg = build_scenario_config(
        args.scenario,
        steps=args.steps,
        seed=args.seed,
        landscape_seed=args.landscape_seed,
    )
    cfg = cfg.with_time(real_seconds_per_step=args.seconds_per_step, render_fps=args.render_fps)
    cfg = cfg.with_output(render_every_n_steps=max(1, args.render_every))

    features = FeatureConfig(
        foraging    = not args.disable_foraging,
        social      = not args.disable_social,
        circadian   = not args.disable_circadian,
        memory      = not args.disable_memory,
        personality = not args.disable_personality,
        terrain     = not args.disable_terrain,
    )
    cfg = replace(cfg, features=features)

    active = [name for name, on in asdict(features).items() if on]
    print(f"  Features enabled : {', '.join(active) if active else 'none (base layer only)'}")

    output_dir = Path(args.output_dir) if args.output_dir else Path("outputs") / args.scenario

    sim    = SheepSimulation(cfg)
    result = sim.run(render=args.render, output_dir=output_dir)

    df    = result.metrics.group_dataframe()
    final = df.iloc[-1]
    speedup = cfg.time.real_seconds_per_step * cfg.time.render_fps / max(cfg.output.render_every_n_steps, 1)

    print("\n" + "=" * 60)
    print(f"  Simulation complete — {args.scenario}")
    print("=" * 60)
    print(f"  Steps            : {args.steps}")
    print(f"  Landscape seed   : {args.landscape_seed}")
    print(f"  Behaviour seed   : {args.seed}")
    print(f"  Output dir       : {output_dir}")
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