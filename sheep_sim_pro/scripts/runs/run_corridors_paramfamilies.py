"""Run the corridors landscape under both parameter families (abundant, scarce)
for three seeds each, saving final-frame PNGs.

The corridors NDVI geometry (food.py:161) is held fixed; only the
food/transitions/flock/field parameter overrides differ between the two halves.
The FSM in states.py treats corridors as rich-path regardless, so the scarce
half is a hybrid (rich FSM, scarce thresholds). Spawn geometry (Gaussian
sigma=5m at field centre, states.py:50) puts all 40 agents inside the 22 m
central island.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path


from sheep_sim.scenarios import (
    _apply_abundant_params,
    _apply_scarce_params,
    build_scenario_config,
)
from sheep_sim.simulation import SheepSimulation

SEEDS = [42, 43, 44]
LANDSCAPE_SEED = 7
STEPS = 1920
SCENARIO = "corridors"


def run_one(family: str, seed: int) -> Path:
    cfg = build_scenario_config(SCENARIO, steps=STEPS, seed=seed, landscape_seed=LANDSCAPE_SEED)



    if family == "scarce":


        cfg = _apply_scarce_params(cfg)
    elif family == "abundant":

        cfg = _apply_abundant_params(cfg)
    else:
        raise ValueError(family)

    out_dir = Path("outputs") / f"corridors_{family}_seed{seed}_L{LANDSCAPE_SEED}"
    sim = SheepSimulation(cfg)
    sim.run(render="final", output_dir=out_dir)
    return out_dir / "final_frame.png"


def main() -> None:
    print("=" * 60)
    print(f"Corridors x {{abundant, scarce}} x seeds {SEEDS}")
    print(f"landscape_seed={LANDSCAPE_SEED}  steps={STEPS}")
    print("=" * 60)
    for family in ("abundant", "scarce"):
        for seed in SEEDS:
            print(f"\n>>> family={family}  seed={seed}")
            png = run_one(family, seed)
            print(f"    saved -> {png}")


if __name__ == "__main__":
    main()
