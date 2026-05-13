from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCENARIOS = ["uniform_low", "uniform_high", "radial_increase", "ring", "abundant", "scarce"]
N_RUNS = 3
STEPS = 2500
LANDSCAPE_SEED = 7

def main():
    for scenario in SCENARIOS:
        print(f"\n{'='*50}")
        print(f"Scenario: {scenario}")
        print(f"{'='*50}")
        for seed in range(1, N_RUNS + 1):
            output_dir = f"outputs/{scenario}/seed_{seed}"
            print(f"  seed={seed} ... ", end="", flush=True)
            result = subprocess.run(
                [
                    sys.executable, "-m", "sheep_sim",
                    "--scenario", scenario,
                    "--steps", str(STEPS),
                    "--landscape-seed", str(LANDSCAPE_SEED),
                    "--seed", str(seed),
                    "--render", "final",
                    "--output-dir", output_dir,
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                png = Path(output_dir) / "final_frame.png"
                print(f"done  ->  {png}")
            else:
                print(f"FAILED")
                print(result.stderr[-400:])

    print(f"\n{'='*50}")
    print("All runs complete. Output structure:")
    for scenario in SCENARIOS:
        for seed in range(1, N_RUNS + 1):
            png = Path(f"outputs/{scenario}/seed_{seed}/final_frame.png")
            status = "OK" if png.exists() else "MISSING"
            print(f"  [{status}]  {png}")

if __name__ == "__main__":
    main()
