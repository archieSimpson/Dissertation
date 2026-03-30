# Sheep Movement Simulation v3

A modular sheep movement simulation with two pasture regimes:

- `abundant`: resource-rich field with stronger patch residence, grazing/rest cycles, and loose-but-cohesive flock structure.
- `scarce`: lower-productivity field with longer travel, stronger search behaviour, and patch switching.

## Key upgrades in v3

- NDVI-style heterogeneous pasture surface
- Grass biomass and carrying capacity separated from field health
- Grazing-pressure memory that persists after heavy grazing
- More realistic grass regrowth based on simulated days rather than per-frame cosmetic refill
- `field_health.csv` output for pasture condition over time
- UI panel for grass field health, remaining biomass, degraded area, and grazing pressure
- Refined movement logic to better match the dissertation prompt:
  - abundant pasture -> shorter, more tortuous movement and longer patch residence
  - scarce pasture -> longer, straighter travel with stronger search bias and weaker residence

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m sheep_sim --scenario abundant --steps 3000 --render live
```

Scarce pasture:

```bash
PYTHONPATH=src python -m sheep_sim --scenario scarce --steps 3000 --render live
```

## Outputs

Each run writes:

- `metrics.csv` -> flock-level movement metrics
- `positions.csv` -> per-sheep trajectories and states
- `field_health.csv` -> field condition over time
- `final_frame.png` -> last rendered frame

## Time scaling

Default:

- `1 simulation step = 30 real-world seconds`
- live animation is accelerated for readability
- grass regrowth uses simulated days internally (`seconds_per_step / 86400`)

## Notes on regrowth realism

The grass model now regrows on a day-based timescale rather than quickly refilling every few frames.
It is still an ecological approximation for simulation purposes, not a site-calibrated agronomy model.
