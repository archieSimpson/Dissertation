# An Agent-Based Model Implementing and Integrating Behavioural Mechanisms to Predict Emergent Grazing Patterns of Autonomous Sheep

**BSc Computer Science Synoptic Project (COMP3932), University of Leeds, 2025/26**
Author: Archibald Simpson


## Demo
| Abundant Step: 0 | Abundant Step: 1350 |
|--------|--------|
| <img width="350" src="https://github.com/user-attachments/assets/e69a375b-8b6d-45c7-b5db-15f69af8164e" /> | <img width="350" src="https://github.com/user-attachments/assets/0f4d0747-6b5c-4067-ac1a-9a26eb3134ca" /> |

**Full scenario videos:** [abundant](https://youtu.be/iY-mb5cHtKU) · [scarce](https://youtu.be/q5w5i8qJjNc) · [corridors](https://youtu.be/EkznMudK62Q) · [radial increase](https://youtu.be/tBTc0IdcpRY) · [uniform high](https://youtu.be/4I8t3kYBJlc) · [uniform low](https://youtu.be/aLfXAYrVxqk) · [base — no mechanisms](https://youtu.be/Q5GCKR7Me0w)

**Per-mechanism verification:** [foraging](https://youtu.be/uoeiTVrs3Iw) · [social](https://youtu.be/Ho0PSxDwLZU) · [circadian](https://youtu.be/hkgaW0vCnR8)


## Summary

Sheep movement in free-ranging environments is shaped by the interactions of multiple biological behaviours, yet existing agent-based models integrate at most two or three. Uneven flock distribution caused by these interactions leads to overgrazing in some areas and underutilisation in others. This project addresses that gap by developing an agent-based model that integrates six biological behaviours into a single computational framework to predict grazing patterns and analyse emergent outcomes.

The behavioural mechanisms implemented and integrated are social cohesion, foraging, circadian activity, memory maps, personality, and terrain. Each mechanism was implemented as an independently activatable module and verified in isolation against a correlated random-walk baseline before integration. The integrated simulation then models forty sheep over a sixteen-hour active day in two distinct environments: an abundant environment with plentiful grass, and a scarce environment with very little grass. The behaviours are governed by a five-state finite-state machine spanning GRAZING, WALKING, TRAVELLING, REGROUPING, and RESTING.

The model was evaluated against flock-pattern behaviours and four GPS-telemetry benchmarks drawn from the existing literature: nearest-neighbour distance, resting fraction, daily distance, and velocity autocorrelation. Heatmaps were generated from the convergence process to depict the flock's movement and grazing patterns over a given day, alongside a state-composition analysis across the two landscapes. Four additional artificial landscapes were used to probe the internal coherence of the integrated mechanisms and observe emergent outcomes under a controlled environment.




## Project Structure

```
random-walk-simulation/
├── pyproject.toml
├── README.md
├── .gitignore
│
├── sheep_sim/
│   ├── __main__.py
│   ├── simulation.py
│   ├── scenarios.py
│   │
│   ├── core/
│   ├── behaviour/
│   ├── environment/
│   ├── metrics/
│   ├── rendering/
│   └── io/
│
├── scripts/
    ├── runs/
    ├── analysis/
    ├── plots/
    ├── convergence_study/
    ├── diagnostics/
```

---

## Installation

Requires Python 3.11.

```bash
git clone https://github.com/<user>/sheep-sim.git
cd sheep-sim
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

**Reference seeds used throughout the dissertation:** 

```bash
python -m sheep_sim --scenario abundant --seed 42 --landscape-seed 7 --render final
```

Watch live instead of writing a PNG: `--render live`. See all options with `python -m sheep_sim --help`.

### Scenarios

```bash
python -m sheep_sim --scenario abundant
python -m sheep_sim --scenario scarce
python -m sheep_sim --scenario uniform_high
python -m sheep_sim --scenario uniform_low
python -m sheep_sim --scenario radial_increase
python -m sheep_sim --scenario corridors
```

### Varying seeds

Behaviour and landscape are seeded independently — useful for ensemble runs and the convergence procedure in §3.5.

```bash
# Same landscape, different flock
python -m sheep_sim --scenario abundant --landscape-seed 7 --seed 42
python -m sheep_sim --scenario abundant --landscape-seed 7 --seed 43

# Same flock, different landscape
python -m sheep_sim --scenario scarce --seed 42 --landscape-seed 7
python -m sheep_sim --scenario scarce --seed 42 --landscape-seed 23
```

The three landscape seeds used for the GPS-benchmark validation tables are `7`, `23`, `41`.

### Activatable toggles

Any of the six mechanisms can be disabled to reproduce the verification matrix in §3.3:

```bash
# Full model minus social cohesion
python -m sheep_sim --scenario abundant --disable-social

# Correlated-random-walk baseline (all mechanisms off)
python -m sheep_sim --scenario abundant \
  --disable-foraging --disable-social --disable-circadian \
  --disable-memory --disable-personality --disable-terrain
```

Flags: `--disable-foraging`, `--disable-social`, `--disable-circadian`, `--disable-memory`, `--disable-personality`, `--disable-terrain`.

---

## Movement Results: 
| Abundant Movement Heatmap | Scarce Movement Heatmap |
|--------|--------|
| <img width="1909" height="1343" alt="heatmap_abundant_n9_polished" src="https://github.com/user-attachments/assets/df196bd0-1177-4976-a44f-8a0064ba1853" /> | <img width="1909" height="1343" alt="heatmap_scarce_n9_polished" src="https://github.com/user-attachments/assets/7748b479-c686-4487-9c78-65f022baf5eb" /> |

Landscape seed: 7 Behavioural seed: 42
