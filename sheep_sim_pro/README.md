# Sheep Movement Simulation

An agent-based model of 40 free-ranging sheep on a $220 \times 140$ m field, integrating foraging, social cohesion, circadian rhythm, spatial memory, personality, and terrain.

**BSc Computer Science Synoptic Project (COMP3932) — University of Leeds, 2025/26**
Author: Archibald Simpson

---

## Demo

| Scenario | Preview |
|---|---|
| Abundant | `![abundant](docs/media/abundant.gif)` |
| Scarce | `![scarce](docs/media/scarce.gif)` |
| Corridors | `![corridors](docs/media/corridors.gif)` |

**Full scenario videos:** [abundant](https://youtu.be/iY-mb5cHtKU) · [scarce](https://youtu.be/q5w5i8qJjNc) · [corridors](https://youtu.be/EkznMudK62Q) · [radial increase](https://youtu.be/tBTc0IdcpRY) · [uniform high](https://youtu.be/4I8t3kYBJlc) · [uniform low](https://youtu.be/aLfXAYrVxqk) · [base — no mechanisms](https://youtu.be/Q5GCKR7Me0w)

**Per-mechanism verification:** [foraging](https://youtu.be/uoeiTVrs3Iw) · [social](https://youtu.be/Ho0PSxDwLZU) · [circadian](https://youtu.be/hkgaW0vCnR8)

---

## Project Structure

```
sheep_sim_pro/
├── pyproject.toml
├── README.md
│
├── sheep_sim/                    # Simulation package (importable, CLI-runnable)
│   ├── __main__.py               #   CLI entry point — `python -m sheep_sim`
│   ├── simulation.py             #   Step-loop coordinator
│   ├── scenarios.py              #   7 scenario presets
│   │
│   ├── core/                     #   Config, agent dataclass, utilities
│   ├── behaviour/                #   FSM, movement, stochastic personality
│   ├── environment/              #   Terrain + NDVI landscape
│   ├── metrics/                  #   Per-step recorder + post-hoc analysis
│   ├── rendering/                #   Live animation + final-frame PNG
│   └── io/                       #   Run packaging (config.json + CSVs)
│
└── scripts/                      # Reproducible drivers — not imported by the package
    ├── runs/                     #   Batch simulation drivers
    ├── analysis/                 #   Post-hoc metric extraction
    ├── plots/                    #   Figure-generating scripts (one per figure)
    ├── convergence_study/        #   Spatial-convergence procedure (§3.5)
    ├── diagnostics/               
    └── tools/                     
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

**Canonical run** (reference seeds used throughout the dissertation):

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
python -m sheep_sim --scenario ring
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

### Ablation toggles

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

## Outputs

Each run writes to `outputs/<scenario>/`:

| File | Contents |
|---|---|
| `metrics.csv` | Per-step group metrics — spread, polarisation, NND, state counts, field health |
| `positions.csv` | Per-step per-agent state — position, velocity, state, intake, personality traits |
| `field_health.csv` | Per-step landscape — biomass, health, grazing pressure, NDVI |
| `final_frame.png` | Final-frame plot (with `--render final` or `live`) |
