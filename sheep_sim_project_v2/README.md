# Sheep Movement Simulation

A research-oriented agent-based simulation of sheep movement under two pasture conditions:

- **Abundant food**: area-restricted grazing, higher patch residence, stronger routine use of familiar space.
- **Scarce food**: more exploratory travel, longer steps, stronger reliance on food-memory and local sensory cues.

The model is designed to reflect the behavioural themes in the dissertation materials:

- movement as **state switching** between grazing, travelling, regrouping, and resting
- **social coupling** through attraction, alignment, and short-range repulsion
- **resource-rich vs resource-scarce** search differences
- **circadian modulation** of activity
- **boundary and topographic effects**
- an upgraded **dashboard-style live frontend** with a state legend and time-scale display

This is a simulation scaffold, not a calibrated digital twin. It is designed to be strong enough for dissertation development and later biological calibration.

## File structure

```text
sheep_sim_project/
├── README.md
├── requirements.txt
├── pyproject.toml
└── src/
    └── sheep_sim/
        ├── __init__.py
        ├── __main__.py
        ├── agents.py
        ├── behaviour.py
        ├── config.py
        ├── environment.py
        ├── food.py
        ├── main.py
        ├── metrics.py
        ├── rendering.py
        ├── scenarios.py
        ├── simulation.py
        └── utils.py
```

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
PYTHONPATH=src python -m sheep_sim --scenario abundant --steps 2500 --render live
```

Run the scarce-food scenario:

```bash
PYTHONPATH=src python -m sheep_sim --scenario scarce --steps 2500 --render live
```

## New live dashboard frontend

The live renderer now includes:

- a **field view** with food-density heatmap, sheep positions, headings, centroid, and spread ring
- a **state legend** explaining exactly what each state means
- a **runtime summary panel** with mean speed, spread, polarization, and per-state counts
- a **simulated clock**
- a visible **time-scale readout** showing how fast the animation is relative to real life

### State colours

- **Green** = Grazing
- **Blue** = Travelling
- **Orange** = Regrouping
- **Grey** = Resting

## Time scale

By default, the simulation is **not 1:1 with real time**.

Default mapping:

- `1 simulation step = 30 real-world seconds`
- live renderer defaults to `25 FPS`
- if drawing every 4 simulation steps, the animation shows about:

```text
(30 seconds × 25 frames) / 4 = 187.5x real time
```

So by default, the live animation is roughly **188x faster than real life**.

### Change the time mapping

You can make the simulation slower or faster by changing the time arguments:

```bash
PYTHONPATH=src python -m sheep_sim \
  --scenario abundant \
  --steps 2500 \
  --render live \
  --seconds-per-step 10 \
  --render-fps 20 \
  --render-every 2
```

That gives a different real-time relationship between the simulation and the animation.

## What the animation should show

### Abundant field

You should see:

- slower, more local movement
- stronger patch residence
- repeated grazing/rest cycles
- looser but still coherent flock structure
- fewer long straight exploratory runs

### Scarce field

You should see:

- longer travel bouts
- more patch switching
- more visible exploration and regrouping
- stronger contrast between travelling and grazing states

## Outputs

The simulation writes:

- `metrics.csv`: group-level metrics through time
- `positions.csv`: per-sheep trajectories through time
- `final_frame.png`: final snapshot with legend

## Useful commands

Live dashboard:

```bash
PYTHONPATH=src python -m sheep_sim --scenario abundant --steps 3000 --render live
```

Live dashboard with slower, more interpretable timing:

```bash
PYTHONPATH=src python -m sheep_sim --scenario scarce --steps 3000 --render live --seconds-per-step 15 --render-fps 20 --render-every 2
```

Final snapshot only:

```bash
PYTHONPATH=src python -m sheep_sim --scenario scarce --steps 4000 --render final --output-dir outputs/scarce
```

## Dissertation extensions you can add next

- sheepdog / predator module
- virtual fencing
- video export
- terrain raster import
- NDVI import
- calibration against empirical GPS / accelerometer data
