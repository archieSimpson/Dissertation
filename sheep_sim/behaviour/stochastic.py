from __future__ import annotations

import numpy as np

from sheep_sim.core.agents import BehaviourState, SheepAgent

OU_THETA: float = 0.05

OU_SIGMA: float = 0.018

OU_CLIP: float = 0.25

MARKOV_LO: float = 0.80
MARKOV_HI: float = 1.20

_STATES: list[BehaviourState] = list(BehaviourState)
_STATE_IDX: dict[BehaviourState, int] = {s: i for i, s in enumerate(_STATES)}

def initialise_stochastic_traits(
    sheep: SheepAgent,
    rng: np.random.Generator,
    ou_sigma_init: float = 0.05,
    personality: bool = True,
) -> None:
    if personality:
        sheep.ou_attraction = float(rng.normal(0.0, ou_sigma_init))
        sheep.ou_alignment  = float(rng.normal(0.0, ou_sigma_init))
        sheep.ou_repulsion  = float(rng.normal(0.0, ou_sigma_init))

        mat = rng.uniform(MARKOV_LO, MARKOV_HI, size=(5, 5)).astype(float)
        np.fill_diagonal(mat, 1.0)
        sheep.transition_matrix = mat
    else:
        sheep.ou_attraction = 0.0
        sheep.ou_alignment  = 0.0
        sheep.ou_repulsion  = 0.0
        sheep.transition_matrix = None

def step_ou(sheep: SheepAgent, rng: np.random.Generator, dt: float = 1.0) -> None:
    sqrt_dt = float(np.sqrt(dt))
    dw = rng.standard_normal(3)

    sheep.ou_attraction = float(np.clip(
        sheep.ou_attraction
        + OU_THETA * (0.0 - sheep.ou_attraction) * dt
        + OU_SIGMA * sqrt_dt * dw[0],
        -OU_CLIP, OU_CLIP,
    ))
    sheep.ou_alignment = float(np.clip(
        sheep.ou_alignment
        + OU_THETA * (0.0 - sheep.ou_alignment) * dt
        + OU_SIGMA * sqrt_dt * dw[1],
        -OU_CLIP, OU_CLIP,
    ))
    sheep.ou_repulsion = float(np.clip(
        sheep.ou_repulsion
        + OU_THETA * (0.0 - sheep.ou_repulsion) * dt
        + OU_SIGMA * sqrt_dt * dw[2],
        -OU_CLIP, OU_CLIP,
    ))

def get_effective_weights(
    sheep: SheepAgent,
    base_attraction: float,
    base_alignment: float,
    base_repulsion: float,
) -> tuple[float, float, float]:
    return (
        max(base_attraction * 0.10, base_attraction + sheep.ou_attraction),
        max(base_alignment  * 0.10, base_alignment  + sheep.ou_alignment),
        max(base_repulsion  * 0.10, base_repulsion  + sheep.ou_repulsion),
    )

def markov_bias_for(sheep: SheepAgent, to_state: BehaviourState) -> float:
    if sheep.transition_matrix is None:
        return 1.0
    from_idx = _STATE_IDX[sheep.state]
    to_idx   = _STATE_IDX[to_state]
    return float(sheep.transition_matrix[from_idx, to_idx])
