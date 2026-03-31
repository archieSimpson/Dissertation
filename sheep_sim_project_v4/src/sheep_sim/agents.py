from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from sheep_sim.utils import heading_vector


class BehaviourState(str, Enum):
    GRAZING = "grazing"
    WALKING = "walking"
    TRAVELLING = "travelling"
    REGROUPING = "regrouping"
    RESTING = "resting"


@dataclass(slots=True)
class SheepAgent:
    sheep_id: int
    position: np.ndarray
    velocity: np.ndarray
    heading: float
    state: BehaviourState
    energy: float = 0.85
    memory_map: np.ndarray | None = None
    last_food_intake: float = 0.0
    cumulative_food: float = 0.0
    state_age: int = 0
    path_length: float = 0.0
    separation_steps: int = 0
    regroup_cooldown: int = 0
    site_fidelity: float = 1.0
    sociability: float = 1.0
    boldness: float = 1.0
    movement_vigor: float = 1.0
    turning_bias: float = 1.0
    patch_leave_bias: float = 1.0
    preferred_spacing: float = 1.0
    home_target: np.ndarray | None = None
    explore_heading: float = 0.0
    explore_timer: int = 0

    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def heading_vector(self) -> np.ndarray:
        return heading_vector(self.heading)
