from dataclasses import dataclass, field
import numpy as np


@dataclass
class Sheep:
    idx: int
    pos: np.ndarray
    vel: np.ndarray
    alert: bool = False
    state: str = "calm"

    wander_dir: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    wander_timer: float = 0.0

    # Heterogeneous traits
    fear_strength: float = 1.0
    max_speed_multiplier: float = 1.0
    reaction_delay_frames: int = 1
    graze_bias: float = 1.0

    # Reaction-delay state
    previous_force: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float64))
    delay_counter: int = 0


@dataclass
class Dog:
    pos: np.ndarray
    vel: np.ndarray
    mode: str = "manual"         # manual / auto
    submode: str = "drive"       # collect / drive


@dataclass
class Goal:
    x: float
    y: float
    w: float
    h: float

    def contains(self, pos: np.ndarray) -> bool:
        return self.x <= pos[0] <= self.x + self.w and self.y <= pos[1] <= self.y + self.h


@dataclass
class Obstacle:
    points: list[np.ndarray] = field(default_factory=list)