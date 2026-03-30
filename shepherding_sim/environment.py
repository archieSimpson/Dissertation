from agents import Goal
from config import SimulationConfig


class Environment:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.goal = Goal(cfg.goal_x, cfg.goal_y, cfg.goal_w, cfg.goal_h)

    def sheep_in_goal(self, sheep_list) -> int:
        return sum(1 for s in sheep_list if self.goal.contains(s.pos))

    def all_sheep_in_goal(self, sheep_list) -> bool:
        return self.sheep_in_goal(sheep_list) == len(sheep_list)