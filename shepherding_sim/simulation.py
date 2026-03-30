import random
import math
import numpy as np

from agents import Dog, Sheep
from behaviours import sheep_force
from config import SimulationConfig
from control import autonomous_dog_acceleration, manual_dog_acceleration
from environment import Environment
from flock import centroid
from logging_io import CSVLogger, SheepStateLogger
from math2d import vec
from metrics import compute_metrics
from physics import (
    integrate_dog,
    integrate_sheep,
    clamp_to_world,
    resolve_sheep_collisions,
    resolve_dog_sheep_collisions,
)


class ShepherdingSimulation:
    def __init__(self, cfg: SimulationConfig):
        self.cfg = cfg
        self.env = Environment(cfg)

        self.py_rng = random.Random(cfg.python_seed)
        self.np_rng = np.random.default_rng(cfg.numpy_seed)

        self.sheep: list[Sheep] = []
        self.dog = Dog(
            pos=vec(180, cfg.height / 2),
            vel=vec(0, 0),
            mode=cfg.dog_mode_default,
            submode="drive",
        )

        self.time_elapsed = 0.0
        self.frames = 0
        self.success = False

        self.logger = CSVLogger(cfg.output_csv) if cfg.enable_logging else None
        self.sheep_logger = SheepStateLogger(cfg.output_sheep_csv) if cfg.enable_logging else None
        self.current_metrics = {}

        self._spawn_sheep()
        self.current_metrics = compute_metrics(self.sheep, self.dog, self.env, self.cfg, self.time_elapsed)

    def _spawn_sheep(self):
        centre = vec(self.cfg.sheep_spawn_cx, self.cfg.sheep_spawn_cy)

        for i in range(self.cfg.num_sheep):
            jitter = self.np_rng.normal(0, self.cfg.sheep_spawn_std, size=2)
            pos = centre + jitter
            pos = clamp_to_world(pos, self.cfg)

            angle = self.py_rng.uniform(0, 2 * math.pi)
            speed = self.py_rng.uniform(
                self.cfg.sheep_spawn_speed_min,
                self.cfg.sheep_spawn_speed_max,
            )
            vel = vec(math.cos(angle), math.sin(angle)) * speed

            wander_angle = self.py_rng.uniform(0, 2 * math.pi)
            wander_dir = vec(math.cos(wander_angle), math.sin(wander_angle))

            sheep = Sheep(
                idx=i,
                pos=pos,
                vel=vel,
                alert=False,
                state="calm",
                wander_dir=wander_dir,
                wander_timer=self.py_rng.uniform(
                    self.cfg.wander_change_interval_min,
                    self.cfg.wander_change_interval_max,
                ),
                fear_strength=self.py_rng.uniform(self.cfg.fear_strength_min, self.cfg.fear_strength_max),
                max_speed_multiplier=self.py_rng.uniform(self.cfg.max_speed_mult_min, self.cfg.max_speed_mult_max),
                reaction_delay_frames=self.py_rng.randint(
                    self.cfg.reaction_delay_min_frames, self.cfg.reaction_delay_max_frames
                ),
                graze_bias=self.py_rng.uniform(self.cfg.graze_bias_min, self.cfg.graze_bias_max),
                delay_counter=0,
            )

            self.sheep.append(sheep)

    def step(self, keys):
        if self.success:
            return

        if self.dog.mode == "manual":
            accel = manual_dog_acceleration(keys, self.cfg)
            self.dog.submode = "drive"
        else:
            accel, submode = autonomous_dog_acceleration(self.dog, self.sheep, self.env.goal, self.cfg)
            self.dog.submode = submode

        self.dog.pos, self.dog.vel = integrate_dog(self.dog.pos, self.dog.vel, accel, self.cfg)

        flock_centroid = centroid(self.sheep)

        new_positions = []
        new_velocities = []

        for sheep in self.sheep:
            force = sheep_force(
                sheep=sheep,
                sheep_list=self.sheep,
                dog=self.dog,
                flock_centroid=flock_centroid,
                cfg=self.cfg,
                rng=self.np_rng,
                py_rng=self.py_rng,
            )

            new_pos, new_vel = integrate_sheep(
                pos=sheep.pos,
                vel=sheep.vel,
                force=force,
                alert=sheep.alert,
                cfg=self.cfg,
                max_speed_multiplier=sheep.max_speed_multiplier,
            )

            new_positions.append(new_pos)
            new_velocities.append(new_vel)

        for sheep, pos, vel in zip(self.sheep, new_positions, new_velocities):
            sheep.pos = pos
            sheep.vel = vel

        resolve_sheep_collisions(self.sheep, self.cfg)
        resolve_dog_sheep_collisions(self.dog, self.sheep, self.cfg)

        self.time_elapsed += self.cfg.dt
        self.frames += 1
        self.success = self.env.all_sheep_in_goal(self.sheep)

        self.current_metrics = compute_metrics(self.sheep, self.dog, self.env, self.cfg, self.time_elapsed)

        if self.logger and (self.frames % self.cfg.log_every_n_frames == 0):
            self.logger.write_row(self.current_metrics)

        if self.sheep_logger and (self.frames % self.cfg.log_every_n_frames == 0):
            self.sheep_logger.write_states(self.time_elapsed, self.sheep)

    def toggle_dog_mode(self):
        self.dog.mode = "auto" if self.dog.mode == "manual" else "manual"