"""
Flying at a single target point, the way the greedy-gradient planner does: every step aim straight at the target; when it
is reached (within one grid step) a new target is requested, in the same step.

One extra rule, from the original script: on the 2 m grid a move can round to "no move" while the target is still far
away (the unit direction times the step is too small in every axis). In that case the drone steps one grid step along
the axis with the largest remaining distance instead of standing still.
"""
import numpy as np

from aipp.config import GridSpec
from aipp.follower import direction_to
from aipp.interfaces import Command
from sim.vehicle import PointMassVehicle


class TargetFollower:
    def __init__(self, grid: GridSpec, z_range):
        self.grid, self.z_range = grid, tuple(z_range)
        self.margin = grid.step / 2
        self.target = None
        self._last_xy = None

    def load(self, path):
        self.target = tuple(float(v) for v in np.asarray(path, dtype=float)[0])

    def needs_new_plan(self, pose=None):
        return self.target is None or direction_to(pose, self.target, self.grid.step)[1]

    def command(self, pose) -> Command:
        step = self.grid.step
        self._last_xy = (pose[0], pose[1])
        direction, _ = direction_to(pose, self.target, step)
        command = Command(direction, self.margin, goal=self.target)
        next_pose = PointMassVehicle.next_pose(pose, command, self.grid, self.z_range)
        delta = np.array([self.target[0] - pose[0], self.target[1] - pose[1], self.target[2] - pose[2]], dtype=float)
        if np.allclose(next_pose, pose) and np.linalg.norm(delta) > step:
            axis = int(np.argmax(np.abs(delta)))
            unit = [0.0, 0.0, 0.0]
            unit[axis] = float(np.sign(delta[axis]))
            command = Command(tuple(unit), self.margin, goal=self.target)
        return command

    def remaining_xy(self):
        return [self._last_xy, (self.target[0], self.target[1])]
