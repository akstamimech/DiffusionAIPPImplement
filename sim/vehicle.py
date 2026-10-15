"""Simulated vehicle: a point mass that moves one grid step per command and sits on the grid (position and altitude
are snapped to the grid step). A real drone replaces this with a link to the autopilot."""
import numpy as np

from aipp.config import GridSpec
from aipp.interfaces import Command


class PointMassVehicle:
    def __init__(self, start_pose, grid: GridSpec, z_range, samplestep=None):
        self.grid = grid
        self.z_min, self.z_max = float(z_range[0]), float(z_range[1])
        self.samplestep = grid.step if samplestep is None else float(samplestep)
        self._pose = tuple(start_pose)

    @property
    def pose(self):
        return self._pose

    @staticmethod
    def next_pose(pose, command: Command, grid: GridSpec, z_range, samplestep=None):
        """Where a point mass at `pose` ends up after `command` (the whole dynamics, without moving anything)."""
        if command.hold:
            return tuple(pose)
        s, m = grid.step, command.margin
        samplestep = s if samplestep is None else samplestep
        gx, gy, gz = command.direction
        cx, cy, cz = pose
        cx = np.clip(cx + gx * samplestep, grid.xmin + m, grid.xmax - m)
        cy = np.clip(cy + gy * samplestep, grid.ymin + m, grid.ymax - m)
        cz = np.clip(cz + gz * samplestep, z_range[0], z_range[1])
        return (s * np.round(cx / s), s * np.round(cy / s), s * np.round(cz / s))

    def fly(self, command: Command):
        self._pose = self.next_pose(self._pose, command, self.grid, (self.z_min, self.z_max), self.samplestep)
