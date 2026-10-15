"""Simulated vehicle: a point mass that moves one grid step per command and sits on the grid (position and altitude
are snapped to the grid step). A real drone replaces this with a link to the autopilot."""
import numpy as np

from ..config import GridSpec
from ..interfaces import Command


class PointMassVehicle:
    def __init__(self, start_pose, grid: GridSpec, z_range, samplestep=None):
        self.grid = grid
        self.z_min, self.z_max = float(z_range[0]), float(z_range[1])
        self.samplestep = grid.step if samplestep is None else float(samplestep)
        self._pose = tuple(start_pose)

    @property
    def pose(self):
        return self._pose

    def fly(self, command: Command):
        if command.hold:
            return
        g, s, m = self.grid, self.grid.step, command.margin
        gx, gy, gz = command.direction
        cx, cy, cz = self._pose
        cx = np.clip(cx + gx * self.samplestep, g.xmin + m, g.xmax - m)
        cy = np.clip(cy + gy * self.samplestep, g.ymin + m, g.ymax - m)
        cz = np.clip(cz + gz * self.samplestep, self.z_min, self.z_max)
        self._pose = (s * np.round(cx / s), s * np.round(cy / s), s * np.round(cz / s))
