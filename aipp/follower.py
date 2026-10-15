"""
Executing a flight plan: turns the dense spline into one motion command per step.

The plan is a list of 3D points. Each step the follower aims at the current point; once the vehicle is within one grid
step of it, it advances to the next point. A plan is flown for at most execution_chunk points, then a new one is
requested (receding horizon). numpy only.
"""
import numpy as np

from .config import GridSpec
from .interfaces import Command


def direction_to(pose, goal, tolerance):
    """Unit vector from pose toward goal, and whether the goal counts as reached (within tolerance, then the
    direction is zero)."""
    displacement = np.array([goal[0] - pose[0], goal[1] - pose[1], goal[2] - pose[2]])
    distance = np.linalg.norm(displacement)
    if distance <= tolerance:
        return (0.0, 0.0, 0.0), True
    direction = displacement / distance
    return (direction[0], direction[1], direction[2]), False


class PathFollower:
    def __init__(self, grid: GridSpec, z_range, execution_chunk=40, margin=None):
        self.grid = grid
        self.z_min, self.z_max = float(z_range[0]), float(z_range[1])
        self.execution_chunk = int(execution_chunk)
        self.margin = grid.step / 2 if margin is None else float(margin)    # keep-out margin handed to the vehicle
        self.path = []           # list of [x, y, z]
        self.index = 0

    def load(self, path):
        """Start flying a new plan (array or list of [x, y, z])."""
        self.path = np.asarray(path, dtype=float).tolist()
        self.index = 0

    def needs_new_plan(self, pose=None):
        return self.index >= self.execution_chunk or self.index >= len(self.path)

    def remaining_xy(self):
        return [(x, y) for x, y, _ in self.path[self.index:]]

    def command(self, pose) -> Command:
        step = self.grid.step
        margin = self.margin
        if self.index >= len(self.path):
            return Command((0.0, 0.0, 0.0), margin, hold=True)

        direction, reached = direction_to(pose, self.path[self.index], step)
        if reached:
            self.index += 1
            if self.index < len(self.path):
                direction, _ = direction_to(pose, self.path[self.index], step)
            else:
                direction = (0.0, 0.0, 0.0)

        # The next point is pulled back into the padded field (the planner already clips its output, so this only
        # fires if a plan was produced elsewhere). The correction applies from the following step on.
        clamped = False
        if self.index < len(self.path):
            x, y, z = self.path[self.index]
            pad = 2 * step
            cx = min(max(x, self.grid.xmin + pad), self.grid.xmax - pad)
            cy = min(max(y, self.grid.ymin + pad), self.grid.ymax - pad)
            cz = min(max(z, self.z_min), self.z_max)
            if cx != x or cy != y or cz != z:
                self.path[self.index] = [cx, cy, cz]
                clamped = True
        return Command(direction, margin, clamped=clamped)
