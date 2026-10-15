"""The lawnmower is planned once: fly the whole sweep, then stay at its end (never replan)."""
from aipp.config import GridSpec
from aipp.follower import PathFollower


class SweepFollower(PathFollower):
    def __init__(self, grid: GridSpec, z_range):
        # the original flies the sweep with a keep-out margin of two grid steps
        super().__init__(grid, z_range, execution_chunk=10 ** 9, margin=grid.step * 2)
        self._planned = False

    def load(self, path):
        super().load(path)
        self._planned = True

    def needs_new_plan(self, pose=None):
        return not self._planned
