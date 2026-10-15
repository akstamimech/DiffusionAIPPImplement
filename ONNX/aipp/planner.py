"""
The diffusion planner as the mission loop sees it: belief maps and a pose in, a FlightPlan out.

The network math (input normalization, DDIM loop, denormalization, spline) lives in onnx_sample_reference.py, the
single-file reference the C++ port follows and the one the self-test checks against PyTorch. This wrapper adds what the
mission needs on top of it: the sampling settings, keeping the plan inside the padded field, timing, and the FlightPlan
type. Needs onnxruntime (aipp.__init__ does not import this module, so the rest of aipp stays numpy only).
"""
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_ONNX_DIR = str(Path(__file__).resolve().parent.parent)
if _ONNX_DIR not in sys.path:
    sys.path.insert(0, _ONNX_DIR)
from onnx_sample_reference import Planner as _Network  # noqa: E402

from .config import GridSpec  # noqa: E402
from .interfaces import FlightPlan  # noqa: E402


@dataclass(frozen=True)
class PlannerConfig:
    eta: float = 0.0                    # DDIM stochasticity, 0 = deterministic given the initial noise
    ddim_steps: Optional[int] = None    # None = all T denoising steps
    clip_x0: bool = True
    padding_cells: float = 2.0          # plans stay this many grid steps inside the field edge


class DiffusionPlanner:
    def __init__(self, model_path, deploy_json, grid: GridSpec, config: PlannerConfig = PlannerConfig(),
                 threads=None, seed=None):
        self.grid = grid
        self.config = config
        self.network = _Network(model_path, deploy_json, threads=threads, seed=seed)

    @property
    def T(self):
        return self.network.T

    @property
    def z_range(self):
        w = self.network.w
        return float(w["z_min_m"]), float(w["z_max_m"])

    def plan(self, mean_map, var_map, pose) -> FlightPlan:
        started = time.time()
        position = np.asarray(pose, dtype=np.float64).reshape(3)
        maps, position_normalized, total_variance = self.network.build_inputs(mean_map, var_map, position)
        sample = self.network.sample_normalized(
            maps, position_normalized, total_variance,
            eta=self.config.eta, clip_x0=self.config.clip_x0, num_steps=self.config.ddim_steps,
        )
        waypoints = self.network.denormalize(sample)                           # (8, 3) meters
        path = self.network.cubic_spline(position, waypoints)                  # (41, 3)
        lo_x, hi_x, lo_y, hi_y, (z_min, z_max) = self._padded_bounds()
        for points in (path, waypoints):
            points[:, 0] = np.clip(points[:, 0], lo_x, hi_x)
            points[:, 1] = np.clip(points[:, 1], lo_y, hi_y)
            points[:, 2] = np.clip(points[:, 2], z_min, z_max)
        return FlightPlan(waypoints=waypoints, path=path, seconds=time.time() - started)

    def _padded_bounds(self):
        pad = self.grid.step * self.config.padding_cells
        g = self.grid
        return g.xmin + pad, g.xmax - pad, g.ymin + pad, g.ymax - pad, self.z_range
