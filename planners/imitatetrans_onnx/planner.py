"""
The ONNX imitation planner as the mission loop sees it: belief and pose in, a FlightPlan out.

The network math (input normalization, the position marker, denormalization, the spline) lives in onnx_reference.py,
the single-file reference the C++ port follows and the one its self-test checks against PyTorch. This wrapper adds what
the mission needs on top of it: keeping the plan inside the padded field, timing, and the FlightPlan type.
Needs onnxruntime.
"""
import time
from dataclasses import dataclass

import numpy as np

from aipp.config import GridSpec
from aipp.interfaces import FlightPlan

from .onnx_reference import Planner as _Network


@dataclass(frozen=True)
class PlannerConfig:
    padding_cells: float = 2.0          # plans stay this many grid steps inside the field edge


class ImitationOnnxPlanner:
    def __init__(self, model_path, deploy_json, grid: GridSpec, config: PlannerConfig = PlannerConfig(), threads=None):
        self.grid = grid
        self.config = config
        self.network = _Network(model_path, deploy_json, threads=threads)

    @property
    def z_range(self):
        w = self.network.w
        return float(w["z_min_m"]), float(w["z_max_m"])

    def plan(self, belief, pose) -> FlightPlan:
        mean_map, var_map = belief.mean_map(), belief.var_map()
        started = time.time()
        position = np.asarray(pose, dtype=np.float64).reshape(3)
        maps, position_normalized, total_variance = self.network.build_inputs(mean_map, var_map, position)
        waypoints = self.network.denormalize(self.network.predict_normalized(maps, position_normalized, total_variance))   # (8, 3)
        path = self.network.cubic_spline(position, waypoints)                                                              # (41, 3)
        pad = self.grid.step * self.config.padding_cells
        g = self.grid
        z_min, z_max = self.z_range
        for points in (path, waypoints):
            points[:, 0] = np.clip(points[:, 0], g.xmin + pad, g.xmax - pad)
            points[:, 1] = np.clip(points[:, 1], g.ymin + pad, g.ymax - pad)
            points[:, 2] = np.clip(points[:, 2], z_min, z_max)
        return FlightPlan(waypoints=waypoints, path=path, seconds=time.time() - started)
