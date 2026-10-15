"""
The greedy-gradient planner: the CMA-ES planner's 3D pyramid-lattice grid search with a short horizon, and nothing
else. Each replan scores the lattice (expected variance reduction per unit distance), takes the first lattice point
that is farther than one grid step away, and the drone flies straight at it. No CMA-ES refinement, no spline.
Replanning is event-triggered (see follower.TargetFollower): only once the target is reached.
"""
import time

import numpy as np

from aipp.config import GridSpec
from aipp.interfaces import FlightPlan
from functions import grid_search_3d


class GreedyGradientPlanner:
    def __init__(self, grid: GridSpec, utility_threshold=0.3, beta=1.0, planning_horizon=2, z_range=(10.0, 40.0)):
        self.grid = grid
        self.utility_threshold, self.beta, self.planning_horizon = utility_threshold, beta, planning_horizon
        self.z_range = tuple(z_range)

    def plan(self, belief, pose) -> FlightPlan:
        started = time.time()
        cx, cy, cz = pose
        zmin, zmax = self.z_range
        control_waypoints = grid_search_3d(
            belief.mu, belief.P, self.grid.xs, self.grid.ys,
            start_pose=(cx, cy, cz),
            beta=self.beta,
            utility_threshold=self.utility_threshold,
            planning_horizon=self.planning_horizon,
            zmin=zmin, zmax=zmax,
        )
        current_pose = np.array([cx, cy, cz], dtype=float)
        reachable_now_tolerance = self.grid.step
        target = control_waypoints[-1]
        for candidate in control_waypoints:
            if np.linalg.norm(np.asarray(candidate, dtype=float) - current_pose) > reachable_now_tolerance:
                target = candidate
                break
        seconds = time.time() - started
        print(f"Next grid point: ({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})")
        target = np.asarray(target, dtype=float).reshape(1, 3)
        return FlightPlan(waypoints=target, path=target.copy(), seconds=seconds)
