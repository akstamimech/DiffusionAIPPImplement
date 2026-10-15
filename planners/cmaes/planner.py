"""
The CMA-ES receding-horizon planner (the expert the diffusion policy imitates).

Every replan: a 3D pyramid-lattice grid search proposes `planning_horizon` waypoints that maximize expected variance
reduction per unit distance, CMA-ES refines them into continuous waypoints (objective: expected masked variance
reduction along the spline through them), and a natural cubic spline through the pose and the waypoints is the plan.
It needs the belief's covariance (not only the maps), which is why planners receive the whole belief.
"""
import time

import numpy as np

from aipp.config import GridSpec
from aipp.interfaces import FlightPlan
from functions import (
    build_spline_trajectory_3d,
    clip_waypoints_continuous_3d,
    cma_es_refine_waypoints_3d,
    grid_search_3d,
    grid_search_3d_softmax,
)


class CMAESPlanner:
    def __init__(
        self,
        grid: GridSpec,
        utility_threshold=0.3,
        beta=1.0,
        planning_horizon=8,
        alpha=0.02,                      # distance weight of the lattice score
        cma_seed=55,
        refine=True,                     # False: fly the raw grid-search waypoints (the "lattice-only" baseline)
        softmax_grid_search=False,       # True: sample the warm start stochastically instead of the argmax
        samples_per_segment=5,
        z_range=(10.0, 40.0),
    ):
        self.grid = grid
        self.utility_threshold, self.beta = utility_threshold, beta
        self.planning_horizon, self.alpha = planning_horizon, alpha
        self.cma_seed, self.refine, self.softmax_grid_search = cma_seed, refine, softmax_grid_search
        self.samples_per_segment = samples_per_segment
        self.z_range = tuple(z_range)

    def waypoints(self, mu, P, pose):
        """The control waypoints (planning_horizon of them) for a belief (mu, P) seen from `pose`."""
        cx, cy, cz = pose
        xs, ys = self.grid.xs, self.grid.ys
        zmin, zmax = self.z_range
        grid_search_fn = grid_search_3d_softmax if self.softmax_grid_search else grid_search_3d
        flight_plan_3d = grid_search_fn(
            mu, P, xs, ys,
            start_pose=(cx, cy, cz),
            beta=self.beta,
            utility_threshold=self.utility_threshold,
            planning_horizon=self.planning_horizon,
            zmin=zmin, zmax=zmax,
            alpha=self.alpha,
        )
        if not self.refine:
            return clip_waypoints_continuous_3d(flight_plan_3d, xs, ys, zmin, zmax, margin=self.grid.step * 2)
        return cma_es_refine_waypoints_3d(
            flight_plan_3d, mu, P, xs, ys, cx, cy, cz,
            self.beta, self.utility_threshold, zmin, zmax,
            predictive_variance=True,
            seed=self.cma_seed,
        )

    def plan(self, belief, pose) -> FlightPlan:
        started = time.time()
        cx, cy, cz = pose
        control_waypoints = self.waypoints(belief.mu, belief.P, pose)
        path = build_spline_trajectory_3d(cx, cy, cz, control_waypoints, samples_per_segment=self.samples_per_segment)
        return FlightPlan(
            waypoints=np.asarray(control_waypoints, dtype=float).reshape(-1, 3),
            path=np.asarray(path, dtype=float).reshape(-1, 3),
            seconds=time.time() - started,
        )
