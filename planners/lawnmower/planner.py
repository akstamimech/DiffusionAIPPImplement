"""
The lawnmower planner: a fixed back-and-forth sweep of the field at constant altitude, laid out once from the start
position. It ignores the belief, so it is the non-adaptive reference every adaptive planner has to beat.

Row spacing is two camera footprint radii (rounded to the grid), so neighbouring rows just touch.
"""
import time

import numpy as np

from aipp.config import GridSpec
from aipp.interfaces import FlightPlan
from functions import fov_lateral_radius


def grid_spaced_values(start, stop, spacing, step):
    direction = 1.0 if stop >= start else -1.0
    values = list(np.arange(start, stop + direction * 1e-9, direction * spacing))
    if not values or not np.isclose(values[-1], stop):
        values.append(stop)
    return [float(step * np.round(value / step)) for value in values]


def lawnmower_waypoints(cx, cy, grid: GridSpec, fov_radius, buffer=None):
    """The (x, y) sweep: from the current row up to the top edge, alternating direction row by row."""
    step = grid.step
    buffer = step * 2 if buffer is None else buffer
    x_left, x_right = grid.xmin + buffer, grid.xmax - buffer
    y_top = grid.ymax - buffer
    sweep_spacing = max(step, step * np.round((2.0 * fov_radius) / step))

    cx = step * np.round(cx / step)                      # snap the current position to the grid
    cy = step * np.round(cy / step)
    flight_plan = []
    y_values = grid_spaced_values(cy, y_top, sweep_spacing, step)
    for row_idx, y in enumerate(y_values):
        if row_idx == 0:
            x_values = grid_spaced_values(cx, x_right, sweep_spacing, step)
        elif row_idx % 2 == 1:
            x_values = grid_spaced_values(x_right, x_left, sweep_spacing, step)
        else:
            x_values = grid_spaced_values(x_left, x_right, sweep_spacing, step)
        for x in x_values:
            waypoint = (float(step * np.round(x / step)), float(step * np.round(y / step)))
            if len(flight_plan) == 0 and np.isclose(waypoint[0], cx) and np.isclose(waypoint[1], cy):
                continue                                  # do not start by "flying" to where we already are
            flight_plan.append(waypoint)
    return flight_plan


class LawnmowerPlanner:
    def __init__(self, grid: GridSpec, altitude=10.0, z_range=(10.0, 40.0)):
        self.grid, self.altitude, self.z_range = grid, altitude, tuple(z_range)

    def plan(self, belief, pose) -> FlightPlan:
        started = time.time()
        cx, cy, cz = pose
        waypoints = lawnmower_waypoints(cx, cy, self.grid, fov_radius=fov_lateral_radius(cz))
        path = np.array([[x, y, self.altitude] for x, y in waypoints], dtype=float).reshape(-1, 3)
        return FlightPlan(waypoints=path, path=path.copy(), seconds=time.time() - started)
