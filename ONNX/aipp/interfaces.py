"""
The seams of the system: small data types and the three things the mission loop talks to.

    MeasurementSource  what the camera gives back for a pose     (simulated: aipp.sim.SimSensor, real: the camera pipeline)
    Vehicle            where the drone is and how it is moved    (simulated: aipp.sim.PointMassVehicle, real: the autopilot link)
    PathPlanner        belief maps + pose in, flight plan out    (aipp.planner.DiffusionPlanner)

The mission loop only depends on these, so swapping the simulator for hardware means writing a Vehicle and a
MeasurementSource, nothing else.
"""
from dataclasses import dataclass
from typing import Protocol, Tuple

import numpy as np

Pose = Tuple[float, float, float]


@dataclass(frozen=True)
class FlightPlan:
    waypoints: np.ndarray    # (K, 3) control waypoints the network predicted, meters
    path: np.ndarray         # (N, 3) dense spline through the pose and the waypoints, flown point by point
    seconds: float           # wall-clock time the planner took


@dataclass(frozen=True)
class Command:
    """One step of motion. direction is a unit vector toward the point being tracked (zeros = stay in place).
    margin is how far inside the field edge the vehicle may be sent; a simulated vehicle clips to it, hardware
    can ignore it (the geofence lives in the autopilot). hold = there was nothing to track, do not move at all.
    clamped = the tracked waypoint had to be pulled back into the padded field."""
    direction: Tuple[float, float, float]
    margin: float
    hold: bool = False
    clamped: bool = False


class MeasurementSource(Protocol):
    def measure(self, pose: Pose) -> Tuple[np.ndarray, np.ndarray, float]:
        """Returns (H, z, noise_var): readings z = H @ field + noise, ready for GPBelief.update."""
        ...


class Vehicle(Protocol):
    @property
    def pose(self) -> Pose: ...

    def fly(self, command: Command) -> None:
        """Carry out one step of motion; afterwards self.pose is the new pose."""
        ...


class PathPlanner(Protocol):
    def plan(self, mean_map: np.ndarray, var_map: np.ndarray, pose: Pose) -> FlightPlan: ...
