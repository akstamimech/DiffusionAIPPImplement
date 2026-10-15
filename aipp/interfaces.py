"""
The seams of the system: small data types and the three things the mission loop talks to.

    MeasurementSource  what the camera gives back for a pose     (simulated: sim.SimSensor, real: the camera pipeline)
    Vehicle            where the drone is and how it is moved    (simulated: sim.PointMassVehicle, real: the autopilot link)
    PathPlanner        belief + pose in, flight plan out         (planners/*: diffusion, CMA-ES, greedy, ...)

The mission loop only depends on these, so swapping the simulator for hardware means writing a Vehicle and a
MeasurementSource, nothing else.
"""
from dataclasses import dataclass
from typing import List, Optional, Protocol, Tuple

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
    clamped = the tracked waypoint had to be pulled back into the padded field.
    goal = the point being tracked, for vehicles or followers that need it."""
    direction: Tuple[float, float, float]
    margin: float
    hold: bool = False
    clamped: bool = False
    goal: Optional[Tuple[float, float, float]] = None


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


class Follower(Protocol):
    """Turns the current flight plan into one Command per step (see aipp.follower.PathFollower)."""
    def needs_new_plan(self, pose: Pose) -> bool: ...

    def load(self, path) -> None: ...

    def command(self, pose: Pose) -> Command: ...

    def remaining_xy(self) -> List[Tuple[float, float]]: ...


class PathPlanner(Protocol):
    """The only thing that differs between planners. It sees the belief (mean, covariance, maps) and the pose."""
    z_range: Tuple[float, float]          # altitude band the planner flies in (min, max), meters

    def plan(self, belief, pose: Pose) -> FlightPlan: ...
