"""
aipp: the deployable core of the adaptive informative path planner (numpy only).

    config        grid, kernel and sensor parameters (one place)
    sensor_model  what a camera at a pose observes (footprint, resolution blocks, noise)
    belief        the Gaussian-process belief over the field (mean, covariance, Kalman update)
    interfaces    FlightPlan / Command and the three seams: MeasurementSource, Vehicle, PathPlanner
    follower      turns a flight plan into one motion command per step
    mission       the loop: move, measure, update, replan

Nothing in here may import the simulator, the planners, sklearn, scipy, matplotlib or the training code. The simulator
lives in the top-level `sim` package (ground truth, noisy camera, point-mass drone, scoring) and every planner in
`planners/` plugs into the same loop; a real drone supplies its own Vehicle and MeasurementSource.
"""
from .config import GridSpec, KernelSpec, SensorSpec
from .sensor_model import Observation, observation
from .belief import GPBelief, matern32_covariance
from .interfaces import Command, FlightPlan, Follower, MeasurementSource, PathPlanner, Pose, Vehicle
from .follower import PathFollower, direction_to
from .mission import Mission, MissionConfig, MissionResult, StepRecord

__all__ = [
    "GridSpec", "KernelSpec", "SensorSpec", "Observation", "observation", "GPBelief", "matern32_covariance",
    "Command", "FlightPlan", "Follower", "MeasurementSource", "PathPlanner", "Pose", "Vehicle",
    "PathFollower", "direction_to", "Mission", "MissionConfig", "MissionResult", "StepRecord",
]
