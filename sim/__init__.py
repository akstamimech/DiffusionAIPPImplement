"""
The simulator shared by every planner: the ground-truth world, a noisy camera, a point-mass drone, and the flight
driver that scores a planner. Lazy about the heavy parts: `from sim import run_flight` pulls in plotting and the
metrics, `from sim import SimSensor` does not.
"""
from .sensor import SimSensor
from .vehicle import PointMassVehicle
from .world import load_ground_truth

__all__ = ["SimSensor", "PointMassVehicle", "load_ground_truth", "FlightSettings", "run_flight"]


def __getattr__(name):
    if name in ("FlightSettings", "FlightResult", "run_flight"):
        from . import flight
        return getattr(flight, name)
    raise AttributeError(name)
