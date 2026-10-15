"""Simulated camera: readings come from a ground-truth field plus seeded noise. Implements the MeasurementSource idea:
measure(pose) -> (H, z, noise_var), which is exactly what GPBelief.update consumes."""
import numpy as np

from aipp.config import GridSpec, SensorSpec
from aipp.sensor_model import observation


class SimSensor:
    def __init__(self, truth_flat, grid: GridSpec, sensor: SensorSpec, rng):
        self.truth = np.asarray(truth_flat, dtype=float)
        self.grid, self.sensor, self.rng = grid, sensor, rng

    def measure(self, pose):
        cx, cy, cz = pose
        obs = observation(cx, cy, cz, self.grid, self.sensor)
        sd = np.sqrt(obs.noise_var)
        noise = np.array([self.rng.normal(0.0, sd) for _ in range(obs.H.shape[0])])   # one draw per block, in block order
        return obs.H, obs.H @ self.truth + noise, obs.noise_var
