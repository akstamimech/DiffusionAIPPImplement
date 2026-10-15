"""Parameters shared by the belief and the sensor model. Defaults are the NAIP settings used for training."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class GridSpec:
    """The field as a regular grid of cells. Cell (yi, xi) has flat index yi * nx + xi (rows = y, columns = x)."""
    xmin: float = 0.0
    xmax: float = 100.0
    ymin: float = 0.0
    ymax: float = 100.0
    step: float = 2.0

    @property
    def xs(self):
        return np.arange(self.xmin, self.xmax + 1e-9, self.step)

    @property
    def ys(self):
        return np.arange(self.ymin, self.ymax + 1e-9, self.step)

    @property
    def nx(self):
        return len(self.xs)

    @property
    def ny(self):
        return len(self.ys)

    @property
    def n_cells(self):
        return self.nx * self.ny

    def meshgrid(self):
        """X, Y of shape (ny, nx)."""
        return np.meshgrid(self.xs, self.ys)

    def cell_coordinates(self):
        """(n_cells, 2) array of (x, y) per flat cell index."""
        X, Y = self.meshgrid()
        return np.column_stack([X.ravel(), Y.ravel()])


@dataclass(frozen=True)
class KernelSpec:
    """Matern 3/2 prior over the field: variance sigma2, lengthscale in meters (NAIP-fitted)."""
    sigma2: float = 0.0079
    lengthscale: float = 4.78


@dataclass(frozen=True)
class SensorSpec:
    """Downward camera: square footprint of half-width altitude * tan(aov/2); coarser resolution and more noise with altitude."""
    angle_of_view_deg: float = 60.0
    noise_min_variance: float = 0.01       # at altitude noise_min_altitude
    noise_max_variance: float = 0.04       # asymptote at high altitude
    noise_min_altitude: float = 10.0
    noise_growth_altitude: float = 30.0    # noise grows as 1 - exp(-ln(4) * (z - min) / growth)
    block_altitude_1: float = 20.0         # z <= this: one reading per cell
    block_altitude_2: float = 30.0         # z <= this: one reading per 2 x 2 cells, above: per 4 x 4 cells
