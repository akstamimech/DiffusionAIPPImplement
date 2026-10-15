"""
What a camera at a pose observes. Pure geometry, numpy only, shared by the simulator and the real drone.

At altitude z the camera sees every grid cell inside a square of half-width z * tan(aov / 2). Cells are read out in
square resolution blocks (1, 2 or 4 cells wide depending on altitude); each block yields ONE reading, the average of
its visible cells, with noise variance noise_variance(z) that grows with altitude.
"""
from dataclasses import dataclass

import numpy as np

from .config import GridSpec, SensorSpec


def fov_radius(altitude, sensor: SensorSpec):
    return float(altitude) * np.tan(np.deg2rad(sensor.angle_of_view_deg) / 2.0)


def block_size(altitude, sensor: SensorSpec):
    if altitude <= sensor.block_altitude_1:
        return 1
    if altitude <= sensor.block_altitude_2:
        return 2
    return 4


def noise_variance(altitude, sensor: SensorSpec):
    b = np.log(4.0) / sensor.noise_growth_altitude
    return sensor.noise_min_variance + (sensor.noise_max_variance - sensor.noise_min_variance) * (
        1.0 - np.exp(-b * (altitude - sensor.noise_min_altitude))
    )


@dataclass(frozen=True)
class Observation:
    """H (B x n_cells, dense): row b averages the visible cells of block b. A reading is z = H @ field + noise,
    noise independent per block with variance noise_var. Rows are ordered by first appearance when scanning the
    footprint x-major (x outer loop, y inner loop)."""
    H: np.ndarray
    noise_var: float
    cells: np.ndarray            # flat indices of the visible cells, footprint scan order
    block_of_cell: np.ndarray    # block row index per visible cell


def observation(cx, cy, cz, grid: GridSpec, sensor: SensorSpec) -> Observation:
    xs, ys = grid.xs, grid.ys
    r = fov_radius(cz, sensor)
    vis_xi = np.flatnonzero((xs >= cx - r) & (xs <= cx + r))
    vis_yi = np.flatnonzero((ys >= cy - r) & (ys <= cy + r))
    n = grid.n_cells
    if len(vis_xi) == 0 or len(vis_yi) == 0:
        return Observation(np.zeros((0, n)), float(noise_variance(cz, sensor)), np.zeros(0, dtype=int), np.zeros(0, dtype=int))

    XI, YI = np.meshgrid(vis_xi, vis_yi, indexing="ij")          # x-major scan like the original footprint loop
    XI, YI = XI.ravel(), YI.ravel()
    bs = block_size(cz, sensor)
    block_key = (YI // bs) * (grid.nx // bs + 2) + (XI // bs)    # unique id per resolution block
    _, first, inverse, counts = np.unique(block_key, return_index=True, return_inverse=True, return_counts=True)
    order = np.argsort(first)                                    # blocks by first appearance in the scan
    rank = np.empty_like(order)
    rank[order] = np.arange(len(order))
    block_of_cell = rank[inverse.ravel()]
    cells = YI * grid.nx + XI

    H = np.zeros((len(order), n))
    H[block_of_cell, cells] = (1.0 / counts[order])[block_of_cell]
    return Observation(H, float(noise_variance(cz, sensor)), cells, block_of_cell)
