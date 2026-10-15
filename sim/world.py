"""The simulated world: the ground-truth field the drone is trying to map."""
from pathlib import Path

import numpy as np

from aipp.config import GridSpec


def load_ground_truth(csv_dir, map_id, maptype, grid: GridSpec):
    """Reads csv/map_<id>_<MAPTYPE>_grid_counts.csv. Returns (pts, true_map_flat): the rows (x, y, value) that lie
    on the grid, and the field as a flat vector in the belief's cell order (cells not listed stay 0)."""
    data = np.loadtxt(Path(csv_dir) / f"map_{map_id}_{maptype}_grid_counts.csv", delimiter=",", skiprows=1)
    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], grid.step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], grid.step), 0.0, atol=tol)
    )
    pts = pts[mask]

    cells = grid.cell_coordinates()
    true_map_flat = np.zeros(cells.shape[0], dtype=float)
    for x_true, y_true, value in pts:
        idx = np.where(np.isclose(cells[:, 0], x_true) & np.isclose(cells[:, 1], y_true))[0]
        if idx.size:
            true_map_flat[idx[0]] = value
    return pts, true_map_flat
