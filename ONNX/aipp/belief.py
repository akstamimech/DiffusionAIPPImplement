"""
The Gaussian-process belief over the field: posterior mean mu and covariance P over all grid cells, with the
Kalman update for one observation. numpy only.

The prior is a zero-mean Matern 3/2 process (closed form below, identical to sklearn's ConstantKernel * Matern(nu=1.5))
shifted by a constant prior mean. For the NAIP setup the prior mean sits below the importance threshold.
"""
import numpy as np

from .config import GridSpec, KernelSpec


def matern32_covariance(grid: GridSpec, kernel: KernelSpec):
    """(n_cells, n_cells) prior covariance: sigma2 * (1 + sqrt(3) d / l) * exp(-sqrt(3) d / l)."""
    xy = grid.cell_coordinates()
    d = np.hypot(xy[:, 0][:, None] - xy[:, 0][None, :], xy[:, 1][:, None] - xy[:, 1][None, :])
    a = np.sqrt(3.0) * d / kernel.lengthscale
    return kernel.sigma2 * (1.0 + a) * np.exp(-a)


class GPBelief:
    def __init__(self, grid: GridSpec, kernel: KernelSpec, prior_mean: float, prior_cov=None):
        self.grid = grid
        self.mu = np.full(grid.n_cells, float(prior_mean))
        self.P = matern32_covariance(grid, kernel) if prior_cov is None else prior_cov.copy()

    def update(self, H, z, noise_var):
        """Kalman update with readings z = H @ field + noise (independent noise of variance noise_var per row).
        Replaces self.mu and self.P with new arrays (the old ones are left untouched)."""
        if H.shape[0] == 0:
            return
        v = z - H @ self.mu
        projected = H @ self.P                                    # (B, n)
        S = H @ projected.T + noise_var * np.eye(H.shape[0])      # (B, B), small
        try:
            solved = np.linalg.solve(S, projected)
        except np.linalg.LinAlgError:
            solved = np.linalg.pinv(S) @ projected
        K = solved.T                                              # (n, B)
        self.mu = self.mu + (K @ v).flatten()
        self.P = self.P - K @ projected

    def variance(self):
        return np.diag(self.P)

    def total_variance(self):
        return float(np.sum(np.diag(self.P)))

    def mean_map(self):
        """(ny, nx), rows = y, columns = x: the network input layout."""
        return self.mu.reshape(self.grid.ny, self.grid.nx)

    def var_map(self):
        return self.variance().reshape(self.grid.ny, self.grid.nx)
