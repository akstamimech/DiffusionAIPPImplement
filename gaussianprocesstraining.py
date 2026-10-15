import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
import cma

from scipy.interpolate import CubicSpline
from scipy.linalg import block_diag, solve as spd_solve
from scipy.sparse import csr_matrix, issparse, vstack as sparse_vstack

step = 2.0
CMA_SEED = 31
# Env-configurable so HPC/local sweep scripts can vary the CMA-ES budget
# (effective evaluations ~= CMA_PREDICTIVE_MAXITER * CMA_PREDICTIVE_POPSIZE,
# since maxiter binds before maxfevals at these defaults) without editing this
# file per run.
CMA_PREDICTIVE_MAXITER = int(os.environ.get("CMA_PREDICTIVE_MAXITER", "25"))
CMA_PREDICTIVE_MAXFEVALS = int(os.environ.get("CMA_PREDICTIVE_MAXFEVALS", "1000"))
CMA_PREDICTIVE_POPSIZE = int(os.environ.get("CMA_PREDICTIVE_POPSIZE", "12"))

# Per-axis initial CMA-ES step sizes for cma_es_refine_waypoints_3d, replacing a
# single flat sigma0. Following Popovic et al. (2020)'s own step-size tuning
# methodology (their CMA-ES(x,y,z) sweep against a lattice-only baseline), but
# scaled to this workspace's larger xy extent (~92m vs their ~30m) and z range
# (30m vs their ~25m). A flat sigma0=4.0 was an oversized step relative to the
# 30m z-range specifically, which - as in their own (10,12) case - made CMA-ES
# perform no better (often worse) than the lattice search it refines; these
# smaller, per-axis values were swept against the lattice-only baseline on
# real maps and consistently beat both the flat sigma0=4.0 default and, on
# most maps/metrics, lattice search alone. NOTE: that sweep predates this
# codebase's cost-normalization/boundary-penalty fixes to trajectory_objective_3d
# (see its docstring/history), so it has not been re-validated under the fixed
# objective - env-configurable so it can be swept again without editing this
# file per run, the same treatment CMA_PREDICTIVE_MAXITER/BETA already got.
CMA_STEP_SIZE_XY = float(os.environ.get("CMA_STEP_SIZE_XY", "20"))
CMA_STEP_SIZE_Z = float(os.environ.get("CMA_STEP_SIZE_Z", "8"))

# --- grid_search_3d softmax-selection hyperparameters ---
# grid_search_3d_softmax replaces the per-step hard argmax over candidate
# waypoints with a softmax sample, so the greedy warm start handed to CMA-ES
# (and therefore the final refined trajectory) can vary instead of being
# bit-identical every call.
GRID_SEARCH_SOFTMAX_TEMPERATURE = 0.2  # 0 -> recovers hardmax/argmax; higher -> more
                                        # uniform over candidates. Scores are min-max
                                        # normalized to [0, 1] before scaling by this,
                                        # so the value is comparable across maps/rounds
                                        # regardless of the raw gain/distance magnitude.
GRID_SEARCH_SOFTMAX_SEED = None  # None -> draws from numpy's global RNG state (a
                                  # different sample every call); set an int for
                                  # reproducible waypoint sampling.

# --- grid_search_3d tie-break selection hyperparameters ---
# grid_search_3d_tiebreak replaces the per-step hard argmax with a uniform-random
# pick among candidates within GRID_SEARCH_TIE_TOLERANCE of the top score, instead
# of softmax's approach of weighting every candidate (including strictly worse
# ones). This never trades away score - it only resolves genuine (or
# near-floating-point) ties, which the plain argmax otherwise always breaks the
# same way (first candidate in build_pyramid_lattice_3d's fixed generation order).
GRID_SEARCH_TIE_TOLERANCE = 1e-6  # fraction of (max - min) score spread within
                                   # this round that counts as "tied" with the top
                                   # score. Small on purpose - only meant to catch
                                   # true/near-exact ties, not meaningfully worse
                                   # candidates.
GRID_SEARCH_TIEBREAK_SEED = None  # None -> draws from numpy's global RNG state;
                                   # set an int for reproducible tie-breaking.


LCB = True  # LCB/NAIP: cells are important when mu - beta*sigma stays at/below the threshold.

COST_EXPONENT = float(os.environ.get("COST_EXPONENT", "1.0"))  # Exponent applied to the distance/cost denominator in the grid-search
                      # step score (gain / distance**COST_EXPONENT, see _grid_search_3d_impl)
                      # and the CMA-ES refinement objective (variance_reduction / cost**COST_EXPONENT,
                      # see trajectory_objective_3d). 1.0 = original linear gain/cost ratio
                      # (mirrors Popovic et al. 2020); lower values flatten the distance
                      # penalty so a large gain far away competes more easily against a
                      # smaller gain nearby, encouraging trajectories that travel to a
                      # separate high-value cluster instead of exhausting the nearest one.
                      # Values below 1.0 remain available as an explicit override for maps
                      # with spatially separated high-value clusters.

###FOV FUNCTIONS 

def fov_lateral_radius(altitude, angle_of_view=60.0):
    return float(altitude) * np.tan(np.deg2rad(angle_of_view) / 2.0)


def resolution_block_size(altitude): 
    if altitude <= 20:
        return 1
    elif altitude <= 30:
        return 2
    else:
        return 4


def noise_model(altitude, min_altitude=10.0):
    min_variance = 0.01
    max_variance = 0.04
    b = np.log(4.0) / 30.0

    return min_variance + (max_variance - min_variance) * (
        1.0 - np.exp(-b * (altitude - min_altitude))
    )


def build_sensor_matrix(fov, cz, xs, ys, return_block_ids=False):
    nx = len(xs)
    ny = len(ys)
    block_size = resolution_block_size(cz)

    if len(fov) == 0:
        visible_indices = []
    else:
        # Callers (compute_fov/fov_grid_points) always build `fov` by filtering
        # the xs/ys arrays themselves, so every point here already lies exactly
        # on the (uniformly spaced) grid. Compute the index arithmetically
        # instead of doing an O(len(xs)) argmin scan per point.
        fov_arr = np.asarray(fov, dtype=float)
        fx, fy = fov_arr[:, 0], fov_arr[:, 1]

        dx_step = xs[1] - xs[0] if nx > 1 else 1.0
        dy_step = ys[1] - ys[0] if ny > 1 else 1.0

        xi = np.rint((fx - xs[0]) / dx_step).astype(int)
        yi = np.rint((fy - ys[0]) / dy_step).astype(int)

        in_bounds = (xi >= 0) & (xi < nx) & (yi >= 0) & (yi < ny)
        xi_c = np.clip(xi, 0, nx - 1)
        yi_c = np.clip(yi, 0, ny - 1)
        matches = in_bounds & np.isclose(xs[xi_c], fx) & np.isclose(ys[yi_c], fy)

        visible_indices = list(zip(yi[matches].tolist(), xi[matches].tolist()))

    visible_set = set(visible_indices)
    sensor_rows = []
    sensor_columns = []
    sensor_values = []
    block_keys = []

    for row_index, (yi, xi) in enumerate(visible_indices):
        block_y0 = (yi // block_size) * block_size
        block_x0 = (xi // block_size) * block_size
        block_keys.append((block_y0, block_x0))
        block_indices = []

        for dy in range(block_size):
            for dx in range(block_size):
                block_y = block_y0 + dy
                block_x = block_x0 + dx

                if block_y >= ny or block_x >= nx:
                    continue

                if (block_y, block_x) in visible_set:
                    block_indices.append(block_y * nx + block_x)

        if block_indices:
            weight = 1.0 / len(block_indices)
            sensor_rows.extend([row_index] * len(block_indices))
            sensor_columns.extend(block_indices)
            sensor_values.extend([weight] * len(block_indices))

    sensor = csr_matrix(
        (sensor_values, (sensor_rows, sensor_columns)),
        shape=(len(visible_indices), nx * ny),
        dtype=float,
    )

    if not return_block_ids:
        return sensor

    block_lookup = {
        block_key: block_id
        for block_id, block_key in enumerate(dict.fromkeys(block_keys))
    }
    block_ids = np.asarray([block_lookup[key] for key in block_keys], dtype=int)
    return sensor, block_ids


def build_correlated_noise_covariance(block_ids, variance):
    block_ids = np.asarray(block_ids)
    row_variances = np.asarray(variance, dtype=float)
    if row_variances.ndim == 0:
        row_variances = np.full(len(block_ids), float(row_variances))
    elif row_variances.shape != (len(block_ids),):
        raise ValueError("variance must be scalar or contain one value per sensor row")

    shared_block = block_ids[:, None] == block_ids[None, :]
    covariance = np.sqrt(row_variances[:, None] * row_variances[None, :])
    return np.where(shared_block, covariance, 0.0)


def sample_correlated_sensor_noise(block_ids, variance, rng):
    block_ids = np.asarray(block_ids)
    row_variances = np.asarray(variance, dtype=float)
    if row_variances.ndim == 0:
        row_variances = np.full(len(block_ids), float(row_variances))
    elif row_variances.shape != (len(block_ids),):
        raise ValueError("variance must be scalar or contain one value per sensor row")

    noise = np.zeros(len(block_ids), dtype=float)
    for block_id in np.unique(block_ids):
        rows = np.flatnonzero(block_ids == block_id)
        block_variance = float(row_variances[rows[0]])
        if not np.allclose(row_variances[rows], block_variance):
            raise ValueError("rows in the same sensor block must have equal variance")
        noise[rows] = rng.normal(0.0, np.sqrt(block_variance))
    return noise


def measurement_noise_covariance(R, row_count):
    noise = np.asarray(R, dtype=float)
    if noise.ndim == 0:
        return float(noise) * np.eye(row_count)
    if noise.shape == (row_count,):
        return np.diag(noise)
    if noise.shape == (row_count, row_count):
        return noise
    raise ValueError(
        "R must be scalar, one variance per sensor row, or a full covariance matrix"
    )


def compress_shared_sensor_rows(sensor, z_meas, R, block_ids):
    block_ids = np.asarray(block_ids)
    if block_ids.shape != (sensor.shape[0],):
        raise ValueError("block_ids must contain one id per sensor row")

    _, unique_rows = np.unique(block_ids, return_index=True)
    unique_rows = np.sort(unique_rows)
    compressed_sensor = sensor[unique_rows]
    if not issparse(compressed_sensor):
        compressed_sensor = csr_matrix(compressed_sensor)
    else:
        compressed_sensor = compressed_sensor.tocsr()

    compressed_z = None
    if z_meas is not None:
        compressed_z = np.asarray(z_meas)[unique_rows]

    noise_covariance = measurement_noise_covariance(R, sensor.shape[0])
    compressed_R = noise_covariance[np.ix_(unique_rows, unique_rows)]
    return compressed_sensor, compressed_z, compressed_R


def fov_grid_points(cx, cy, cz, xs, ys, angle_of_view=60.0):
    radius = fov_lateral_radius(cz, angle_of_view)
    visible_xs = np.asarray(xs)[
        (np.asarray(xs) >= cx - radius) & (np.asarray(xs) <= cx + radius)
    ]
    visible_ys = np.asarray(ys)[
        (np.asarray(ys) >= cy - radius) & (np.asarray(ys) <= cy + radius)
    ]
    return [(x, y) for x in visible_xs for y in visible_ys]


def future_sensor_model_3d(
    spline_path_3d,
    xs,
    ys,
    angle_of_view=60.0,
    trajectory_stride=4,
    max_measurements=64,
):
    selected_path = list(spline_path_3d[::trajectory_stride])
    if spline_path_3d and selected_path[-1] != spline_path_3d[-1]:
        selected_path.append(spline_path_3d[-1])

    sensor_blocks = []
    covariance_blocks = []
    for px, py, pz in selected_path:
        cx = float(xs[np.argmin(np.abs(xs - px))])
        cy = float(ys[np.argmin(np.abs(ys - py))])
        fov = fov_grid_points(cx, cy, pz, xs, ys, angle_of_view)
        sensor, block_ids = build_sensor_matrix(
            fov, pz, xs, ys, return_block_ids=True
        )
        if sensor.shape[0] > 0:
            noise_covariance = build_correlated_noise_covariance(
                block_ids, noise_model(pz)
            )
            sensor, _, noise_covariance = compress_shared_sensor_rows(
                sensor, None, noise_covariance, block_ids
            )
            sensor_blocks.append(sensor)
            covariance_blocks.append(noise_covariance)

    if not sensor_blocks:
        return (
            csr_matrix((0, len(xs) * len(ys)), dtype=float),
            np.zeros((0, 0), dtype=float),
        )

    sensor = sparse_vstack(sensor_blocks, format="csr")
    noise_covariance = block_diag(*covariance_blocks)
    if max_measurements is not None and sensor.shape[0] > max_measurements:
        sample_indices = np.linspace(
            0, sensor.shape[0] - 1, max_measurements, dtype=int
        )
        sensor = sensor[sample_indices]
        noise_covariance = noise_covariance[np.ix_(sample_indices, sample_indices)]
    return sensor, noise_covariance


def future_sensor_matrix_3d(
    spline_path_3d,
    xs,
    ys,
    angle_of_view=60.0,
    trajectory_stride=5,
    max_measurements=64,
):
    sensor, _ = future_sensor_model_3d(
        spline_path_3d,
        xs,
        ys,
        angle_of_view=angle_of_view,
        trajectory_stride=trajectory_stride,
        max_measurements=max_measurements,
    )
    return sensor


###CMAES, utility, and trajectory functions


def sampler(cx, cy, X, Y, P_history, samplestep):
    var_field = np.diag(P_history[-1]).reshape(X.shape)

    # gradient in physical coordinates
    gy, gx = np.gradient(var_field, Y[:, 0], X[0, :])
    grad_mag = np.hypot(gx, gy)

    
    mask = grad_mag >= 0.95 * grad_mag.max()
    rows, cols = np.where(mask)

    candidate_x = X[rows, cols]
    candidate_y = Y[rows, cols]

    distances = np.hypot(candidate_x - cx, candidate_y - cy)
    closest_idx = np.argmin(distances)

    target_x = candidate_x[closest_idx]
    target_y = candidate_y[closest_idx]

    dx = target_x - cx
    dy = target_y - cy

    norm = np.hypot(dx, dy)
    if norm > 1e-12:
        dx = dx / norm * samplestep
        dy = dy / norm * samplestep
    else:
        dx, dy = 0.0, 0.0

    return np.array([dx, dy]), (gx, gy), (target_x, target_y)




# importance_filter returns variance form for easy utility deduction and uses
# either the lower or upper confidence bound according to LCB.
def importance_filter(mu, P, beta, threshold = 0.5, eps=1e-12):


    sigma = np.sqrt(np.diag(P))

    # mu_min, mu_max = np.min(mu), np.max(mu)
    # sigma_min, sigma_max = np.min(sigma), np.max(sigma)
    # mu_norm = (mu - mu_min) / max(mu_max - mu_min, eps)
    # sigma_norm = (sigma - sigma_min) / max(sigma_max - sigma_min, eps)
    if LCB == True:
        importance = mu - beta * sigma
        importance_threshold = threshold

        importance_mask = importance <= importance_threshold
    elif LCB == False:
        importance = mu + beta * sigma
        importance_threshold = threshold

        importance_mask = importance >= importance_threshold

    if not np.any(importance_mask):
        importance_mask = np.ones_like(importance, dtype=bool)


    sigma2 = np.diag(P)

    # sigma2_min, sigma2_max = np.min(sigma2), np.max(sigma2)
    # sigma2_norm = (sigma2 - sigma2_min) / max(sigma2_max - sigma2_min, eps)

    utility = np.zeros_like(mu)
    utility[importance_mask] = sigma2[importance_mask]

    return utility


def grid_measure(filtered_utility, xs, ys, margin=None):

    if margin is None:
        margin = step * 2   # match your dynamics buffer

    X, Y = np.meshgrid(xs, ys)

    nx = 8
    ny = 8

    # sample ONLY inside valid region
    x_positions = np.linspace(X.min() + margin, X.max() - margin, nx)
    y_positions = np.linspace(Y.min() + margin, Y.max() - margin, ny)

    selected_points = []
    for x in x_positions:
        xg = X[0, np.argmin(np.abs(X[0, :] - x))]
        for y in y_positions:
            yg = Y[np.argmin(np.abs(Y[:, 0] - y)), 0]
            selected_points.append((xg, yg))

    selected_points = list(dict.fromkeys(selected_points))

    util_values = []
    for xg, yg in selected_points:
        idx = np.where(
            np.isclose(X.ravel(), xg) &
            np.isclose(Y.ravel(), yg)
        )[0]

        if len(idx) == 0:
            continue

        idx = idx[0]
        util_value = filtered_utility[idx]

        # shape preserved exactly
        util_values.append((util_value, (xg, yg)))

    return util_values


def build_pyramid_lattice_3d(xs, ys, zmin, zmax, margin=None): ###changing hardcoded lattice
    if margin is None:
        margin = step * 2

    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    xmin, xmax = xs.min() + margin, xs.max() - margin
    ymin, ymax = ys.min() + margin, ys.max() - margin
    center_x, center_y = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
    half_x, half_y = (xmax - xmin) / 2.0, (ymax - ymin) / 2.0

    def snap_positions(values, grid):
        return [float(grid[np.argmin(np.abs(grid - value))]) for value in values]

    def layer_bounds(inset_frac):
        # inset_frac=0 -> full inset domain (the base of the pyramid); larger
        # inset_frac shrinks the layer's footprint symmetrically toward the
        # center, giving the lattice its tapering pyramid/frustum shape.
        hx = half_x * (1.0 - inset_frac)
        hy = half_y * (1.0 - inset_frac)
        return center_x - hx, center_x + hx, center_y - hy, center_y + hy

    # Insets chosen (rather than plain thirds) so each layer's snapped grid
    # points land on distinct cells - with thirds, the low and top layers'
    # snapped x/y values coincide, putting stacked same-xy candidates back
    # into the lattice despite the tapering footprint.
    # Low tier count bumped 5->9 per axis: at z=zmin the FOV footprint side
    # (~2*zmin*tan(30 deg)) is roughly half this tier's 5-point spacing,
    # leaving real gaps between candidate footprints - the densest tier is
    # the one place the coarse lattice was actually under-resolved relative
    # to what the sensor can see. Middle/top tiers are left alone: their
    # footprints already roughly tile (middle) or overlap (top), matching
    # Popovic et al. (2020)'s own "sparser at top due to increasing FoV"
    # design (Fig. 5) rather than an oversight to fix.
    low_xmin, low_xmax, low_ymin, low_ymax = layer_bounds(0.0)
    low_x = snap_positions(np.linspace(low_xmin, low_xmax, 9), xs)
    low_y = snap_positions(np.linspace(low_ymin, low_ymax, 9), ys)

    middle_xmin, middle_xmax, middle_ymin, middle_ymax = layer_bounds(0.30)
    middle_x = snap_positions(np.linspace(middle_xmin, middle_xmax, 3), xs)
    middle_y = snap_positions(np.linspace(middle_ymin, middle_ymax, 3), ys)

    top_xmin, top_xmax, top_ymin, top_ymax = layer_bounds(0.60)
    top_x = snap_positions(np.linspace(top_xmin, top_xmax, 2), xs)
    top_y = snap_positions(np.linspace(top_ymin, top_ymax, 2), ys)

    # dynamics_3d snaps cz to the nearest multiple of `step` every timestep
    # (cz = step * round(cz / step)), so an unsnapped middle z-tier (e.g. the
    # z=25 midpoint of a 10..40 range) can be approached but never landed on
    # exactly - "reached" would never become literally true, and a
    # single-step planner re-querying candidates from that stuck position
    # would keep re-selecting the same unreachable tier forever. Snap the
    # tiers onto the same step-spaced z grid dynamics_3d actually reaches,
    # the same way the xy layers above are already snapped onto xs/ys.
    zs = np.arange(float(zmin), float(zmax) + 1e-9, step)
    z_layers = snap_positions(np.linspace(float(zmin), float(zmax), 3), zs)
    lattice = [(x, y, float(z_layers[0])) for x in low_x for y in low_y]
    lattice.extend((x, y, float(z_layers[1])) for x in middle_x for y in middle_y)
    lattice.extend((x, y, float(z_layers[2])) for x in top_x for y in top_y)
    return list(dict.fromkeys(lattice))


def covariance_after_sensor(P, sensor, R):
    if sensor.shape[0] == 0:
        return P.copy()

    noise_covariance = measurement_noise_covariance(R, sensor.shape[0])

    projected_cov = np.asarray(sensor @ P)
    innovation_cov = np.asarray(sensor @ projected_cov.T) + noise_covariance
    try:
        # innovation_cov = sensor @ P @ sensor.T + R is symmetric
        # positive-definite (covariance + noise), so Cholesky beats LU here.
        solved = spd_solve(innovation_cov, projected_cov, assume_a="pos")
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(innovation_cov) @ projected_cov

    posterior = P - projected_cov.T @ solved
    return 0.5 * (posterior + posterior.T)


def _select_argmax_candidate(scored_candidates, rng):
    return max(scored_candidates, key=lambda item: item[0])


def _select_softmax_candidate(scored_candidates, rng, temperature):
    scores = np.asarray([item[0] for item in scored_candidates], dtype=float)
    score_range = scores.max() - scores.min()
    normalized = (scores - scores.min()) / max(score_range, 1e-12)
    scaled = normalized / max(float(temperature), 1e-12)
    scaled -= scaled.max()  # numerical stability, doesn't change the resulting probabilities
    weights = np.exp(scaled)
    probabilities = weights / weights.sum()
    choice_idx = rng.choice(len(scored_candidates), p=probabilities)
    return scored_candidates[choice_idx]


def _select_tiebreak_candidate(scored_candidates, rng, tie_tolerance):
    scores = np.asarray([item[0] for item in scored_candidates], dtype=float)
    score_range = scores.max() - scores.min()
    threshold = scores.max() - tie_tolerance * max(score_range, 1e-12)
    tied_indices = np.flatnonzero(scores >= threshold)
    choice_idx = tied_indices[rng.integers(len(tied_indices))]
    return scored_candidates[choice_idx]


def _grid_search_3d_impl(
    mu,
    P,
    xs,
    ys,
    start_pose,
    beta,
    utility_threshold,
    planning_horizon,
    zmin,
    zmax,
    select_fn,
    rng,
    alpha=0.02,
    angle_of_view=60.0,
    max_measurements=64,
):
    available = build_pyramid_lattice_3d(xs, ys, zmin, zmax)
    simulated_covariance = P.copy()
    current_pose = np.asarray(start_pose, dtype=float)
    selected_waypoints = []

    for _ in range(min(planning_horizon, len(available))):
        utility = importance_filter(
            mu, simulated_covariance, beta, threshold=utility_threshold
        )
        importance_mask = utility > 0
        scored_candidates = []

        for candidate in available:
            sensor, measurement_noise = future_sensor_model_3d(
                [candidate],
                xs,
                ys,
                angle_of_view=angle_of_view,
                trajectory_stride=1,
                max_measurements=max_measurements,
            )
            gain = masked_expected_variance_reduction_from_sensor(
                simulated_covariance,
                importance_mask,
                sensor,
                measurement_noise,
            )
            distance = float(
                np.linalg.norm(np.asarray(candidate, dtype=float) - current_pose)
            )
            score = gain / (max(distance, step) ** COST_EXPONENT)
            scored_candidates.append((score, candidate, sensor, measurement_noise))

        _, best, best_sensor, best_variance = select_fn(scored_candidates, rng)
        selected_waypoints.append(best)
        available.remove(best)
        simulated_covariance = covariance_after_sensor(
            simulated_covariance, best_sensor, best_variance
        )
        current_pose = np.asarray(best, dtype=float)

    return selected_waypoints


def grid_search_3d(
    mu,
    P,
    xs,
    ys,
    start_pose,
    beta,
    utility_threshold,
    planning_horizon,
    zmin,
    zmax,
    alpha=0.02,
    angle_of_view=60.0,
    max_measurements=64,
):
    """Deterministic version: greedy argmax over candidate waypoints every step."""
    return _grid_search_3d_impl(
        mu,
        P,
        xs,
        ys,
        start_pose,
        beta,
        utility_threshold,
        planning_horizon,
        zmin,
        zmax,
        select_fn=_select_argmax_candidate,
        rng=None,
        alpha=alpha,
        angle_of_view=angle_of_view,
        max_measurements=max_measurements,
    )


def grid_search_3d_softmax(
    mu,
    P,
    xs,
    ys,
    start_pose,
    beta,
    utility_threshold,
    planning_horizon,
    zmin,
    zmax,
    alpha=0.02,
    angle_of_view=60.0,
    max_measurements=64,
    temperature=GRID_SEARCH_SOFTMAX_TEMPERATURE,
    seed=GRID_SEARCH_SOFTMAX_SEED,
):
    """Stochastic version: samples each step's waypoint from a softmax over
    candidate scores instead of taking the argmax, so repeated calls (and
    therefore the warm start handed to CMA-ES) can diverge into different
    trajectories. Drop-in replacement for grid_search_3d - same signature
    plus temperature/seed."""
    rng = np.random.default_rng(seed)

    def select_fn(scored_candidates, rng):
        return _select_softmax_candidate(scored_candidates, rng, temperature)

    return _grid_search_3d_impl(
        mu,
        P,
        xs,
        ys,
        start_pose,
        beta,
        utility_threshold,
        planning_horizon,
        zmin,
        zmax,
        select_fn=select_fn,
        rng=rng,
        alpha=alpha,
        angle_of_view=angle_of_view,
        max_measurements=max_measurements,
    )


def grid_search_3d_tiebreak(
    mu,
    P,
    xs,
    ys,
    start_pose,
    beta,
    utility_threshold,
    planning_horizon,
    zmin,
    zmax,
    alpha=0.02,
    angle_of_view=60.0,
    max_measurements=64,
    tie_tolerance=GRID_SEARCH_TIE_TOLERANCE,
    seed=GRID_SEARCH_TIEBREAK_SEED,
):
    """Stochastic version that only randomizes among candidates within
    tie_tolerance of the top score each step. Unlike grid_search_3d_softmax,
    this never trades away score for diversity - it only resolves genuine (or
    near-floating-point) ties that argmax would otherwise always break the same
    way. Drop-in replacement for grid_search_3d - same signature plus
    tie_tolerance/seed."""
    rng = np.random.default_rng(seed)

    def select_fn(scored_candidates, rng):
        return _select_tiebreak_candidate(scored_candidates, rng, tie_tolerance)

    return _grid_search_3d_impl(
        mu,
        P,
        xs,
        ys,
        start_pose,
        beta,
        utility_threshold,
        planning_horizon,
        zmin,
        zmax,
        select_fn=select_fn,
        rng=rng,
        alpha=alpha,
        angle_of_view=angle_of_view,
        max_measurements=max_measurements,
    )


def next_best_waypoint(grid_util_values, curr_x, curr_y, alpha = 0.1):
    scored = []
    for util, (x, y) in grid_util_values:
        dist = np.hypot(x - curr_x, y - curr_y)
        if dist == 0:
            dist = 1e-6
        score = util * np.exp(-alpha * dist)
        scored.append((score, (x, y)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]
    
def flatten_waypoints(waypoints):
    return np.array([coord for pt in waypoints for coord in pt], dtype=float)

def flatten_waypoints_3d(waypoints_3d):
    waypoints_3d = np.asarray(waypoints_3d, dtype=float)
    if waypoints_3d.size == 0:
        return np.array([], dtype=float)
    if waypoints_3d.ndim != 2 or waypoints_3d.shape[1] != 3:
        raise ValueError("3D waypoints must have shape (N, 3).")
    return waypoints_3d.reshape(-1)


def unflatten_waypoints(z):
    return [(z[i], z[i + 1]) for i in range(0, len(z), 2)]

def unflatten_waypoints_3d(values):
    values = np.asarray(values, dtype=float)
    if values.size % 3 != 0:
        raise ValueError("A flattened 3D waypoint array must contain a multiple of 3 values.")
    return [tuple(point) for point in values.reshape(-1, 3)]


def clip_waypoints_continuous(waypoints, xs, ys, margin=None):
    if margin is None:
        margin = step * 2

    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    clipped = []
    for x, y in waypoints:
        x = np.clip(x, xmin + margin, xmax - margin)
        y = np.clip(y, ymin + margin, ymax - margin)
        clipped.append((x, y))

    return clipped


def clip_waypoints_continuous_3d(waypoints_3d, xs, ys, zmin, zmax, margin=None):
    if margin is None:
        margin = step * 2

    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    return [
        (
            float(np.clip(x, xmin + margin, xmax - margin)),
            float(np.clip(y, ymin + margin, ymax - margin)),
            float(np.clip(z, zmin, zmax)),
        )
        for x, y, z in waypoints_3d
    ]



def snap_waypoints_to_grid(waypoints, xs, ys):
    snapped = []
    xs = np.asarray(xs)
    ys = np.asarray(ys)

    for x, y in waypoints:
        x_snap = xs[np.argmin(np.abs(xs - x))]
        y_snap = ys[np.argmin(np.abs(ys - y))]
        snapped.append((float(x_snap), float(y_snap)))

    return snapped


def future_observation_indices(spline_path, xs, ys, lateral_coverage, trajectory_stride=10, max_obs=64):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    nx = len(xs)
    selected_path = list(spline_path[::trajectory_stride])
    if spline_path and selected_path[-1] != spline_path[-1]:
        selected_path.append(spline_path[-1])

    obs_indices = []
    for px, py in selected_path:
        cx = xs[np.argmin(np.abs(xs - px))]
        cy = ys[np.argmin(np.abs(ys - py))]

        for x in np.arange(cx - lateral_coverage, cx + lateral_coverage + 1e-9, step):
            for y in np.arange(cy - lateral_coverage, cy + lateral_coverage + 1e-9, step):
                xi = int(np.argmin(np.abs(xs - x)))
                yi = int(np.argmin(np.abs(ys - y)))
                if np.isclose(xs[xi], x) and np.isclose(ys[yi], y):
                    obs_indices.append(yi * nx + xi)

    obs_indices = list(dict.fromkeys(obs_indices))
    if len(obs_indices) > max_obs:
        sample_idx = np.linspace(0, len(obs_indices) - 1, max_obs, dtype=int)
        obs_indices = [obs_indices[i] for i in sample_idx]

    return np.array(obs_indices, dtype=int)


def masked_expected_variance_reduction(P, mask, obs_indices, R):
    if len(obs_indices) == 0:
        return 0.0

    mask_indices = np.flatnonzero(mask)
    if len(mask_indices) == 0:
        mask_indices = np.arange(P.shape[0])

    S = P[np.ix_(obs_indices, obs_indices)] + R * np.eye(len(obs_indices))
    cross_cov = P[np.ix_(mask_indices, obs_indices)]

    try:
        # S = P[obs,obs] + R*I is symmetric positive-definite (covariance +
        # noise), so a Cholesky-based solve is ~2x faster than general LU.
        solved = spd_solve(S, cross_cov.T, assume_a="pos")
        reduction = np.sum(cross_cov.T * solved, axis=0)
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(S) @ cross_cov.T
        reduction = np.sum(cross_cov.T * solved, axis=0)

    prior_diag = np.diag(P)[mask_indices]
    reduction = np.clip(reduction, 0.0, prior_diag)
    return float(np.sum(reduction))


def masked_expected_variance_reduction_from_sensor(P, mask, sensor, R):
    if sensor.shape[0] == 0:
        return 0.0

    mask_indices = np.flatnonzero(mask)
    if len(mask_indices) == 0:
        mask_indices = np.arange(P.shape[0])

    projected_cov = np.asarray(sensor @ P)
    innovation_cov = (
        np.asarray(sensor @ projected_cov.T)
        + measurement_noise_covariance(R, sensor.shape[0])
    )
    cross_cov = projected_cov[:, mask_indices].T

    try:
        # innovation_cov = sensor @ P @ sensor.T + R is symmetric
        # positive-definite (covariance + noise), so Cholesky beats LU here.
        solved = spd_solve(innovation_cov, cross_cov.T, assume_a="pos")
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(innovation_cov) @ cross_cov.T

    reduction = np.sum(cross_cov * solved.T, axis=1)
    prior_diag = np.diag(P)[mask_indices]
    reduction = np.clip(reduction, 0.0, prior_diag)
    return float(np.sum(reduction))


def trajectory_objective(
    z,
    mu,
    P,
    xs,
    ys,
    start_x,
    start_y,
    beta,
    utility_threshold,
    R=None,
    lateral_coverage=None,
    predictive_variance=False,
    cached_utility=None,
    cached_importance_mask=None,
):
    raw_control_waypoints = unflatten_waypoints(z)
    control_waypoints = clip_waypoints_continuous(raw_control_waypoints, xs, ys)

    spline_path = build_spline_trajectory(
        start_x, start_y, control_waypoints, samples_per_segment=5
    )

    utility = cached_utility
    if utility is None:
        utility = importance_filter(mu, P, beta, threshold=utility_threshold)

    total_score = 0.0
    total_distance = 0.0
    curr_x, curr_y = start_x, start_y
    margin = step * 2
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)

    for wx, wy in raw_control_waypoints:
        if wx < xmin + margin or wx > xmax - margin or wy < ymin + margin or wy > ymax - margin:
            total_score -= 1000.0

    for px, py in spline_path:
        if px < xmin + margin or px > xmax - margin or py < ymin + margin or py > ymax - margin:
            total_score -= 1000.0
        total_distance += np.hypot(px - curr_x, py - curr_y)
        curr_x, curr_y = px, py

    if predictive_variance:
        if R is None or lateral_coverage is None:
            raise ValueError("R and lateral_coverage are required for predictive variance scoring.")

        importance_mask = cached_importance_mask
        if importance_mask is None:
            importance_mask = utility > 0
        obs_indices = future_observation_indices(spline_path, xs, ys, lateral_coverage)
        variance_reduction = masked_expected_variance_reduction(P, importance_mask, obs_indices, R)
        total_score += variance_reduction
        total_score -= 0.01 * total_distance
        return -total_score

    X, Y = np.meshgrid(xs, ys)
    curr_x, curr_y = start_x, start_y
    for px, py in spline_path:
        idx = np.argmin((X.ravel() - px) ** 2 + (Y.ravel() - py) ** 2)
        total_score += utility[idx]
        total_score -= 0.01 * np.hypot(px - curr_x, py - curr_y)
        curr_x, curr_y = px, py

    return -total_score


def trajectory_objective_3d(
    values,
    mu,
    P,
    xs,
    ys,
    start_x,
    start_y,
    start_z,
    beta,
    utility_threshold,
    zmin,
    zmax,
    predictive_variance=True,
    cached_utility=None,
    cached_importance_mask=None,
):
    raw_control_waypoints = unflatten_waypoints_3d(values)
    control_waypoints = clip_waypoints_continuous_3d(
        raw_control_waypoints, xs, ys, zmin, zmax
    )
    spline_path_3d = build_spline_trajectory_3d(
        start_x,
        start_y,
        start_z,
        control_waypoints,
        samples_per_segment=5,
    )
    spline_path_2d = [(x, y) for x, y, _ in spline_path_3d]

    utility = cached_utility
    if utility is None:
        utility = importance_filter(mu, P, beta, threshold=utility_threshold)

    margin = step * 2
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    penalty = 0.0

    def _boundary_overshoot(x, y, z):
        # Magnitude of the worst single boundary violation at this point, 0
        # if in bounds - mirrors Popovic et al. (2020)'s actual
        # compute_objective.m penalty (a continuous max(overshoot, 0) ramp
        # over sampled path points) rather than a flat per-violation
        # constant, so CMA-ES gets a gradient back toward feasibility
        # instead of every infeasible candidate looking equally (and
        # uninformatively) bad.
        return max(
            x - (xmax - margin),
            (xmin + margin) - x,
            y - (ymax - margin),
            (ymin + margin) - y,
            z - zmax,
            zmin - z,
            0.0,
        )

    # Checked on the raw (pre-clip) waypoints so a wildly out-of-bounds
    # proposal is scored worse than a barely-out-of-bounds one, before
    # clip_waypoints_continuous_3d collapses both to the same boundary
    # point above.
    for x, y, z in raw_control_waypoints:
        penalty -= _boundary_overshoot(x, y, z)

    # Also checked on the sampled (post-clip) spline path, matching
    # Popovic's points_meas check - catches a spline bowing outside the
    # domain between two already-in-bounds control waypoints, which the
    # vertex-only check above can't see.
    for x, y, z in spline_path_3d:
        penalty -= _boundary_overshoot(x, y, z)

    if predictive_variance:
        importance_mask = cached_importance_mask
        if importance_mask is None:
            importance_mask = utility > 0
        sensor, measurement_variances = future_sensor_model_3d(
            spline_path_3d,
            xs,
            ys,
        )
        variance_reduction = masked_expected_variance_reduction_from_sensor(
            P, importance_mask, sensor, measurement_variances
        )
        # Cost-normalize the gain, mirroring Popovic et al. (2020)'s actual
        # reference implementation (github.com/marija-p/mav_ipp,
        # tools/planning/compute_objective.m: obj = -gain/cost + penalty,
        # cost = trajectory time) rather than the paper's prose description,
        # which omits this term. Without it, CMA-ES has no reason to prefer
        # a trajectory reachable in less total travel distance over one that
        # eked out marginally more raw variance reduction by roaming further
        # - unlike the grid search (score = gain / max(distance, step)
        # above), which already penalizes distance per step. total_distance
        # here is the same start-through-waypoints path length
        # build_spline_trajectory_3d itself sums into its spline parameter
        # t, so this mirrors Popovic's single whole-trajectory cost term
        # (get_trajectory_total_time) rather than the grid search's
        # per-step one.
        path_points = np.asarray(
            [(start_x, start_y, start_z)] + list(control_waypoints), dtype=float
        )
        total_distance = float(
            np.sum(np.linalg.norm(np.diff(path_points, axis=0), axis=1))
        )
        cost = max(total_distance, step) ** COST_EXPONENT
        total_score = penalty + variance_reduction / cost
    else:
        total_score = penalty
        X, Y = np.meshgrid(xs, ys)
        for x, y in spline_path_2d:
            idx = np.argmin((X.ravel() - x) ** 2 + (Y.ravel() - y) ** 2)
            total_score += utility[idx]

    return -total_score


def cma_es_refine_waypoints(
    initial_waypoints,
    mu,
    P,
    xs,
    ys,
    cx,
    cy,
    beta,
    utility_threshold,
    R=None,
    lateral_coverage=None,
    predictive_variance=False,
    seed=CMA_SEED,
):
    x0 = flatten_waypoints(initial_waypoints)
    sigma0 = 4.0
    margin = step * 2
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    lower_bounds = []
    upper_bounds = []
    for _ in initial_waypoints:
        lower_bounds.extend([xmin + margin, ymin + margin])
        upper_bounds.extend([xmax - margin, ymax - margin])

    maxiter = 18 if predictive_variance else 40
    popsize = 8 if predictive_variance else 12
    cached_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    cached_importance_mask = cached_utility > 0

    es = cma.CMAEvolutionStrategy(
    x0,
    sigma0,
    {
        "bounds": [lower_bounds, upper_bounds],
        "maxiter": maxiter,
        "popsize": popsize,
        "seed": seed,
        "verb_disp": 0,
        "verb_log": 0,
    })

    while not es.stop():
        solutions = es.ask()
        values = [
            trajectory_objective(
                sol,
                mu,
                P,
                xs,
                ys,
                cx,
                cy,
                beta,
                utility_threshold,
                R=R,
                lateral_coverage=lateral_coverage,
                predictive_variance=predictive_variance,
                cached_utility=cached_utility,
                cached_importance_mask=cached_importance_mask,
            )
            for sol in solutions
        ]
        es.tell(solutions, values)

    best = es.result.xbest
    best_waypoints = clip_waypoints_continuous(unflatten_waypoints(best), xs, ys, margin=margin)
    return snap_waypoints_to_grid(best_waypoints, xs, ys)


def cma_es_refine_waypoints_3d(
    initial_waypoints,
    mu,
    P,
    xs,
    ys,
    cx,
    cy,
    cz,
    beta,
    utility_threshold,
    zmin,
    zmax,
    predictive_variance=True,
    maxiter=None,
    popsize=None,
    maxfevals=None,
    seed=CMA_SEED,
):
    margin = step * 2
    xmin, xmax = np.min(xs), np.max(xs)
    ymin, ymax = np.min(ys), np.max(ys)
    initial_waypoints = clip_waypoints_continuous_3d(
        initial_waypoints,
        xs,
        ys,
        zmin,
        zmax,
        margin=margin,
    )
    x0 = flatten_waypoints_3d(initial_waypoints)
    # sigma0 is a nominal base of 1.0; the actual per-axis step size is carried
    # by CMA_stds below (x/y get CMA_STEP_SIZE_XY, z gets CMA_STEP_SIZE_Z).
    sigma0 = 1.0
    cma_stds = np.tile([CMA_STEP_SIZE_XY, CMA_STEP_SIZE_XY, CMA_STEP_SIZE_Z], len(initial_waypoints))

    lower_bounds = []
    upper_bounds = []
    for _ in initial_waypoints:
        lower_bounds.extend([xmin + margin, ymin + margin, zmin])
        upper_bounds.extend([xmax - margin, ymax - margin, zmax])

    if maxiter is None:
        maxiter = CMA_PREDICTIVE_MAXITER if predictive_variance else 40
    if popsize is None:
        popsize = CMA_PREDICTIVE_POPSIZE if predictive_variance else 12
    if maxfevals is None and predictive_variance:
        maxfevals = CMA_PREDICTIVE_MAXFEVALS

    cma_options = {
        "bounds": [lower_bounds, upper_bounds],
        "maxiter": maxiter,
        "popsize": popsize,
        "seed": seed,
        "verb_disp": 0,
        "verb_log": 0,
        "CMA_stds": cma_stds,
    }
    if maxfevals is not None:
        cma_options["maxfevals"] = maxfevals

    cached_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    cached_importance_mask = cached_utility > 0

    es = cma.CMAEvolutionStrategy(
        x0,
        sigma0,
        cma_options,
    )

    while not es.stop() and (
        maxfevals is None or es.countevals < maxfevals
    ):
        solutions = es.ask()
        values = [
            trajectory_objective_3d(
                solution,
                mu,
                P,
                xs,
                ys,
                cx,
                cy,
                cz,
                beta,
                utility_threshold,
                zmin,
                zmax,
                predictive_variance=predictive_variance,
                cached_utility=cached_utility,
                cached_importance_mask=cached_importance_mask,
            )
            for solution in solutions
        ]
        es.tell(solutions, values)

    return clip_waypoints_continuous_3d(
        unflatten_waypoints_3d(es.result.xbest),
        xs,
        ys,
        zmin,
        zmax,
        margin=margin,
    )




def build_spline_trajectory(cx, cy, control_waypoints, samples_per_segment=10):
    waypoints = np.array([(cx, cy)] + list(control_waypoints), dtype=float)

    if len(waypoints) < 2:
        return [tuple(waypoints[0])]

    deltas = np.diff(waypoints, axis=0)
    seg_lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    t = np.concatenate([[0.0], np.cumsum(seg_lengths)])

    # Remove duplicate points that create zero-length segments
    keep = np.concatenate([[True], np.diff(t) > 1e-9])
    waypoints = waypoints[keep]
    t = t[keep]

    if len(t) < 2:
        return [tuple(waypoints[0])]

    cs_x = CubicSpline(t, waypoints[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, waypoints[:, 1], bc_type="natural")

    trajectory = []
    for i in range(len(t) - 1):
        t_segment = np.linspace(t[i], t[i + 1], samples_per_segment, endpoint=False)
        for ts in t_segment:
            trajectory.append((float(cs_x(ts)), float(cs_y(ts))))

    trajectory.append((float(waypoints[-1, 0]), float(waypoints[-1, 1])))
    return trajectory


def build_spline_trajectory_3d(
    cx,
    cy,
    cz,
    control_waypoints,
    samples_per_segment=10,
):
    waypoints = np.asarray(
        [(cx, cy, cz)] + list(control_waypoints),
        dtype=float,
    )

    deltas = np.diff(waypoints, axis=0)
    segment_lengths = np.linalg.norm(deltas, axis=1)
    t = np.concatenate([[0.0], np.cumsum(segment_lengths)])

    keep = np.concatenate([[True], np.diff(t) > 1e-9])
    waypoints = waypoints[keep]
    t = t[keep]

    if len(t) < 2:
        return [tuple(waypoints[0])]

    cs_x = CubicSpline(t, waypoints[:, 0], bc_type="natural")
    cs_y = CubicSpline(t, waypoints[:, 1], bc_type="natural")
    cs_z = CubicSpline(t, waypoints[:, 2], bc_type="natural")

    trajectory = []
    for i in range(len(t) - 1):
        sample_times = np.linspace(
            t[i], t[i + 1],
            samples_per_segment,
            endpoint=False,
        )

        trajectory.extend(
            (float(cs_x(s)), float(cs_y(s)), float(cs_z(s)))
            for s in sample_times
        )

    trajectory.append(tuple(waypoints[-1]))
    return trajectory







def utility_function(mu, P, threshold, beta, eps=1e-12):
    sigma = np.sqrt(np.diag(P))
    beta = np.clip(beta, 0.0, 1.0)

    # Keep mean and uncertainty on comparable scales so beta controls the
    # tradeoff instead of whichever term happens to have the larger magnitude.
    mu_min, mu_max = np.min(mu), np.max(mu)
    sigma_min, sigma_max = np.min(sigma), np.max(sigma)

    mu_norm = (mu - mu_min) / max(mu_max - mu_min, eps)
    sigma_norm = (sigma - sigma_min) / max(sigma_max - sigma_min, eps)

    utility = ((1 - beta) * mu_norm) + (beta * sigma_norm) - threshold
    return np.asarray(utility)

def kalman_update(mu, P, sensor, z_meas, R, block_ids=None):
    if block_ids is not None:
        sensor, z_meas, R = compress_shared_sensor_rows(
            sensor, z_meas, R, block_ids
        )

    v = z_meas - (sensor @ mu)

    projected_cov = np.asarray(sensor @ P)
    S = np.asarray(sensor @ projected_cov.T) + measurement_noise_covariance(
        R, sensor.shape[0]
    )
    # Cholesky-first (S is SPD by construction), pinv fallback only on
    # failure - same pattern as covariance_after_sensor and the
    # masked_expected_variance_reduction* functions above, which this
    # function had been the one holdout from.
    try:
        solved = spd_solve(S, projected_cov, assume_a="pos")
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(S) @ projected_cov
    K = solved.T

    mu = mu + (K @ v).flatten()
    P = P - K @ projected_cov

    return mu, P


# NAIP-fitted kernel scale (must match GP_KERNEL_SIGMA2/GP_KERNEL_LENGTHSCALE in the data
# collector so planners see the same prior the training data was collected under).
# GRF-era defaults were sigma2=0.05, lengthscale=6.08.
def initialize_gp(sigma2=0.0079, lengthscale=4.78, xmin=0.0, xmax=100.0, ymin=0.0, ymax=100.0):
    kernel = ConstantKernel(
        sigma2, constant_value_bounds="fixed"
    ) * Matern(
        length_scale=lengthscale,
        length_scale_bounds="fixed",
        nu=1.5
    )

    gp = GaussianProcessRegressor(kernel=kernel, optimizer=None)

    xs = np.arange(xmin, xmax + 1e-9, step)
    ys = np.arange(ymin, ymax + 1e-9, step)

    X, Y = np.meshgrid(xs, ys) #X, Y are 2D arrays of shape (len(ys), len(xs)) and represent the grid of points in the 2D space
    X_test = np.column_stack([X.ravel(), Y.ravel()])

    # initialization
    mean, cov = gp.predict(X_test, return_cov=True)

    return gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step


def create_plots_and_gifs(path, mu_history, P_history, step_numbers, grad_history, pos_history, utility_history, sorted_util_values_list, X, Y, xs, ys, cx, cy, lateral_coverage, xmin, xmax, ymin, ymax, plot_utility=True, plot_grad=True, planned_path_history=None, control_waypoint_history=None, pose_history=None, angle_of_view=60.0):
    def footprint_for_frame(index=None):
        if pose_history:
            if index is None:
                px, py, pz = pose_history[-1]
            else:
                px, py, pz = pose_history[min(index, len(pose_history) - 1)]
            return float(px), float(py), fov_lateral_radius(pz, angle_of_view)

        if index is None:
            return float(cx), float(cy), float(lateral_coverage)

        px, py = pos_history[min(index, len(pos_history) - 1)]
        return float(px), float(py), float(lateral_coverage)

    def add_fov_rectangle(ax, px, py, radius, edgecolor):
        ax.add_patch(
            Rectangle(
                (px - radius, py - radius),
                2 * radius,
                2 * radius,
                linewidth=2,
                edgecolor=edgecolor,
                facecolor="none",
            )
        )

    def overlay_planned_path(ax, path_points):
        if not path_points:
            return
        path_arr = np.asarray(path_points, dtype=float)
        if path_arr.ndim != 2 or path_arr.shape[1] != 2:
            return
        ax.plot(path_arr[:, 0], path_arr[:, 1], color="cyan", linewidth=1.5, alpha=0.9)

    def overlay_control_waypoints(ax, control_points):
        if not control_points:
            return
        control_arr = np.asarray(control_points, dtype=float)
        if control_arr.ndim != 2 or control_arr.shape[1] != 2:
            return
        ax.scatter(
            control_arr[:, 0],
            control_arr[:, 1],
            c="magenta",
            s=28,
            marker="x",
            linewidths=1.5,
            alpha=0.95,
        )

    def append_current_figure(writer):
        fig = plt.gcf()
        fig.canvas.draw()
        width, height = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(height, width, 4)
        writer.append_data(frame[:, :, :3])

    utility_vmin = 0.0
    utility_vmax = max(float(np.max(u)) for u in utility_history) if utility_history else 1.0
    if utility_vmax <= utility_vmin:
        utility_vmax = utility_vmin + 1.0
    frame_stride = 1
    video_figsize = (5, 4)

    # variance plots
    P_before = np.diag(P_history[0]).reshape(X.shape)
    P_after = np.diag(P_history[-1]).reshape(X.shape)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im0 = ax[0].imshow(
        P_before,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal"
    )
    ax[0].set_title("Variance before update")
    plt.colorbar(im0, ax=ax[0])

    im1 = ax[1].imshow(
        P_after,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal"
    )
    ax[1].set_title("Variance after update")
    plt.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.show()

    # mean plots
    mu_before = mu_history[0]
    mu_after = mu_history[-1]

    Z_before = mu_before.reshape(len(ys), len(xs))
    Z_after = mu_after.reshape(len(ys), len(xs))

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    im0 = ax[0].imshow(
        Z_before,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal"
    )
    ax[0].set_title("Mean before update")

    im1 = ax[1].imshow(
        Z_after,
        extent=[xmin, xmax, ymin, ymax],
        origin="lower",
        aspect="equal"
    )

    final_x, final_y, final_radius = footprint_for_frame()
    add_fov_rectangle(ax[1], final_x, final_y, final_radius, "red")
    ax[1].plot(final_x, final_y, "wo", markersize=6)
    if planned_path_history:
        overlay_planned_path(ax[1], planned_path_history[-1])
    if control_waypoint_history:
        overlay_control_waypoints(ax[1], control_waypoint_history[-1])

    ax[1].set_title("Mean after update")
    plt.colorbar(im1, ax=ax[1])

    plt.tight_layout()
    plt.show()

    # utility plots
    if plot_utility:
        util_before = utility_history[0]
        util_after = utility_history[-1]

        U_before = util_before.reshape(len(ys), len(xs))
        U_after = util_after.reshape(len(ys), len(xs))

        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        im0 = ax[0].imshow(
            U_before,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            aspect="equal",
            cmap="viridis",
            vmin=utility_vmin,
            vmax=utility_vmax,
        )
        ax[0].set_title("Masked Information Gain Before Update")
        plt.colorbar(im0, ax=ax[0], label="Masked information gain")

        im1 = ax[1].imshow(
            U_after,
            extent=[xmin, xmax, ymin, ymax],
            origin="lower",
            aspect="equal",
            cmap="viridis",
            vmin=utility_vmin,
            vmax=utility_vmax,
        )

        final_x, final_y, final_radius = footprint_for_frame()
        add_fov_rectangle(ax[1], final_x, final_y, final_radius, "red")
        ax[1].plot(final_x, final_y, "wo", markersize=6)
        if planned_path_history:
            overlay_planned_path(ax[1], planned_path_history[-1])
        if control_waypoint_history:
            overlay_control_waypoints(ax[1], control_waypoint_history[-1])

        ax[1].set_title("Masked Information Gain After Update")
        plt.colorbar(im1, ax=ax[1], label="Masked information gain")

        plt.tight_layout()
        plt.show()

    vmin = min(m.min() for m in mu_history)
    vmax = max(m.max() for m in mu_history)
    mp4_path = os.path.join(path, "gp_mean_evolution.mp4")
    with imageio.get_writer(mp4_path, fps=4, codec='libx264') as writer:
        for i, (mu_snap, step_num) in enumerate(zip(mu_history, step_numbers)):
            if i % frame_stride != 0:
                continue

            mu_field = mu_snap.reshape(X.shape)

            plt.figure(figsize=video_figsize)
            plt.imshow(
                mu_field,
                origin="lower",
                extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                aspect="equal",
                cmap="hot",
                vmin=vmin,
                vmax=vmax
            )

            px, py, frame_radius = footprint_for_frame(i)
            plt.plot(px, py, "wo", markersize=5)
            add_fov_rectangle(plt.gca(), px, py, frame_radius, "cyan")
            if planned_path_history and i < len(planned_path_history):
                overlay_planned_path(plt.gca(), planned_path_history[i])
            if control_waypoint_history and i < len(control_waypoint_history):
                overlay_control_waypoints(plt.gca(), control_waypoint_history[i])

            plt.colorbar(label="GP mean")
            plt.title(f"GP mean after {step_num} measurements")
            plt.xlabel("x")
            plt.ylabel("y")
            append_current_figure(writer)
            plt.close()

    print(f"MP4 saved to: {mp4_path}")

    # GIF generation for utility
    if plot_utility:
        print(f"Generating utility MP4 with {len(utility_history)} frames, sorted_util_values_list has {len(sorted_util_values_list)} entries")

        util_mp4_path = os.path.join(path, "gp_utility_evolution.mp4")
        with imageio.get_writer(util_mp4_path, fps=4, codec='libx264') as writer:
            for i, (util_snap, step_num) in enumerate(zip(utility_history, step_numbers)):
                if i % frame_stride != 0:
                    continue

                util_field = util_snap.reshape(X.shape)

                plt.figure(figsize=video_figsize)
                plt.imshow(
                    util_field,
                    origin="lower",
                    extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                    aspect="equal",
                    cmap="viridis",
                    vmin=utility_vmin,
                    vmax=utility_vmax
                )

                print(f"Frame {i}: len(sorted_util_values_list)={len(sorted_util_values_list[0])}")
                if i < len(sorted_util_values_list) and isinstance(sorted_util_values_list[i], list) and len(sorted_util_values_list[i]) > 0 and isinstance(sorted_util_values_list[i][0], tuple):
                    # print(f"Plotting {len(sorted_util_values_list[i])} points")
                    for util, (x, y) in sorted_util_values_list[i]:
                        plt.scatter(x, y, c='yellow', s=35, edgecolors='black', alpha=0.9)

                px, py, frame_radius = footprint_for_frame(i)
                plt.plot(px, py, "wo", markersize=5)
                add_fov_rectangle(plt.gca(), px, py, frame_radius, "cyan")
                if planned_path_history and i < len(planned_path_history):
                    overlay_planned_path(plt.gca(), planned_path_history[i])
                if control_waypoint_history and i < len(control_waypoint_history):
                    overlay_control_waypoints(plt.gca(), control_waypoint_history[i])

                plt.colorbar(label="Masked information gain")
                plt.title(f"Masked Information Gain after {step_num} measurements")
                plt.xlabel("x")
                plt.ylabel("y")
                append_current_figure(writer)
                plt.close()

        print(f"Utility MP4 saved to: {util_mp4_path}")

    varmin = min(np.diag(P_snap).min() for P_snap in P_history)
    varmax = max(np.diag(P_snap).max() for P_snap in P_history)

    var_mp4_path = os.path.join(path, "gp_variance_evolution.mp4")
    with imageio.get_writer(var_mp4_path, fps=4, codec='libx264') as writer:
        for i, (P_snap, step_num) in enumerate(zip(P_history, step_numbers)):
            if i % frame_stride != 0:
                continue

            var_field = np.diag(P_snap).reshape(X.shape)

            plt.figure(figsize=video_figsize)
            plt.imshow(
                var_field,
                origin="lower",
                extent=[xs.min(), xs.max(), ys.min(), ys.max()],
                aspect="equal",
                cmap="viridis",
                vmin=varmin,
                vmax=varmax
            )

            px, py, frame_radius = footprint_for_frame(i)
            plt.plot(px, py, "wo", markersize=5)
            add_fov_rectangle(plt.gca(), px, py, frame_radius, "cyan")

            plt.colorbar(label="Variance")
            plt.title(f"Variance after {step_num} measurements")
            plt.xlabel("x")
            plt.ylabel("y")
            append_current_figure(writer)
            plt.close()

    print(f"Variance MP4 saved to: {var_mp4_path}")


def grid_search(X, Y, cx, cy, utility_history, margin=None):
    if margin is None:
        margin = step * 2   # match your dynamics buffer

    nx = 8
    ny = 8

    # sample ONLY inside valid region
    x_positions = np.linspace(X.min() + margin, X.max() - margin, nx)
    y_positions = np.linspace(Y.min() + margin, Y.max() - margin, ny)

    selected_points = []
    for x in x_positions:
        xg = X[0, np.argmin(np.abs(X[0, :] - x))]
        for y in y_positions:
            yg = Y[np.argmin(np.abs(Y[:, 0] - y)), 0]
            selected_points.append((xg, yg))

    # remove duplicates (important when snapping)
    selected_points = list(dict.fromkeys(selected_points))

    util_values = []
    for xg, yg in selected_points:
        idx = np.where(
            np.isclose(X.ravel(), xg) &
            np.isclose(Y.ravel(), yg)
        )[0]

        if len(idx) == 0:
            continue

        idx = idx[0]
        util_value = utility_history[-1][idx]

        # shape preserved exactly
        util_values.append((util_value, (xg, yg)))

    util_values.sort(key=lambda x: x[0], reverse=True)
    return util_values

# #grid_search is set to greedy step, searching for next best.
# def grid_search(X, Y, cx, cy, utility_history, grid_edge = step*2):
#     # Select 20 equidistant points over the grid: 5 in x, 4 in y
#     nx = 5
#     ny = 4
#     x_positions = np.linspace(X.min() - grid_edge, X.max() + grid_edge, nx)
#     y_positions = np.linspace(Y.min() - grid_edge, Y.max() + grid_edge, ny)
    
#     selected_points = []
#     for x in x_positions:
#         # Find closest grid x
#         xg = X[0, np.argmin(np.abs(X[0, :] - x))]
#         for y in y_positions:
#             # Find closest grid y
#             yg = Y[np.argmin(np.abs(Y[:, 0] - y)), 0]
#             selected_points.append((xg, yg))
    
#     util_values = []
#     for xg, yg in selected_points:
#         idx = np.where(
#             np.isclose(X.ravel(), xg) &
#             np.isclose(Y.ravel(), yg)
#         )[0]

#         if len(idx) == 0:
#             continue

#         idx = idx[0]
#         util_value = utility_history[-1][idx]
#         util_values.append((util_value, (xg, yg)))
#         # print(f"Utility at ({xg}, {yg}): {util_value:.4f}")
    
#     util_values.sort(key=lambda x: x[0], reverse=True)
#     return util_values  # Return sorted list of utility values and their corresponding coordinates




if __name__ == "__main__":
    path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"
    data = np.loadtxt(rf"{path}\map_1_blob_grid_counts.csv", delimiter=",", skiprows=1)

    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    # discretized points
    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    N = X_test.shape[0]
    mu = mean.copy()
    P = cov.copy()
    R = 1e-6

    mu_history = []
    P_history = []
    step_numbers = []
    grad_history = []
    pos_history = []
    utility_history = []
    sorted_util_values_list = []

    mu_history.append(mu.copy())
    P_history.append(P.copy())
    step_numbers.append(0)
    utility_threshold = 0.4

    initial_utility = utility_function(mu, P, utility_threshold)
    utility_history.append(initial_utility)
    
    # sorted_util_values_list.append(sorted_util_values)
    

    save_every = 5
    lateral_coverage = step * 2
    samplestep = 4.0
    timealloted = 150

   

    # initial stating position and gradient
    cx, cy = 50.0, 50.0
    grad_x, grad_y = 0.0, 0.0
    pos_history.append((cx, cy))

    sorted_util_values_list.append(grid_search(X, Y, cx, cy, utility_history))

    # store initial gradient field
    initial_var_field = np.diag(P_history[-1]).reshape(X.shape)
    gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
    grad_history.append((gx0, gy0))

    #from here we consider movement and measurement updates

    for ts in range(0, timealloted):
        cx = np.clip(cx + grad_x + np.ceil(np.random.uniform(-step, step)), xmin, xmax)
        cy = np.clip(cy + grad_y + np.ceil(np.random.uniform(-step, step)), ymin, ymax)
        cx = step * np.round(cx / step)
        cy = step * np.round(cy / step)
        pos_history.append((cx, cy))

        print(cx, cy)
        step_numbers.append(ts + 1)

        fov = [
            (x, y)
            for x in np.arange(cx - lateral_coverage, cx + lateral_coverage + 1e-9, step)
            for y in np.arange(cy - lateral_coverage, cy + lateral_coverage + 1e-9, step)
        ]

        sensor = np.zeros((len(fov), N))

        for i, (x_meas, y_meas) in enumerate(fov):
            idx = np.where(
                np.isclose(X_test[:, 0], x_meas) &
                np.isclose(X_test[:, 1], y_meas)
            )[0]

            if len(idx) == 0:
                continue

            idx = idx[0]
            sensor[i, idx] = 1.0

        measurement_list = []
        for x_fov, y_fov in fov:
            idx = np.where(
                np.isclose(pts[:, 0], x_fov) &
                np.isclose(pts[:, 1], y_fov)
            )[0]

            if idx.size > 0:
                measurement_list.append(pts[idx[0], 2])
            else:
                measurement_list.append(0.0)

        z_meas = np.array(measurement_list)

        mu, P = kalman_update(mu, P, sensor, z_meas, R)

        

        mu_history.append(mu.copy())
        P_history.append(P.copy())
        utility = utility_function(mu, P, utility_threshold)
        utility_history.append(utility)
        sorted_util_value = grid_search(X, Y, cx, cy, utility_history)
        sorted_util_values_list.append(sorted_util_value)
        

        grad, grad_field, target = sampler(cx, cy, X, Y, P_history, samplestep)
        grad_x, grad_y = grad
        target_x, target_y = target

        grad_history.append(grad_field)

        print("step vector:", grad_x, grad_y)
        print("target:", target_x, target_y)
        print(sorted_util_value)

    create_plots_and_gifs(path, mu_history, P_history, step_numbers, grad_history, pos_history, utility_history, sorted_util_values_list, X, Y, xs, ys, cx, cy, lateral_coverage, xmin, xmax, ymin, ymax)
