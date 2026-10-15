import os

# Must be set before numpy/scipy import their BLAS backend. Each MPI rank now
# runs its own BRANCH_COUNT-worker process pool (see PARALLEL_BRANCHES below);
# without this, every one of those worker processes would also try to grab
# many BLAS threads internally, oversubscribing whatever cores the rank was
# actually allocated. setdefault so an explicit external override (e.g. set
# by the SLURM job script) always wins.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

script_dir = Path(__file__).resolve().parent
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from CMAES_classic_singlemap import dynamics_3d, waypoint_3d
from evalmetrics import compute_reconstruction_rmse
from gaussianprocesstraining import (
    build_correlated_noise_covariance,
    build_pyramid_lattice_3d,
    build_sensor_matrix,
    build_spline_trajectory_3d,
    cma_es_refine_waypoints_3d,
    covariance_after_sensor,
    fov_grid_points,
    future_sensor_model_3d,
    importance_filter,
    initialize_gp,
    kalman_update,
    masked_expected_variance_reduction_from_sensor,
    noise_model,
    sample_correlated_sensor_noise,
)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


# Multimodal branching copy of DataCollector_3D_randomstart.py. Everything about
# the linear-chain structure (STARTS_PER_MAP independent chains per map, MPI
# parallelizing across chains, no per-round MPI sync) is unchanged - see that
# file's header for the full rationale. What's added here:
#   - at every round, instead of generating a single candidate plan, the first
#     step of grid_search_3d is ranked and the top BRANCH_COUNT candidates are
#     each expanded into a full plan (grid_search_3d's normal greedy fill for
#     steps 2-planning_horizon, unchanged) and refined with CMA-ES.
#   - each of the BRANCH_COUNT branches is then ACTUALLY EXECUTED (not just
#     scored by the CMA-ES predictive objective) for execution_chunk real
#     simulated steps against the true map, exactly like simulate_candidate
#     already did for the single-branch case.
#   - branches are ranked, and "which are within the neighbourhood of the
#     best" is decided, purely on VARIANCE reduction (the drop in the
#     belief-state covariance P's masked trace, real-execution-derived, not
#     the pre-execution CMA-ES prediction) - never on RMSE against the true
#     map. RMSE is only computable in simulation, where the ground truth
#     happens to be known; a real planner only ever has (mu, P), so grading
#     branches on RMSE would bake in information the planner could never
#     have used to make the choice. RMSE_correction is still computed and
#     recorded per sample as a diagnostic/eval signal, it just no longer
#     drives which branches are kept.
#   - the chain only ever advances using the single best-performing branch
#     (highest variance_correction) - the other kept branches are extra
#     recorded labels for the same conditioning state, they don't fork the
#     chain itself.
#   - branches within NEIGHBOURHOOD_THRESHOLD of the best branch's
#     variance_correction are ALL recorded (not discarded), sharing the same
#     condition_id/parent_beam_id (identifying them as alternative
#     trajectories for the same belief state) with parent_beam_index=0 for
#     the winner and 1..k for the kept alternates. These fields existed in
#     the original schema but were vestigial (always 0) - this revives their
#     original beam-search meaning for the multimodal use case.
#   - this multiplies CMA-ES cost by up to BRANCH_COUNT relative to
#     DataCollector_3D_randomstart.py, since every round now runs BRANCH_COUNT
#     independent plan+refine+execute passes instead of one.

step = 2.0
RANKLIM = 12
BETA = 1.0
alpha = 0.02
utility_threshold = 0.4
planning_horizon = 8
initial_map = 1
mapcount = 25
samples_per_segment = 5
execution_chunk = 40
SENSORNOISE_SEED = 123
MAPTYPE = "multiblob_normalized"
INIT_ALTITUDE = 10.0
ZMIN = 10.0
ZMAX = 40.0
ANGLE_OF_VIEW = 60.0

# --- multimodal branching configuration ---
BRANCH_COUNT = 4  # how many of grid_search_3d's top first-step candidates to
                   # expand into full plans each round
NEIGHBOURHOOD_THRESHOLD = 0.96  # a branch is kept (recorded) if its real
                                 # achieved variance_correction (masked-trace
                                 # reduction in P, NOT RMSE against the true
                                 # map - see header) is at least this fraction
                                 # of the best branch's
PARALLEL_BRANCHES = True  # True -> each rank evaluates its BRANCH_COUNT branches
                           # concurrently via a local process pool (one pool per
                           # rank, created once in main() and reused for every
                           # chain/round that rank processes - not recreated per
                           # round). False -> evaluate branches sequentially, useful
                           # for local debugging without paying process-pool
                           # overhead. This is local parallelism nested inside each
                           # existing MPI rank; it does not change how chains are
                           # distributed across ranks or add any per-round MPI sync.
                           # For real speedup, the job must give each rank at least
                           # BRANCH_COUNT dedicated cores (e.g. SLURM
                           # --cpus-per-task=BRANCH_COUNT) - otherwise there are no
                           # spare cores for the pool to use.

# --- random-start configuration ---
STARTS_PER_MAP = 3
START_MARGIN = step * 2  # keep starts off the map edge, same margin convention
                          # used elsewhere in gaussianprocesstraining.py
START_POSITION_SEED = 990_001  # arbitrary, disjoint from SENSORNOISE_SEED so the
                                # position draw and the sensor-noise stream don't
                                # share a seed value
START_POSITION_CENTER = (50.0, 50.0)  # mean of the Gaussian start-position sampler
START_POSITION_STD = 20.0  # standard deviation (map units) in both x and y,
                            # independently. At 20, ~95% of unclipped draws land
                            # within 40 units of center (i.e. within [10, 90] on
                            # the default 0-100 map), comfortably inside the
                            # margin-inset bounds most of the time. Draws outside
                            # [lo, hi] are clipped rather than resampled, so a
                            # larger std will pile up more starts exactly at the
                            # margin boundary instead of spreading further out.
WARMUP_OFFSET = 40.0  # warmup target = start + this offset, clipped to bounds
# Optional: force specific (cx, cy) starts to always be included alongside the
# random ones, e.g. to guarantee coverage of points other scripts hard-code as
# their own reset position (DPPOTRAINING*.py uses map center;
# Diffusionplanner_singlemap.py uses (4.0, 4.0)). Empty by default.
EXTRA_FIXED_STARTS = []

target_trajectory_len = planning_horizon * samples_per_segment + 1
final_dataset_path = script_dir / "CMAES_beamsearch_dataset_3d_synthetic_final.pt"
chunk_dir = script_dir / "CMAES_beamsearch_dataset_3d_randomstart_multimodal_chunks"


def gaussian_grid_start(rng, xmin, xmax, ymin, ymax, step, margin, center=START_POSITION_CENTER, std=START_POSITION_STD):
    lo_x, hi_x = xmin + margin, xmax - margin
    lo_y, hi_y = ymin + margin, ymax - margin

    cx = float(np.clip(rng.normal(center[0], std), lo_x, hi_x))
    cy = float(np.clip(rng.normal(center[1], std), lo_y, hi_y))

    cx = step * round(cx / step)
    cy = step * round(cy / step)
    # snapping to the grid can only move a point that was already within
    # [lo, hi] by less than one step, and lo/hi are themselves grid-aligned,
    # so this re-clip is defensive rather than load-bearing.
    cx = float(np.clip(cx, lo_x, hi_x))
    cy = float(np.clip(cy, lo_y, hi_y))
    return cx, cy


def make_round_seed(selected_map, start_index, round_idx, branch_idx=0, purpose=0):
    """A valid numpy/cma RNG seed derived from the map/start/round/branch/
    purpose indices, via SeedSequence rather than raw multiply-and-add. numpy
    requires seeds in [0, 2**32 - 1]; the old formula (SENSORNOISE_SEED +
    selected_map*100_000_000 + start_index*1_000_000 + round_idx*1_000)
    silently overflowed that range once selected_map reached ~43, crashing
    every worker with "Seed must be between 0 and 2**32 - 1". SeedSequence
    hashes arbitrarily large/many integer inputs into a valid seed, so this
    can't overflow regardless of how large mapcount/RANKLIM/STARTS_PER_MAP/
    BRANCH_COUNT get. purpose distinguishes the sensor-noise seed from the
    CMA-ES planner seed for the same (map, start, round, branch) so they
    don't share a stream.
    """
    return int(
        np.random.SeedSequence(
            [SENSORNOISE_SEED, selected_map, start_index, round_idx, branch_idx, purpose]
        ).generate_state(1)[0]
    )


def condition_id_for(selected_map, start_index, round_idx):
    return (
        selected_map * 100_000_000
        + start_index * 1_000_000
        + round_idx * 1_000
    )


def assigned_tasks_for_rank(tasks, rank, size):
    return tasks[rank::size]


def get_mpi_context():
    if MPI is None:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def detect_mpi_launch_issue(env, mpi_size):
    slurm_ntasks = int(env.get("SLURM_NTASKS", "1"))
    return slurm_ntasks > 1 and mpi_size == 1


def make_empty_dataset():
    return {
        "trajectories": [],
        "control_waypoints": [],
        "current_position": [],
        "current_mean": [],
        "current_var": [],
        "current_util": [],
        "map_id": [],
        "beta": [],
        "timestep": [],
        "RMSE_correction": [],
        "variance_correction": [],
        "condition_id": [],
        "parent_beam_id": [],
        "parent_beam_index": [],
        "initial_heading_velocity": [],
        "start_position": [],
        "start_index": [],
    }


def tensorize_dataset(dataset):
    return {
        "trajectories": torch.tensor(
            np.asarray(dataset["trajectories"], dtype=np.float32)
        ).permute(0, 2, 1),
        "control_waypoints": torch.tensor(
            np.asarray(dataset["control_waypoints"], dtype=np.float32)
        ).permute(0, 2, 1),
        "current_position": torch.tensor(
            np.asarray(dataset["current_position"], dtype=np.float32)
        ),
        "current_mean": torch.tensor(
            np.asarray(dataset["current_mean"], dtype=np.float32)
        ),
        "current_var": torch.tensor(
            np.asarray(dataset["current_var"], dtype=np.float32)
        ),
        "current_util": torch.tensor(
            np.asarray(dataset["current_util"], dtype=np.float32)
        ),
        "map_id": torch.tensor(np.asarray(dataset["map_id"], dtype=np.int64)),
        "beta": torch.tensor(np.asarray(dataset["beta"], dtype=np.float32)),
        "timestep": torch.tensor(np.asarray(dataset["timestep"], dtype=np.int64)),
        "RMSE_correction": torch.tensor(
            np.asarray(dataset["RMSE_correction"], dtype=np.float32)
        ),
        "variance_correction": torch.tensor(
            np.asarray(dataset["variance_correction"], dtype=np.float32)
        ),
        "condition_id": torch.tensor(
            np.asarray(dataset["condition_id"], dtype=np.int64)
        ),
        "parent_beam_id": torch.tensor(
            np.asarray(dataset["parent_beam_id"], dtype=np.int64)
        ),
        "parent_beam_index": torch.tensor(
            np.asarray(dataset["parent_beam_index"], dtype=np.int64)
        ),
        "initial_heading_velocity": torch.tensor(
            np.asarray(dataset["initial_heading_velocity"], dtype=np.float32)
        ),
        "start_position": torch.tensor(
            np.asarray(dataset["start_position"], dtype=np.float32)
        ),
        "start_index": torch.tensor(
            np.asarray(dataset["start_index"], dtype=np.int64)
        ),
    }


def dataset_size(dataset):
    return len(dataset["trajectories"])


def save_dataset_chunk(dataset, selected_map):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"map_{selected_map:03d}_ranked_3d.pt"
    torch.save(tensorize_dataset(dataset), chunk_path)
    print(f"Saved chunk {chunk_path} with {dataset_size(dataset)} samples")
    return chunk_path


def consolidate_chunks(chunk_paths, output_path, delete_chunks=True):
    chunk_paths = [Path(path) for path in chunk_paths]
    if not chunk_paths:
        print("No chunks were created; final dataset was not written.")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_payload = torch.load(chunk_paths[0], map_location="cpu")
    final_payload = {key: [] for key in first_payload}

    for path in chunk_paths:
        payload = torch.load(path, map_location="cpu")
        for key, value in payload.items():
            final_payload[key].append(value)

    final_payload = {
        key: torch.cat(values, dim=0)
        for key, values in final_payload.items()
    }
    torch.save(final_payload, output_path)
    print(f"Saved consolidated dataset to {output_path}")
    print(f"Samples: {len(final_payload['map_id'])}")

    if delete_chunks:
        for path in chunk_paths:
            try:
                path.unlink(missing_ok=True)
            except PermissionError:
                print(f"Could not delete locked chunk {path}; leaving it on disk.")
        try:
            chunk_dir.rmdir()
        except OSError:
            pass


def resample_trajectory(path, target_len):
    path = np.asarray(path, dtype=np.float32)

    if path.ndim == 1:
        path = path.reshape(-1, 3)

    if len(path) == target_len:
        return path.astype(np.float32)
    if len(path) == 0:
        return np.zeros((target_len, 3), dtype=np.float32)
    if len(path) == 1:
        return np.repeat(path, target_len, axis=0).astype(np.float32)

    source_t = np.linspace(0.0, 1.0, len(path))
    target_t = np.linspace(0.0, 1.0, target_len)
    resampled = [
        np.interp(target_t, source_t, path[:, dim])
        for dim in range(path.shape[1])
    ]
    return np.stack(resampled, axis=-1).astype(np.float32)


def build_true_map_flat(pts, X_test):
    true_map_flat = np.zeros(X_test.shape[0], dtype=float)
    for x_true, y_true, value in pts:
        idx = np.where(
            np.isclose(X_test[:, 0], x_true)
            & np.isclose(X_test[:, 1], y_true)
        )[0]
        if idx.size > 0:
            true_map_flat[idx[0]] = value
    return true_map_flat


def apply_measurement_update_3d(cx, cy, cz, mu, P, true_map_flat, xs, ys, rng):
    fov = fov_grid_points(cx, cy, cz, xs, ys, angle_of_view=ANGLE_OF_VIEW)
    sensor, sensor_block_ids = build_sensor_matrix(
        fov, cz, xs, ys, return_block_ids=True
    )
    R = noise_model(cz)
    z_meas = sensor @ true_map_flat
    z_meas += sample_correlated_sensor_noise(sensor_block_ids, R, rng)
    R_cov = build_correlated_noise_covariance(sensor_block_ids, R)
    return kalman_update(mu, P, sensor, z_meas, R_cov, block_ids=sensor_block_ids)


def top_k_first_candidates(
    mu, P, xs, ys, start_pose, beta, utility_threshold, zmin, zmax, k,
    alpha=0.02, angle_of_view=ANGLE_OF_VIEW, max_measurements=64,
):
    """Rank grid_search_3d's first-step candidates and return the top k (x, y, z)
    points, without committing to any of them - this is exactly grid_search_3d's
    own step-1 scoring, isolated so we can branch instead of collapsing to argmax."""
    available = build_pyramid_lattice_3d(xs, ys, zmin, zmax)
    current_pose = np.asarray(start_pose, dtype=float)
    utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    importance_mask = utility > 0
    scored = []
    for candidate in available:
        sensor, measurement_noise = future_sensor_model_3d(
            [candidate], xs, ys, angle_of_view=angle_of_view,
            trajectory_stride=1, max_measurements=max_measurements,
        )
        gain = masked_expected_variance_reduction_from_sensor(
            P, importance_mask, sensor, measurement_noise
        )
        distance = float(np.linalg.norm(np.asarray(candidate, dtype=float) - current_pose))
        score = gain / max(distance, step)
        scored.append((score, candidate))
    scored.sort(key=lambda item: -item[0])
    return [candidate for _, candidate in scored[:k]]


def branch_grid_search_3d(
    mu, P, xs, ys, start_pose, beta, utility_threshold, planning_horizon, zmin, zmax,
    forced_first=None, alpha=0.02, angle_of_view=ANGLE_OF_VIEW, max_measurements=64,
):
    """Same greedy loop as grid_search_3d, except the first step's winner can be
    forced to a specific candidate instead of always taking the argmax - steps
    2..planning_horizon are always the normal argmax. forced_first=None recovers
    plain grid_search_3d exactly."""
    available = build_pyramid_lattice_3d(xs, ys, zmin, zmax)
    simulated_covariance = P.copy()
    current_pose = np.asarray(start_pose, dtype=float)
    selected_waypoints = []

    for step_idx in range(min(planning_horizon, len(available))):
        utility = importance_filter(
            mu, simulated_covariance, beta, threshold=utility_threshold
        )
        importance_mask = utility > 0
        scored_candidates = []

        for candidate in available:
            sensor, measurement_noise = future_sensor_model_3d(
                [candidate], xs, ys, angle_of_view=angle_of_view,
                trajectory_stride=1, max_measurements=max_measurements,
            )
            gain = masked_expected_variance_reduction_from_sensor(
                simulated_covariance, importance_mask, sensor, measurement_noise
            )
            distance = float(
                np.linalg.norm(np.asarray(candidate, dtype=float) - current_pose)
            )
            score = gain / max(distance, step)
            scored_candidates.append((score, candidate, sensor, measurement_noise))

        if step_idx == 0 and forced_first is not None:
            _, best, best_sensor, best_variance = next(
                c for c in scored_candidates if c[1] == forced_first
            )
        else:
            _, best, best_sensor, best_variance = max(
                scored_candidates, key=lambda item: item[0]
            )

        selected_waypoints.append(best)
        available.remove(best)
        simulated_covariance = covariance_after_sensor(
            simulated_covariance, best_sensor, best_variance
        )
        current_pose = np.asarray(best, dtype=float)

    return selected_waypoints


def real_receding_horizon_planner(
    cx,
    cy,
    cz,
    mu,
    P,
    xs,
    ys,
    utility_threshold,
    beta,
    planning_horizon,
    alpha=0.1,
    planner_seed=None,
    forced_first=None,
):
    flight_plan_3d = branch_grid_search_3d(
        mu,
        P,
        xs,
        ys,
        start_pose=(cx, cy, cz),
        beta=beta,
        utility_threshold=utility_threshold,
        planning_horizon=planning_horizon,
        zmin=ZMIN,
        zmax=ZMAX,
        forced_first=forced_first,
        alpha=alpha,
        angle_of_view=ANGLE_OF_VIEW,
    )

    return cma_es_refine_waypoints_3d(
        flight_plan_3d,
        mu,
        P,
        xs,
        ys,
        cx,
        cy,
        cz,
        beta,
        utility_threshold,
        ZMIN,
        ZMAX,
        predictive_variance=True,
        seed=planner_seed,
    )


def step_along_spline(
    cx,
    cy,
    cz,
    spline_path,
    spline_idx,
    samplestep,
    xmin,
    xmax,
    ymin,
    ymax,
):
    if spline_idx >= len(spline_path):
        return cx, cy, cz, spline_idx

    goal_x, goal_y, goal_z = spline_path[spline_idx]
    grad_x, grad_y, grad_z, waypoint_reached = waypoint_3d(
        cx, cy, cz, goal_x, goal_y, goal_z, step
    )

    if waypoint_reached:
        spline_idx += 1
        if spline_idx < len(spline_path):
            goal_x, goal_y, goal_z = spline_path[spline_idx]
            grad_x, grad_y, grad_z, _ = waypoint_3d(
                cx, cy, cz, goal_x, goal_y, goal_z, step
            )
        else:
            grad_x, grad_y, grad_z = 0.0, 0.0, 0.0

    if spline_idx < len(spline_path):
        x_next, y_next, z_next = spline_path[spline_idx]
        padding = step * 2
        clamped_x = min(max(x_next, xmin + padding), xmax - padding)
        clamped_y = min(max(y_next, ymin + padding), ymax - padding)
        clamped_z = min(max(z_next, ZMIN), ZMAX)
        if clamped_x != x_next or clamped_y != y_next or clamped_z != z_next:
            spline_path[spline_idx] = [clamped_x, clamped_y, clamped_z]

    cx, cy, cz = dynamics_3d(
        cx,
        cy,
        cz,
        grad_x,
        grad_y,
        grad_z,
        samplestep,
        xmin,
        xmax,
        ymin,
        ymax,
        ZMIN,
        ZMAX,
        buffer=step / 2,
    )
    return cx, cy, cz, spline_idx


def simulate_candidate(
    beta,
    cx,
    cy,
    cz,
    mu,
    P,
    pts,
    true_map_flat,
    xs,
    ys,
    xmin,
    xmax,
    ymin,
    ymax,
    samplestep,
    rng_seed,
    planner_seed,
    forced_first=None,
):
    sim_cx, sim_cy, sim_cz = cx, cy, cz
    sim_mu = mu.copy()
    sim_P = P.copy()
    sim_rng = np.random.default_rng(rng_seed)

    control_waypoints = real_receding_horizon_planner(
        sim_cx,
        sim_cy,
        sim_cz,
        sim_mu,
        sim_P,
        xs,
        ys,
        utility_threshold,
        beta,
        planning_horizon,
        alpha=alpha,
        planner_seed=planner_seed,
        forced_first=forced_first,
    )
    spline_path = build_spline_trajectory_3d(
        sim_cx,
        sim_cy,
        sim_cz,
        control_waypoints,
        samples_per_segment=samples_per_segment,
    )

    spline_idx = 0
    ticks = 0
    # Stop once execution_chunk WAYPOINTS have been reached (or the plan runs
    # out), not after a fixed tick count - step_along_spline only ever
    # advances spline_idx by at most one per tick, so a fixed tick budget
    # would let branches with tightly-spaced waypoints fly much further into
    # their plan than branches with widely-spaced ones, comparing them on an
    # unequal amount of real progress. max_ticks is just a safety cap against
    # a pathologically slow branch consuming unbounded compute; the
    # out-of-bounds clamp below should make true non-termination impossible.
    max_ticks = execution_chunk * 10
    while spline_idx < execution_chunk and spline_idx < len(spline_path) and ticks < max_ticks:
        sim_cx, sim_cy, sim_cz, spline_idx = step_along_spline(
            sim_cx,
            sim_cy,
            sim_cz,
            spline_path,
            spline_idx,
            samplestep,
            xmin,
            xmax,
            ymin,
            ymax,
        )
        sim_mu, sim_P = apply_measurement_update_3d(
            sim_cx,
            sim_cy,
            sim_cz,
            sim_mu,
            sim_P,
            true_map_flat,
            xs,
            ys,
            sim_rng,
        )
        ticks += 1

    rmse = compute_reconstruction_rmse(
        mu=sim_mu,
        pts=pts,
        xs=xs,
        ys=ys,
        step=step,
        utility_threshold=utility_threshold,
        xmin=xmin,
        ymin=ymin,
    )

    return {
        "beta": float(beta),
        "control_waypoints": control_waypoints,
        "spline_path": spline_path,
        "final_cx": sim_cx,
        "final_cy": sim_cy,
        "final_cz": sim_cz,
        "final_mu": sim_mu,
        "final_P": sim_P,
        "occupied_rmse": rmse["occupied_rmse"],
        "global_rmse": rmse["global_rmse"],
    }


def _simulate_branch_worker(payload):
    """Module-level (picklable) wrapper around simulate_candidate for use with
    ProcessPoolExecutor - closures/lambdas aren't picklable under Windows'
    spawn start method, so this has to be a plain top-level function taking a
    single tuple argument."""
    (
        branch_idx, forced_first, beta, cx, cy, cz, mu, P, pts, true_map_flat,
        xs, ys, xmin, xmax, ymin, ymax, samplestep, rng_seed, planner_seed,
    ) = payload
    result = simulate_candidate(
        beta, cx, cy, cz, mu, P, pts, true_map_flat, xs, ys, xmin, xmax, ymin, ymax,
        samplestep, rng_seed=rng_seed, planner_seed=planner_seed, forced_first=forced_first,
    )
    result["branch_idx"] = branch_idx
    return result


def record_candidate(
    dataset, candidate, state, selected_map, round_idx, map_shape, start_position,
    start_index, parent_beam_id=0, parent_beam_index=0,
):
    trajectory = resample_trajectory(candidate["spline_path"], target_trajectory_len)
    current_position = np.asarray(
        [state["cx"], state["cy"], state["cz"]], dtype=np.float32
    )
    if trajectory.shape[0] < 2:
        raise ValueError(
            f"Expected at least two trajectory points, got shape {trajectory.shape}"
        )
    initial_heading_velocity = trajectory[1] - current_position

    dataset["trajectories"].append(trajectory)
    dataset["control_waypoints"].append(
        np.asarray(candidate["control_waypoints"], dtype=np.float32)
    )
    dataset["current_position"].append(current_position)
    dataset["current_mean"].append(state["mu"].reshape(map_shape).astype(np.float32))
    dataset["current_var"].append(np.diag(state["P"]).reshape(map_shape).astype(np.float32))
    dataset["current_util"].append(state["utility"].reshape(map_shape).astype(np.float32))
    dataset["map_id"].append(selected_map)
    dataset["beta"].append(candidate["beta"])
    dataset["timestep"].append(round_idx)
    dataset["RMSE_correction"].append(candidate["rmse_correction"])
    dataset["variance_correction"].append(candidate["variance_correction"])
    dataset["condition_id"].append(candidate["condition_id"])
    dataset["parent_beam_id"].append(parent_beam_id)
    dataset["parent_beam_index"].append(parent_beam_index)
    dataset["initial_heading_velocity"].append(initial_heading_velocity.astype(np.float32))
    dataset["start_position"].append(np.asarray(start_position, dtype=np.float32))
    dataset["start_index"].append(start_index)


def load_map(selected_map):
    csv_path = script_dir / "csv"
    data = np.loadtxt(
        csv_path / f"map_{selected_map}_{MAPTYPE}_grid_counts.csv",
        delimiter=",",
        skiprows=1,
    )
    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    return pts[mask]


def warmup_rollout(
    cx, cy, cz, mu, P, true_map_flat, xs, ys, xmin, xmax, ymin, ymax, rng, goal_x, goal_y
):
    samplestep = step
    for _ in range(2):
        grad_x, grad_y, grad_z, _ = waypoint_3d(
            cx,
            cy,
            cz,
            goal_x=goal_x,
            goal_y=goal_y,
            goal_z=INIT_ALTITUDE,
            step=step,
        )
        cx, cy, cz = dynamics_3d(
            cx,
            cy,
            cz,
            grad_x,
            grad_y,
            grad_z,
            samplestep,
            xmin,
            xmax,
            ymin,
            ymax,
            ZMIN,
            ZMAX,
            buffer=step / 2,
        )
        mu, P = apply_measurement_update_3d(
            cx, cy, cz, mu, P, true_map_flat, xs, ys, rng
        )
    return cx, cy, cz, mu, P


def starts_for_map(selected_map, xmin, xmax, ymin, ymax):
    starts = []
    for start_index in range(STARTS_PER_MAP):
        pos_rng = np.random.default_rng(
            START_POSITION_SEED + selected_map * 1_000 + start_index
        )
        cx, cy = gaussian_grid_start(pos_rng, xmin, xmax, ymin, ymax, step, START_MARGIN)
        starts.append((cx, cy))
    for cx, cy in EXTRA_FIXED_STARTS:
        starts.append((float(cx), float(cy)))
    return starts


def run_chain_and_record(
    local_dataset,
    selected_map,
    start_index,
    start_cx,
    start_cy,
    pts,
    true_map_flat,
    xs,
    ys,
    X,
    xmin,
    xmax,
    ymin,
    ymax,
    X_test,
    cov,
    mpi_rank,
    mpi_size,
    rank_limit=None,
    pool=None,
):
    mean = np.full(X_test.shape[0], utility_threshold + 0.1, dtype=float) #changed from negative to positive for UCB
    mu = mean.copy()
    P = cov.copy()
    cx, cy, cz = start_cx, start_cy, INIT_ALTITUDE
    samplestep = step
    rng = np.random.default_rng(
        SENSORNOISE_SEED + selected_map * 100_000_000 + start_index * 1_000_000
    )

    warmup_goal_x = float(
        np.clip(cx + WARMUP_OFFSET, xmin + START_MARGIN, xmax - START_MARGIN)
    )
    warmup_goal_y = float(
        np.clip(cy + WARMUP_OFFSET, ymin + START_MARGIN, ymax - START_MARGIN)
    )

    cx, cy, cz, mu, P = warmup_rollout(
        cx,
        cy,
        cz,
        mu,
        P,
        true_map_flat,
        xs,
        ys,
        xmin,
        xmax,
        ymin,
        ymax,
        rng,
        goal_x=warmup_goal_x,
        goal_y=warmup_goal_y,
    )

    for round_idx in range(rank_limit if rank_limit is not None else RANKLIM):
        utility = importance_filter(mu, P, BETA, threshold=utility_threshold)
        # Branches are ranked and neighbourhood-filtered on variance reduction,
        # not RMSE: variance is computable purely from the belief state (mu, P)
        # a real planner actually has at decision time, whereas RMSE requires
        # comparing against the true map (pts), which is only available here
        # because this is a simulation - a deployed planner could never use it
        # to judge trajectories. importance_mask is fixed once per round (from
        # the state BEFORE any branch runs) so every branch's variance
        # reduction is measured over the same set of cells.
        importance_mask = utility > 0
        baseline_variance = float(np.sum(np.diag(P)[importance_mask]))
        # RMSE is still computed and recorded (RMSE_correction stays in the
        # dataset as an evaluation/diagnostic signal) - it's just no longer
        # used to pick the winner or the kept neighbourhood.
        baseline_rmse = compute_reconstruction_rmse(
            mu=mu,
            pts=pts,
            xs=xs,
            ys=ys,
            step=step,
            utility_threshold=utility_threshold,
            xmin=xmin,
            ymin=ymin,
        )["global_rmse"]
        record_state = {
            "cx": cx,
            "cy": cy,
            "cz": cz,
            "mu": mu.copy(),
            "P": P.copy(),
            "utility": utility.copy(),
        }

        first_candidates = top_k_first_candidates(
            mu, P, xs, ys, (cx, cy, cz), BETA, utility_threshold, ZMIN, ZMAX,
            k=BRANCH_COUNT, alpha=alpha, angle_of_view=ANGLE_OF_VIEW,
        )

        if pool is not None:
            payloads = [
                (
                    branch_idx, forced_first, BETA, cx, cy, cz, mu, P, pts, true_map_flat,
                    xs, ys, xmin, xmax, ymin, ymax, samplestep,
                    make_round_seed(selected_map, start_index, round_idx, branch_idx, purpose=0),
                    make_round_seed(selected_map, start_index, round_idx, branch_idx, purpose=1),
                )
                for branch_idx, forced_first in enumerate(first_candidates)
            ]
            branch_results = sorted(
                pool.map(_simulate_branch_worker, payloads), key=lambda r: r["branch_idx"]
            )
            for branch_candidate in branch_results:
                branch_candidate["rmse_correction"] = baseline_rmse - branch_candidate["global_rmse"]
                branch_variance = float(np.sum(np.diag(branch_candidate["final_P"])[importance_mask]))
                branch_candidate["variance_correction"] = baseline_variance - branch_variance
        else:
            branch_results = []
            for branch_idx, forced_first in enumerate(first_candidates):
                branch_candidate = simulate_candidate(
                    BETA,
                    cx,
                    cy,
                    cz,
                    mu,
                    P,
                    pts,
                    true_map_flat,
                    xs,
                    ys,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    samplestep,
                    rng_seed=make_round_seed(selected_map, start_index, round_idx, branch_idx, purpose=0),
                    planner_seed=make_round_seed(selected_map, start_index, round_idx, branch_idx, purpose=1),
                    forced_first=forced_first,
                )
                branch_candidate["rmse_correction"] = baseline_rmse - branch_candidate["global_rmse"]
                branch_variance = float(np.sum(np.diag(branch_candidate["final_P"])[importance_mask]))
                branch_candidate["variance_correction"] = baseline_variance - branch_variance
                branch_results.append(branch_candidate)

        winner_idx = max(
            range(len(branch_results)), key=lambda i: branch_results[i]["variance_correction"]
        )
        best_correction = branch_results[winner_idx]["variance_correction"]

        if best_correction > 0:
            kept_indices = [
                i for i, b in enumerate(branch_results)
                if i == winner_idx or (b["variance_correction"] / best_correction) >= NEIGHBOURHOOD_THRESHOLD
            ]
        else:
            kept_indices = [winner_idx]

        # winner is always parent_beam_index 0; kept alternates keep their
        # original branch order after it
        ordered_kept = [winner_idx] + [i for i in kept_indices if i != winner_idx]
        group_id = condition_id_for(selected_map, start_index, round_idx)

        for parent_beam_index, branch_i in enumerate(ordered_kept):
            branch_candidate = branch_results[branch_i]
            branch_candidate["condition_id"] = group_id
            record_candidate(
                local_dataset,
                branch_candidate,
                record_state,
                selected_map,
                round_idx,
                X.shape,
                (start_cx, start_cy),
                start_index,
                parent_beam_id=group_id,
                parent_beam_index=parent_beam_index,
            )

        winner = branch_results[winner_idx]
        print(
            f"[MPI {mpi_rank}/{mpi_size}] Map {selected_map}, start {start_index}, "
            f"round {round_idx}: masked variance {baseline_variance:.4f} -> "
            f"{baseline_variance - winner['variance_correction']:.4f} "
            f"(reduction={winner['variance_correction']:.4f}), global RMSE "
            f"{baseline_rmse:.4f} -> {winner['global_rmse']:.4f} "
            f"({len(ordered_kept)}/{BRANCH_COUNT} branches kept within "
            f"{int(NEIGHBOURHOOD_THRESHOLD * 100)}%)"
        )

        cx, cy, cz = winner["final_cx"], winner["final_cy"], winner["final_cz"]
        mu, P = winner["final_mu"], winner["final_P"]


def main():
    comm, mpi_rank, mpi_size = get_mpi_context()
    if detect_mpi_launch_issue(os.environ, mpi_size):
        raise RuntimeError(
            "SLURM_NTASKS is greater than 1, but mpi4py sees MPI size 1. "
            "This means the script is being launched as repeated serial jobs, "
            "not one MPI job. Use an MPI-aware launch such as "
            "`srun --mpi=pmix python DataCollector_3D_randomstart_multimodal.py` or "
            "`mpiexec -n $SLURM_NTASKS python DataCollector_3D_randomstart_multimodal.py`."
        )

    chunk_paths = []
    if mpi_rank == 0:
        print(
            f"Running 3D multimodal-branching linear-chain collection with "
            f"MPI size={mpi_size}, BRANCH_COUNT={BRANCH_COUNT}, "
            f"NEIGHBOURHOOD_THRESHOLD={NEIGHBOURHOOD_THRESHOLD}, "
            f"PARALLEL_BRANCHES={PARALLEL_BRANCHES}"
        )

    # One pool per rank, created once and reused for every chain/round that
    # rank processes - not recreated per round or per chain, so process
    # startup cost is paid once for the rank's entire lifetime.
    pool = ProcessPoolExecutor(max_workers=BRANCH_COUNT) if PARALLEL_BRANCHES else None

    try:
        for selected_map in range(initial_map, initial_map + mapcount):
            if mpi_rank == 0:
                print(f"\nCollecting 3D CMA-ES multimodal rollout data for map {selected_map}")

            pts = load_map(selected_map)
            _, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, _ = initialize_gp()
            true_map_flat = build_true_map_flat(pts, X_test)

            map_starts = list(enumerate(starts_for_map(selected_map, xmin, xmax, ymin, ymax)))
            local_starts = assigned_tasks_for_rank(map_starts, mpi_rank, mpi_size)

            local_dataset = make_empty_dataset()
            for start_index, (start_cx, start_cy) in local_starts:
                print(
                    f"[MPI {mpi_rank}/{mpi_size}] Map {selected_map}, start {start_index}: "
                    f"({start_cx:.1f}, {start_cy:.1f})"
                )
                run_chain_and_record(
                    local_dataset,
                    selected_map,
                    start_index,
                    start_cx,
                    start_cy,
                    pts,
                    true_map_flat,
                    xs,
                    ys,
                    X,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    X_test,
                    cov,
                    mpi_rank,
                    mpi_size,
                    pool=pool,
                )

            if comm is None:
                gathered_datasets = [local_dataset]
            else:
                gathered_datasets = comm.gather(local_dataset, root=0)

            if mpi_rank == 0:
                map_dataset = make_empty_dataset()
                for partial in gathered_datasets:
                    for key in map_dataset:
                        map_dataset[key].extend(partial[key])
                chunk_paths.append(save_dataset_chunk(map_dataset, selected_map))
                del map_dataset

            del pts, X_test, mean, cov, xs, ys, X, Y, true_map_flat, local_dataset
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if mpi_rank == 0:
        consolidate_chunks(chunk_paths, final_dataset_path, delete_chunks=True)


if __name__ == "__main__":
    main()
