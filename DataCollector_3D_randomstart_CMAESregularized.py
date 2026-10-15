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
import cma
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
    clip_waypoints_continuous_3d,
    cma_es_refine_waypoints_3d,
    COST_EXPONENT,
    covariance_after_sensor,
    CMA_PREDICTIVE_MAXFEVALS,
    CMA_PREDICTIVE_MAXITER,
    CMA_PREDICTIVE_POPSIZE,
    CMA_STEP_SIZE_XY,
    CMA_STEP_SIZE_Z,
    fov_grid_points,
    fov_lateral_radius,
    flatten_waypoints_3d,
    future_sensor_model_3d,
    importance_filter,
    initialize_gp,
    kalman_update,
    masked_expected_variance_reduction_from_sensor,
    noise_model,
    sample_correlated_sensor_noise,
    trajectory_objective_3d,
    unflatten_waypoints_3d,
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
#     scored by the CMA-ES predictive objective) for EXECUTION_CHUNK_SCORING
#     real simulated steps against the true map, exactly like
#     simulate_candidate already did for the single-branch case. The chain
#     itself only advances EXECUTION_CHUNK_UPDATING steps into that same
#     simulated rollout (a receding-horizon-style partial commitment) -
#     scoring and advancing are deliberately decoupled, see
#     execute_refined_candidate.
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
#   - the chain advances along ONE kept mode per round, drawn uniformly at
#     random from the kept set (CONTINUE_RANDOM_KEPT_MODE=True; False restores
#     the old behaviour of always following the best branch). All kept modes
#     are recorded as labels for the same conditioning state, but only the
#     drawn one forks the chain's future, so cost stays linear. Drawing
#     uniformly matches what a policy that reproduces the labels does at
#     inference (it samples among the kept modes), so the states the chain
#     visits cover the states each mode leads to, not only those reached by
#     the best mode. Which mode was followed is stored in the "continued"
#     field of the dataset.
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
RANKLIM = 16
BETA = 1.0
alpha = 0.02
utility_threshold = 0.3  # NAIP (LCB); the GRF/UCB value was 0.5
planning_horizon = 8
initial_map = 0
mapcount = 60
samples_per_segment = 5
# Scoring and advancing are deliberately different lengths along the same
# simulated rollout (see execute_refined_candidate): EXECUTION_CHUNK_SCORING
# is how far each branch/CMA-variant is actually simulated forward to judge
# how much variance/RMSE it reduces (close to the full planned trajectory,
# planning_horizon*samples_per_segment+1=41), while EXECUTION_CHUNK_UPDATING
# is how far along that same rollout the chain actually commits before the
# next replan - a genuinely receding-horizon partial commitment, rather than
# walking almost the entire plan before replanning.
EXECUTION_CHUNK_SCORING = 40
EXECUTION_CHUNK_UPDATING = 20
SENSORNOISE_SEED = 123
MAPTYPE = "NAIP"
INIT_ALTITUDE = 10.0
ZMIN = 10.0
ZMAX = 40.0
ANGLE_OF_VIEW = 60.0

# NAIP-fitted kernel scale (median signal variance and GP-fitted lengthscale
# over the non-degenerate NAIP maps, see
# CARRYOVER_NAIP_NORMALIZATION_BUG_AND_MULTIMODAL_INVESTIGATION.md section 2).
# initialize_gp()'s own defaults must match these for the planners to see the
# same prior as the data collector. GRF-era values were sigma2=0.05,
# lengthscale=6.08.
GP_KERNEL_SIGMA2 = 0.0079
GP_KERNEL_LENGTHSCALE = 4.78

# --- multimodal branching configuration ---
BRANCH_COUNT = 1  # how many of grid_search_3d's top first-step candidates to
                   # expand into full plans each round
NEIGHBOURHOOD_THRESHOLD = 0.95  # a branch is kept (recorded) if its real
                                 # achieved variance_correction (masked-trace
                                 # reduction in P, NOT RMSE against the true
                                 # map - see header) is at least this fraction
                                 # of the best branch's. 0.95 keeps only
                                 # candidates within 5% of the winner, so a kept
                                 # alternative achieves essentially the same
                                 # variance reduction; the Jaccard filter below
                                 # then makes sure it is a distinct mode.
PARALLEL_BRANCHES = False  # True -> each rank evaluates its BRANCH_COUNT branches
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

# --- CMA-ES solutions per round ---
# Every round runs CMA_SOLUTIONS_PER_BRANCH independent, plain (unpenalized)
# CMA-ES refinements from the same grid-search warm start. They differ only by
# seed, so each lands in whatever basin its own random search finds. There is
# deliberately NO diversity/distance term in the CMA-ES objective: distinctness
# is enforced afterwards by the Jaccard filter below. The step size (20 m xy /
# 8 m z) and population (12) come from gaussianprocesstraining.py; the number
# of generations is set here for the collector only, so the live CMA-ES planner
# used as an evaluation baseline keeps its own budget.
CMA_SOLUTIONS_PER_BRANCH = 8
CMA_GENERATIONS = int(os.environ.get("CMA_GENERATIONS", "200"))
CMA_MAXFEVALS = CMA_GENERATIONS * CMA_PREDICTIVE_POPSIZE * 2  # never the binding limit; maxiter (generations) is

# --- Jaccard distinctness filter ---
# After the variance-based neighbourhood cut, a kept alternative is only
# recorded if the set of important cells its path observes differs from the
# set observed by every already-recorded path by at least this Jaccard
# distance (0 = identical footprints, 1 = disjoint). The winner is always kept.
# This is the only mechanism that makes retained solutions distinct modes.
# 0.3 rejects only near-duplicates (calibrated on NAIP map 5: about 2% of
# candidate pairs fall below it; 0.5 rejected about 25%).
JACCARD_MIN_DISTANCE = float(os.environ.get("JACCARD_MIN_DISTANCE", "0.3"))

# --- which kept mode the chain continues from ---
# True: draw uniformly from the kept set each round (deterministic per
# map/start/round via make_round_seed). False: always the best branch.
CONTINUE_RANDOM_KEPT_MODE = os.environ.get("CONTINUE_RANDOM_KEPT_MODE", "1") == "1"
CONTINUATION_SEED_PURPOSE = 100  # disjoint from purposes 0 (sensor noise), 1 (planner) and 2.. (variant seeds)

# --- parallel layout / resume ---
# Every (map, start) chain is an independent task. Tasks are dealt round-robin
# to ALL MPI ranks, so any number of ranks up to the number of chains stays
# busy (the old layout split only the STARTS_PER_MAP starts of one map across
# ranks, leaving ranks beyond that idle, with a sync after every map). Each
# finished chain is saved as its own chunk file straight away, so a job killed
# by the time limit keeps everything it finished. COLLECTOR_RESUME=1 makes a
# re-run skip chains whose chunk already exists (only enable it for re-runs
# with the same settings and the same chunk directory).
COLLECTOR_RESUME = os.environ.get("COLLECTOR_RESUME", "0") == "1"

# --- random-start configuration ---
STARTS_PER_MAP = 4
START_MARGIN = step * 2  # keep starts off the map edge, same margin convention
                          # used elsewhere in gaussianprocesstraining.py
START_POSITION_SEED = 990_002  # arbitrary, disjoint from SENSORNOISE_SEED so the
                                # position draw and the sensor-noise stream don't
                                # share a seed value
WARMUP_OFFSET = 40.0  # warmup target = start + this offset, clipped to bounds
# Optional: force specific (cx, cy) starts to always be included alongside the
# random ones, e.g. to guarantee coverage of points other scripts hard-code as
# their own reset position (DPPOTRAINING*.py uses map center;
# Diffusionplanner_singlemap.py uses (4.0, 4.0)). Empty by default.
EXTRA_FIXED_STARTS = []

target_trajectory_len = planning_horizon * samples_per_segment + 1
final_dataset_path = script_dir / "CMAES_beamsearch_dataset_3d_synthetic_final.pt"
chunk_dir = script_dir / "CMAES_beamsearch_dataset_3d_randomstart_multimodal_chunks"


def uniform_grid_start(rng, xmin, xmax, ymin, ymax, step, margin):
    # Uniform rather than Gaussian: a center-biased sampler under-visits
    # starts near the workspace edge, which is exactly the region where a
    # narrow low-altitude FOV and nearby boundary can most affect planning -
    # uniform coverage over the full margin-inset domain treats edge and
    # center starts as equally likely.
    lo_x, hi_x = xmin + margin, xmax - margin
    lo_y, hi_y = ymin + margin, ymax - margin

    cx = float(rng.uniform(lo_x, hi_x))
    cy = float(rng.uniform(lo_y, hi_y))

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
        "continued": [],
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
        "continued": torch.tensor(
            np.asarray(dataset["continued"], dtype=np.int64)
        ),
    }


def dataset_size(dataset):
    return len(dataset["trajectories"])


def chain_chunk_path(selected_map, start_index=None):
    if start_index is None:
        return chunk_dir / f"map_{selected_map:03d}_ranked_3d.pt"
    return chunk_dir / f"map_{selected_map:03d}_start_{start_index:02d}_ranked_3d.pt"


def save_dataset_chunk(dataset, selected_map, start_index=None):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chain_chunk_path(selected_map, start_index)
    # write to a temp name first so a job killed mid-write never leaves a
    # truncated chunk that a resumed run would mistake for a finished chain
    tmp_path = chunk_path.with_suffix(".tmp")
    torch.save(tensorize_dataset(dataset), tmp_path)
    os.replace(tmp_path, chunk_path)
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
        score = gain / (max(distance, step) ** COST_EXPONENT)
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
            score = gain / (max(distance, step) ** COST_EXPONENT)
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


def path_footprint(spline_path, xs, ys, important_mask):
    """Boolean flat mask of the important cells a path observes: the union of
    the FOVs along the points that scoring actually flies
    (EXECUTION_CHUNK_SCORING). Belief-only, no ground truth involved."""
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    nx = len(xs)
    footprint = np.zeros(nx * len(ys), dtype=bool)
    for px, py, pz in spline_path[:EXECUTION_CHUNK_SCORING]:
        radius = fov_lateral_radius(pz, ANGLE_OF_VIEW)
        x_idx = np.flatnonzero((xs >= px - radius) & (xs <= px + radius))
        y_idx = np.flatnonzero((ys >= py - radius) & (ys <= py + radius))
        footprint[(y_idx[:, None] * nx + x_idx[None, :]).ravel()] = True
    return footprint & np.asarray(important_mask, dtype=bool)


def jaccard_distance(a, b):
    union = np.count_nonzero(a | b)
    if union == 0:
        return 0.0
    return 1.0 - np.count_nonzero(a & b) / union


def select_distinct_modes(branch_results, kept_indices, winner_idx, important_mask, xs, ys):
    """Winner first, then the remaining near-optimal candidates in descending
    variance_correction order, each kept only if its Jaccard distance to every
    already-kept footprint is at least JACCARD_MIN_DISTANCE."""
    footprints = {
        winner_idx: path_footprint(
            branch_results[winner_idx]["spline_path"], xs, ys, important_mask
        )
    }
    distinct = [winner_idx]
    others = sorted(
        (i for i in kept_indices if i != winner_idx),
        key=lambda i: -branch_results[i]["variance_correction"],
    )
    for i in others:
        footprint = path_footprint(
            branch_results[i]["spline_path"], xs, ys, important_mask
        )
        if all(
            jaccard_distance(footprint, footprints[j]) >= JACCARD_MIN_DISTANCE
            for j in distinct
        ):
            footprints[i] = footprint
            distinct.append(i)
    return distinct


def cma_es_refine_waypoint_variants(
    flight_plan_3d,
    mu,
    P,
    xs,
    ys,
    cx,
    cy,
    cz,
    beta,
    planner_seed,
    variant_seeds,
):
    """CMA_SOLUTIONS_PER_BRANCH plain CMA-ES refinements of the same warm start,
    one per seed (planner_seed first, then variant_seeds). No diversity term."""
    seeds = [planner_seed] + list(variant_seeds)
    variants = []
    for index, seed in enumerate(seeds[:CMA_SOLUTIONS_PER_BRANCH]):
        waypoints = cma_es_refine_waypoints_3d(
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
            maxiter=CMA_GENERATIONS,
            maxfevals=CMA_MAXFEVALS,
            seed=seed,
        )
        variants.append((f"cma_seed_{index}", waypoints))
    return variants


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


def execute_refined_candidate(
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
    control_waypoints,
    cma_variant,
    cma_variant_index,
):
    sim_cx, sim_cy, sim_cz = cx, cy, cz
    sim_mu = mu.copy()
    sim_P = P.copy()
    sim_rng = np.random.default_rng(rng_seed)

    spline_path = build_spline_trajectory_3d(
        sim_cx,
        sim_cy,
        sim_cz,
        control_waypoints,
        samples_per_segment=samples_per_segment,
    )

    spline_idx = 0
    ticks = 0
    # update_state is a checkpoint of (cx, cy, cz, mu, P) taken the first time
    # the rollout reaches EXECUTION_CHUNK_UPDATING steps - this is what the
    # chain actually commits to for the next round. The loop keeps running
    # past that point up to EXECUTION_CHUNK_SCORING so branches/CMA-variants
    # are still scored (variance/RMSE reduction) on close to the full planned
    # trajectory, not just the short prefix that gets committed. Both numbers
    # come from the same simulated rollout (same rng draws along the same
    # path), so the checkpointed state and the scoring state are causally
    # consistent with each other, not two independent simulations.
    update_state = None
    # Stop once EXECUTION_CHUNK_SCORING WAYPOINTS have been reached (or the
    # plan runs out), not after a fixed tick count - step_along_spline only
    # ever advances spline_idx by at most one per tick, so a fixed tick
    # budget would let branches with tightly-spaced waypoints fly much
    # further into their plan than branches with widely-spaced ones,
    # comparing them on an unequal amount of real progress. max_ticks is just
    # a safety cap against a pathologically slow branch consuming unbounded
    # compute; the out-of-bounds clamp below should make true non-termination
    # impossible.
    max_ticks = EXECUTION_CHUNK_SCORING * 10
    while spline_idx < EXECUTION_CHUNK_SCORING and spline_idx < len(spline_path) and ticks < max_ticks:
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

        if update_state is None and spline_idx >= EXECUTION_CHUNK_UPDATING:
            update_state = (sim_cx, sim_cy, sim_cz, sim_mu.copy(), sim_P.copy())

    if update_state is None:
        # Plan ran out before reaching EXECUTION_CHUNK_UPDATING steps - commit
        # as far as the (shorter) rollout actually went, same as the scoring
        # state.
        update_state = (sim_cx, sim_cy, sim_cz, sim_mu.copy(), sim_P.copy())
    update_cx, update_cy, update_cz, update_mu, update_P = update_state

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
        # Scoring state: the (near-)full EXECUTION_CHUNK_SCORING rollout -
        # used for variance_correction/RMSE_correction, i.e. judging how good
        # this branch/variant is.
        "final_mu": sim_mu,
        "final_P": sim_P,
        "occupied_rmse": rmse["occupied_rmse"],
        "global_rmse": rmse["global_rmse"],
        # Update state: only EXECUTION_CHUNK_UPDATING steps into the same
        # rollout - used to actually advance the chain if this branch/variant
        # wins.
        "update_cx": update_cx,
        "update_cy": update_cy,
        "update_cz": update_cz,
        "update_mu": update_mu,
        "update_P": update_P,
        "cma_variant": cma_variant,
        "cma_variant_index": cma_variant_index,
    }


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
    variant_seeds,
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
    waypoint_variants = cma_es_refine_waypoint_variants(
        flight_plan_3d,
        mu,
        P,
        xs,
        ys,
        cx,
        cy,
        cz,
        beta,
        planner_seed=planner_seed,
        variant_seeds=variant_seeds,
    )
    return [
        execute_refined_candidate(
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
            control_waypoints,
            cma_variant,
            cma_variant_index,
        )
        for cma_variant_index, (cma_variant, control_waypoints) in enumerate(
            waypoint_variants[:CMA_SOLUTIONS_PER_BRANCH]
        )
    ]


def _simulate_branch_worker(payload):
    """Module-level (picklable) wrapper around simulate_candidate for use with
    ProcessPoolExecutor - closures/lambdas aren't picklable under Windows'
    spawn start method, so this has to be a plain top-level function taking a
    single tuple argument."""
    (
        branch_idx, forced_first, beta, cx, cy, cz, mu, P, pts, true_map_flat,
        xs, ys, xmin, xmax, ymin, ymax, samplestep, rng_seed, planner_seed,
        variant_seeds,
    ) = payload
    results = simulate_candidate(
        beta, cx, cy, cz, mu, P, pts, true_map_flat, xs, ys, xmin, xmax, ymin, ymax,
        samplestep, rng_seed=rng_seed, planner_seed=planner_seed,
        variant_seeds=variant_seeds, forced_first=forced_first,
    )
    for result in results:
        result["branch_idx"] = branch_idx
    return results


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
    dataset["continued"].append(int(candidate.get("continued", 0)))


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
        cx, cy = uniform_grid_start(pos_rng, xmin, xmax, ymin, ymax, step, START_MARGIN)
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
    mean = np.full(X_test.shape[0], utility_threshold - 0.1, dtype=float) #pessimistic prior for LCB/NAIP: everywhere starts important
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
                    [
                        make_round_seed(
                            selected_map,
                            start_index,
                            round_idx,
                            branch_idx,
                            purpose=2 + variant_idx,
                        )
                        for variant_idx in range(max(0, CMA_SOLUTIONS_PER_BRANCH - 1))
                    ],
                )
                for branch_idx, forced_first in enumerate(first_candidates)
            ]
            branch_groups = pool.map(_simulate_branch_worker, payloads)
            branch_results = [
                candidate
                for candidate_group in branch_groups
                for candidate in candidate_group
            ]
            branch_results.sort(
                key=lambda r: (r["branch_idx"], r["cma_variant_index"])
            )
            for branch_candidate in branch_results:
                branch_candidate["rmse_correction"] = baseline_rmse - branch_candidate["global_rmse"]
                branch_variance = float(np.sum(np.diag(branch_candidate["final_P"])[importance_mask]))
                branch_candidate["variance_correction"] = baseline_variance - branch_variance
        else:
            branch_results = []
            for branch_idx, forced_first in enumerate(first_candidates):
                branch_candidates = simulate_candidate(
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
                    variant_seeds=[
                        make_round_seed(
                            selected_map,
                            start_index,
                            round_idx,
                            branch_idx,
                            purpose=2 + variant_idx,
                        )
                        for variant_idx in range(max(0, CMA_SOLUTIONS_PER_BRANCH - 1))
                    ],
                    forced_first=forced_first,
                )
                for branch_candidate in branch_candidates:
                    branch_candidate["branch_idx"] = branch_idx
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
                i for i, candidate in enumerate(branch_results)
                if (
                    i == winner_idx
                    or candidate["variance_correction"] / best_correction
                    >= NEIGHBOURHOOD_THRESHOLD
                )
            ]
        else:
            kept_indices = [winner_idx]

        # winner is always parent_beam_index 0; other CMA-ES variants are only
        # recorded if their realized masked-variance reduction is within the
        # neighbourhood threshold of the best solution for this round AND the
        # important-cell footprint they observe is Jaccard-distinct from every
        # already-recorded one.
        ordered_kept = select_distinct_modes(
            branch_results, kept_indices, winner_idx, importance_mask, xs, ys
        )
        group_id = condition_id_for(selected_map, start_index, round_idx)

        # ordered_kept[0] is the winner (best branch). The chain continues from
        # either the winner or, by default, a uniformly drawn kept mode.
        if CONTINUE_RANDOM_KEPT_MODE and len(ordered_kept) > 1:
            continuation_rng = np.random.default_rng(
                make_round_seed(selected_map, start_index, round_idx, 0, purpose=CONTINUATION_SEED_PURPOSE)
            )
            continue_idx = ordered_kept[int(continuation_rng.integers(len(ordered_kept)))]
        else:
            continue_idx = winner_idx

        for parent_beam_index, branch_i in enumerate(ordered_kept):
            branch_candidate = branch_results[branch_i]
            branch_candidate["condition_id"] = group_id
            branch_candidate["continued"] = int(branch_i == continue_idx)
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
        continued = branch_results[continue_idx]
        print(
            f"[MPI {mpi_rank}/{mpi_size}] Map {selected_map}, start {start_index}, "
            f"round {round_idx}: masked variance {baseline_variance:.4f} -> "
            f"{baseline_variance - continued['variance_correction']:.4f} "
            f"(reduction={continued['variance_correction']:.4f}, best={winner['variance_correction']:.4f}, "
            f"continued mode {ordered_kept.index(continue_idx)} of {len(ordered_kept)}), global RMSE "
            f"{baseline_rmse:.4f} -> {continued['global_rmse']:.4f} "
            f"({len(ordered_kept)}/{len(branch_results)} CMA-ES variants kept: "
            f"{len(kept_indices)} within {round(NEIGHBOURHOOD_THRESHOLD * 100)}% of best "
            f"masked variance reduction, {len(ordered_kept)} after Jaccard "
            f">= {JACCARD_MIN_DISTANCE})"
        )

        cx, cy, cz = continued["update_cx"], continued["update_cy"], continued["update_cz"]
        mu, P = continued["update_mu"], continued["update_P"]


def build_chain_tasks(xmin, xmax, ymin, ymax):
    """Every (map, start) chain of the collection, map-major, in a fixed order."""
    tasks = []
    for selected_map in range(initial_map, initial_map + mapcount):
        for start_index, (start_cx, start_cy) in enumerate(
            starts_for_map(selected_map, xmin, xmax, ymin, ymax)
        ):
            tasks.append((selected_map, start_index, start_cx, start_cy))
    return tasks


def run_rank_tasks(mpi_rank, mpi_size, pool=None):
    """Run this rank's share of the chains; returns the chunk paths it produced
    (or found already finished when COLLECTOR_RESUME is on)."""
    _, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, _ = initialize_gp(
        sigma2=GP_KERNEL_SIGMA2,
        lengthscale=GP_KERNEL_LENGTHSCALE,
    )
    tasks = build_chain_tasks(xmin, xmax, ymin, ymax)
    my_tasks = assigned_tasks_for_rank(tasks, mpi_rank, mpi_size)
    if mpi_rank == 0:
        print(
            f"{len(tasks)} chains ({mapcount} maps x {STARTS_PER_MAP} starts) over {mpi_size} ranks: "
            f"{min(len(tasks), mpi_size)} ranks busy, up to {-(-len(tasks) // mpi_size)} chains per rank",
            flush=True,
        )

    chunk_paths = []
    loaded_map = None
    for task_number, (selected_map, start_index, start_cx, start_cy) in enumerate(my_tasks, start=1):
        chunk_path = chain_chunk_path(selected_map, start_index)
        if COLLECTOR_RESUME and chunk_path.exists():
            print(f"[MPI {mpi_rank}/{mpi_size}] Map {selected_map}, start {start_index}: chunk exists, skipping (resume)", flush=True)
            chunk_paths.append(chunk_path)
            continue
        if loaded_map != selected_map:
            pts = load_map(selected_map)
            true_map_flat = build_true_map_flat(pts, X_test)
            loaded_map = selected_map
        print(
            f"[MPI {mpi_rank}/{mpi_size}] Map {selected_map}, start {start_index}: "
            f"({start_cx:.1f}, {start_cy:.1f}) [chain {task_number} of {len(my_tasks)} on this rank]",
            flush=True,
        )
        local_dataset = make_empty_dataset()
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
        chunk_paths.append(save_dataset_chunk(local_dataset, selected_map, start_index))
        del local_dataset
    return chunk_paths


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

    if mpi_rank == 0:
        print(
            f"Running 3D multimodal-branching linear-chain collection with "
            f"MPI size={mpi_size}, BRANCH_COUNT={BRANCH_COUNT}, "
            f"NEIGHBOURHOOD_THRESHOLD={NEIGHBOURHOOD_THRESHOLD}, "
            f"PARALLEL_BRANCHES={PARALLEL_BRANCHES}, RANKLIM={RANKLIM}, "
            f"EXECUTION_CHUNK_SCORING={EXECUTION_CHUNK_SCORING}, "
            f"EXECUTION_CHUNK_UPDATING={EXECUTION_CHUNK_UPDATING}, "
            f"CMA_GENERATIONS={CMA_GENERATIONS}, JACCARD_MIN_DISTANCE={JACCARD_MIN_DISTANCE}, "
            f"COLLECTOR_RESUME={COLLECTOR_RESUME}",
            flush=True,
        )

    # One pool per rank, created once and reused for every chain/round that
    # rank processes - not recreated per round or per chain, so process
    # startup cost is paid once for the rank's entire lifetime.
    pool = ProcessPoolExecutor(max_workers=BRANCH_COUNT) if PARALLEL_BRANCHES else None

    try:
        local_chunk_paths = run_rank_tasks(mpi_rank, mpi_size, pool=pool)
    finally:
        if pool is not None:
            pool.shutdown(wait=True)

    if comm is None:
        all_chunk_paths = local_chunk_paths
    else:
        gathered = comm.gather(local_chunk_paths, root=0)
        all_chunk_paths = [path for paths in gathered for path in paths] if mpi_rank == 0 else []

    if mpi_rank == 0:
        consolidate_chunks(sorted(all_chunk_paths), final_dataset_path, delete_chunks=True)


if __name__ == "__main__":
    main()
