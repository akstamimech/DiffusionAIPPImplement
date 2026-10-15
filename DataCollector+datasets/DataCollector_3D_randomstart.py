import os
import sys
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
    build_sensor_matrix,
    build_spline_trajectory_3d,
    cma_es_refine_waypoints_3d,
    fov_grid_points,
    grid_search_3d,
    importance_filter,
    initialize_gp,
    kalman_update,
    noise_model,
    sample_correlated_sensor_noise,
)

try:
    from mpi4py import MPI
except ImportError:
    MPI = None


# Copy of DataCollector_3D.py. The physics/sensor/planner stack (dynamics_3d,
# apply_measurement_update_3d, grid_search_3d, cma_es_refine_waypoints_3d) is
# used exactly as it currently lives in CMAES_classic_singlemap.py and
# gaussianprocesstraining.py - unchanged. What differs from DataCollector_3D.py:
#   - no beam search. Each map is rolled out as STARTS_PER_MAP independent
#     linear chains (one per random initial (cx, cy)), each doing
#     plan -> execute execution_chunk steps -> record -> repeat for RANKLIM
#     rounds, with no candidate fan-out/pruning/near-tie retention.
#   - MPI parallelizes across chains directly (each rank runs some subset of
#     the chains start-to-finish independently), instead of parallelizing
#     candidate evaluation within a shared round. There's no per-round
#     MPI sync needed anymore - only one gather per map, of the already-
#     compact per-rank dataset dicts.
#   - beta_values (a tuple of 5 identical 1.0 entries used only to fan a round
#     out into sibling candidates) collapses to a single BETA=1.0 constant,
#     since there's nothing left to fan out - the "5" now means 5 starts.
#   - starting position is a random grid-aligned (cx, cy) instead of the fixed
#     corner (4.0, 4.0); the warmup target is an offset relative to the
#     (random) start rather than the fixed point (80, 80), since a fixed
#     warmup target becomes degenerate when the random start lands near it.
#   - dataset schema is unchanged from DataCollector_3D_randomstart's
#     beam-search version (same field names), but parent_beam_id/
#     parent_beam_index are now always 0 (there's no beam) - kept only for
#     schema compatibility with anything already reading these files.

step = 2.0
RANKLIM = 50
BETA = 1.0
alpha = 0.02
utility_threshold = 0.3
planning_horizon = 8
initial_map = 1
mapcount = 30
samples_per_segment = 5
execution_chunk = 20
SENSORNOISE_SEED = 123
MAPTYPE = "NAIP"
INIT_ALTITUDE = 10.0
ZMIN = 10.0
ZMAX = 40.0
ANGLE_OF_VIEW = 60.0

# --- random-start configuration ---
STARTS_PER_MAP = 5
START_MARGIN = step * 2  # keep starts off the map edge, same margin convention
                          # used elsewhere in gaussianprocesstraining.py
START_POSITION_SEED = 990_001  # arbitrary, disjoint from SENSORNOISE_SEED so the
                                # position draw and the sensor-noise stream don't
                                # share a seed value
WARMUP_OFFSET = 40.0  # warmup target = start + this offset, clipped to bounds
# Optional: force specific (cx, cy) starts to always be included alongside the
# random ones, e.g. to guarantee coverage of points other scripts hard-code as
# their own reset position (DPPOTRAINING*.py uses map center;
# Diffusionplanner_singlemap.py uses (4.0, 4.0)). Empty by default.
EXTRA_FIXED_STARTS = []

target_trajectory_len = planning_horizon * samples_per_segment + 1
final_dataset_path = script_dir / "CMAES_beamsearch_dataset_3d_randomstart.pt"
chunk_dir = script_dir / "CMAES_beamsearch_dataset_3d_randomstart_chunks"


def random_grid_start(rng, xmin, xmax, ymin, ymax, step, margin):
    lo_x, hi_x = xmin + margin, xmax - margin
    lo_y, hi_y = ymin + margin, ymax - margin
    n_x = int(round((hi_x - lo_x) / step)) + 1
    n_y = int(round((hi_y - lo_y) / step)) + 1
    cx = lo_x + step * rng.integers(0, n_x)
    cy = lo_y + step * rng.integers(0, n_y)
    return float(cx), float(cy)


def make_round_seed(selected_map, start_index, round_idx, purpose=0):
    """A valid numpy/cma RNG seed derived from the map/start/round/purpose
    indices, via SeedSequence rather than raw multiply-and-add. numpy
    requires seeds in [0, 2**32 - 1]; the old formula (SENSORNOISE_SEED +
    selected_map*100_000_000 + start_index*1_000_000 + round_idx*1_000)
    silently overflowed that range once selected_map reached ~43, crashing
    with "Seed must be between 0 and 2**32 - 1". SeedSequence hashes
    arbitrarily large/many integer inputs into a valid seed, so this can't
    overflow regardless of how large mapcount/RANKLIM/STARTS_PER_MAP get.
    purpose distinguishes the sensor-noise seed from the CMA-ES planner seed
    for the same (map, start, round) - previously done via `round_seed +
    10000`, which itself could have overflowed if round_seed landed near the
    top of the valid range.
    """
    return int(
        np.random.SeedSequence(
            [SENSORNOISE_SEED, selected_map, start_index, round_idx, purpose]
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
):
    flight_plan_3d = grid_search_3d(
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
    # lets plans with widely-spaced waypoints fly through far less of their
    # plan than ones with tightly-spaced waypoints in the same execution_chunk
    # ticks. max_ticks is a safety cap against a pathologically slow plan
    # consuming unbounded compute; the out-of-bounds clamp above should make
    # true non-termination impossible.
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


def record_candidate(
    dataset, candidate, state, selected_map, round_idx, map_shape, start_position, start_index
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
    dataset["condition_id"].append(candidate["condition_id"])
    dataset["parent_beam_id"].append(0)
    dataset["parent_beam_index"].append(0)
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
        cx, cy = random_grid_start(pos_rng, xmin, xmax, ymin, ymax, step, START_MARGIN)
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
):
    mean = np.full(X_test.shape[0], utility_threshold - 0.1, dtype=float)
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

    for round_idx in range(RANKLIM):
        utility = importance_filter(mu, P, BETA, threshold=utility_threshold)
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

        candidate = simulate_candidate(
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
            rng_seed=make_round_seed(selected_map, start_index, round_idx, purpose=0),
            planner_seed=make_round_seed(selected_map, start_index, round_idx, purpose=1),
        )
        candidate["rmse_correction"] = baseline_rmse - candidate["global_rmse"]
        candidate["condition_id"] = condition_id_for(selected_map, start_index, round_idx)

        record_candidate(
            local_dataset,
            candidate,
            record_state,
            selected_map,
            round_idx,
            X.shape,
            (start_cx, start_cy),
            start_index,
        )

        print(
            f"[MPI {mpi_rank}/{mpi_size}] Map {selected_map}, start {start_index}, "
            f"round {round_idx}: global RMSE {baseline_rmse:.4f} -> {candidate['global_rmse']:.4f}"
        )

        cx, cy, cz = candidate["final_cx"], candidate["final_cy"], candidate["final_cz"]
        mu, P = candidate["final_mu"], candidate["final_P"]


def main():
    comm, mpi_rank, mpi_size = get_mpi_context()
    if detect_mpi_launch_issue(os.environ, mpi_size):
        raise RuntimeError(
            "SLURM_NTASKS is greater than 1, but mpi4py sees MPI size 1. "
            "This means the script is being launched as repeated serial jobs, "
            "not one MPI job. Use an MPI-aware launch such as "
            "`srun --mpi=pmix python DataCollector_3D_randomstart.py` or "
            "`mpiexec -n $SLURM_NTASKS python DataCollector_3D_randomstart.py`."
        )

    chunk_paths = []
    if mpi_rank == 0:
        print(f"Running 3D linear-chain collection with MPI size={mpi_size}")

    for selected_map in range(initial_map, initial_map + mapcount):
        if mpi_rank == 0:
            print(f"\nCollecting 3D CMA-ES rollout data for map {selected_map}")

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

    if mpi_rank == 0:
        consolidate_chunks(chunk_paths, final_dataset_path, delete_chunks=True)


if __name__ == "__main__":
    main()
