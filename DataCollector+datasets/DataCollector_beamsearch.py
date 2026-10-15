import numpy as np
import matplotlib
import os
import sys

matplotlib.use("Agg")

from pathlib import Path

import torch

script_dir = Path(__file__).resolve().parent
project_root = script_dir
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from gaussianprocesstraining import (
    importance_filter,
    grid_measure,
    next_best_waypoint,
    cma_es_refine_waypoints,
    build_spline_trajectory,
    kalman_update,
    initialize_gp,
)
from evalmetrics import compute_reconstruction_rmse

try:
    from mpi4py import MPI
except ImportError:
    MPI = None



#NEW DYNAMIC DATA COLLECTION 
step = 2.0
RANKLIM = 60
BEAM_WIDTH = 3
beta_values = np.round(np.arange(0.1, 1.0, 0.1), 1)
alpha = 0.02
utility_threshold = 0.5
planning_horizon = 8
initial_map = 60
mapcount = 30
samples_per_segment = 5
execution_chunk = 20
SENSORNOISE_SEED = 123
R = 500
MAPTYPE = "multiblob"
NEAR_TIE_REL_TOL = 0.03
MIN_USEFUL_RMSE_DROP = 1e-6

target_trajectory_len = planning_horizon * samples_per_segment + 1
final_dataset_path = script_dir / "CMAES_beamsearch_dataset_dynamic.pt"
chunk_dir = script_dir / "CMAES_beamsearch_dataset_chunks"


def make_candidate_seed(selected_map, rank, beam_id):
    return SENSORNOISE_SEED + selected_map * 1_000_000 + rank * 1_000 + beam_id


def select_next_beam(candidates, beam_width=BEAM_WIDTH):
    return sorted(candidates, key=lambda item: item["beam_score"])[:beam_width]


def condition_id_for(selected_map, rank, parent_beam_id):
    return selected_map * 1_000_000 + rank * 1_000 + parent_beam_id


def select_record_candidates_for_parent(parent_candidates):
    if not parent_candidates:
        return []

    ranked_by_drop = sorted(
        parent_candidates,
        key=lambda item: item["rmse_correction"],
        reverse=True,
    )
    best_drop = ranked_by_drop[0]["rmse_correction"]

    if best_drop <= MIN_USEFUL_RMSE_DROP:
        return [ranked_by_drop[0]]

    keep_threshold = best_drop * (1.0 - NEAR_TIE_REL_TOL)
    return [
        candidate
        for candidate in ranked_by_drop
        if candidate["rmse_correction"] >= keep_threshold
    ]


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
        "condition_id": torch.tensor(np.asarray(dataset["condition_id"], dtype=np.int64)),
        "parent_beam_id": torch.tensor(np.asarray(dataset["parent_beam_id"], dtype=np.int64)),
        "parent_beam_index": torch.tensor(np.asarray(dataset["parent_beam_index"], dtype=np.int64)),
        "initial_heading_velocity": torch.tensor(
            np.asarray(dataset["initial_heading_velocity"], dtype=np.float32)
        ),
    }


def dataset_size(dataset):
    return len(dataset["trajectories"])


def save_dataset_chunk(dataset, selected_map):
    chunk_dir.mkdir(parents=True, exist_ok=True)
    chunk_path = chunk_dir / f"map_{selected_map:03d}_ranked.pt"
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

    if len(path) == target_len:
        return path
    if len(path) == 0:
        return np.zeros((target_len, 2), dtype=np.float32)
    if len(path) == 1:
        return np.repeat(path, target_len, axis=0)

    source_t = np.linspace(0.0, 1.0, len(path))
    target_t = np.linspace(0.0, 1.0, target_len)
    x = np.interp(target_t, source_t, path[:, 0])
    y = np.interp(target_t, source_t, path[:, 1])
    return np.stack([x, y], axis=-1).astype(np.float32)


def dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax, buffer=None):
    if buffer is None:
        buffer = step * 2

    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    return cx, cy


def waypoint(cx, cy, goal_x, goal_y, step):
    dx = goal_x - cx
    dy = goal_y - cy
    dist = np.hypot(dx, dy)

    if dist <= step:
        return 0.0, 0.0, True

    return dx / dist, dy / dist, False


def build_sensor_update(cx, cy, pts, X_test, N, lateral_coverage, step, rng):
    fov = [
        (x, y)
        for x in np.arange(cx - lateral_coverage, cx + lateral_coverage + 1e-9, step)
        for y in np.arange(cy - lateral_coverage, cy + lateral_coverage + 1e-9, step)
    ]

    sensor = np.zeros((len(fov), N))
    for i, (x_meas, y_meas) in enumerate(fov):
        idx = np.where(
            np.isclose(X_test[:, 0], x_meas) & np.isclose(X_test[:, 1], y_meas)
        )[0]
        if len(idx) > 0:
            sensor[i, idx[0]] = 1.0

    measurement_list = []
    for x_fov, y_fov in fov:
        idx = np.where(
            np.isclose(pts[:, 0], x_fov) & np.isclose(pts[:, 1], y_fov)
        )[0]
        measurement_list.append(pts[idx[0], 2] if idx.size > 0 else 0.0)

    z_meas = np.asarray(measurement_list) + rng.normal(0, np.sqrt(R), size=len(fov))
    return sensor, z_meas


def apply_measurement_update(cx, cy, mu, P, pts, X_test, N, lateral_coverage, step, rng):
    sensor, z_meas = build_sensor_update(
        cx,
        cy,
        pts,
        X_test,
        N,
        lateral_coverage,
        step,
        rng,
    )
    return kalman_update(mu, P, sensor, z_meas, R)


def real_receding_horizon_planner(
    cx,
    cy,
    mu,
    P,
    xs,
    ys,
    utility_threshold,
    beta,
    planning_horizon,
    alpha=0.1,
    R=None,
    lateral_coverage=None,
):
    flight_plan = []
    filtered_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    utility_at_gridpoints = grid_measure(filtered_utility, xs, ys)
    curr_x, curr_y = cx, cy

    for _ in range(planning_horizon):
        best_waypoint = next_best_waypoint(
            utility_at_gridpoints,
            curr_x,
            curr_y,
            alpha=alpha,
        )
        flight_plan.append(best_waypoint)
        curr_x, curr_y = best_waypoint
        utility_at_gridpoints = [
            (util, point)
            for util, point in utility_at_gridpoints
            if point != best_waypoint
        ]

    return cma_es_refine_waypoints(
        flight_plan,
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
        predictive_variance=True,
    )


def step_along_spline(cx, cy, spline_path, spline_idx, samplestep, xmin, xmax, ymin, ymax):
    if spline_idx >= len(spline_path):
        return cx, cy, spline_idx

    goal_x, goal_y = spline_path[spline_idx]
    grad_x, grad_y, waypoint_reached = waypoint(cx, cy, goal_x, goal_y, step)

    if waypoint_reached:
        spline_idx += 1
        if spline_idx < len(spline_path):
            goal_x, goal_y = spline_path[spline_idx]
            grad_x, grad_y, _ = waypoint(cx, cy, goal_x, goal_y, step)
        else:
            grad_x, grad_y = 0.0, 0.0

    cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
    return cx, cy, spline_idx


def simulate_candidate(
    beta,
    cx,
    cy,
    mu,
    P,
    pts,
    X_test,
    N,
    xs,
    ys,
    xmin,
    xmax,
    ymin,
    ymax,
    lateral_coverage,
    samplestep,
    rng_seed,
):
    sim_cx, sim_cy = cx, cy
    sim_mu = mu.copy()
    sim_P = P.copy()
    sim_rng = np.random.default_rng(rng_seed)

    control_waypoints = real_receding_horizon_planner(
        sim_cx,
        sim_cy,
        sim_mu,
        sim_P,
        xs,
        ys,
        utility_threshold,
        beta,
        planning_horizon,
        alpha=alpha,
        R=R,
        lateral_coverage=lateral_coverage,
    )
    spline_path = build_spline_trajectory(
        sim_cx,
        sim_cy,
        control_waypoints,
        samples_per_segment=samples_per_segment,
    )

    spline_idx = 0
    for _ in range(execution_chunk):
        sim_cx, sim_cy, spline_idx = step_along_spline(
            sim_cx,
            sim_cy,
            spline_path,
            spline_idx,
            samplestep,
            xmin,
            xmax,
            ymin,
            ymax,
        )
        sim_mu, sim_P = apply_measurement_update(
            sim_cx,
            sim_cy,
            sim_mu,
            sim_P,
            pts,
            X_test,
            N,
            lateral_coverage,
            step,
            sim_rng,
        )

    rmse = compute_reconstruction_rmse(
        mu=sim_mu,
        pts=pts,
        xs=xs,
        ys=ys,
        step=step,
        xmin=xmin,
        ymin=ymin,
    )

    return {
        "beta": float(beta),
        "control_waypoints": control_waypoints,
        "spline_path": spline_path,
        "final_cx": sim_cx,
        "final_cy": sim_cy,
        "final_mu": sim_mu,
        "final_P": sim_P,
        "weighted_rmse": rmse["weighted_rmse"],
        "occupied_rmse": rmse["occupied_rmse"],
        "global_rmse": rmse["global_rmse"],
    }


def record_candidate(dataset, candidate, state, selected_map, rank, map_shape):
    trajectory = resample_trajectory(candidate["spline_path"], target_trajectory_len)
    current_position = np.asarray([state["cx"], state["cy"]], dtype=np.float32)
    if trajectory.shape[0] < 2:
        raise ValueError(f"Expected at least two trajectory points, got shape {trajectory.shape}")
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
    dataset["timestep"].append(rank)
    dataset["RMSE_correction"].append(candidate["rmse_correction"])
    dataset["condition_id"].append(candidate["condition_id"])
    dataset["parent_beam_id"].append(candidate["parent_beam_id"])
    dataset["parent_beam_index"].append(candidate["parent_beam_index"])
    dataset["initial_heading_velocity"].append(initial_heading_velocity.astype(np.float32))


def load_map(selected_map):
    csv_path = project_root / "csv"
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


def main():
    comm, mpi_rank, mpi_size = get_mpi_context()
    if detect_mpi_launch_issue(os.environ, mpi_size):
        raise RuntimeError(
            "SLURM_NTASKS is greater than 1, but mpi4py sees MPI size 1. "
            "This means the script is being launched as repeated serial jobs, "
            "not one MPI job. Use an MPI-aware launch such as "
            "`srun --mpi=pmix python DataCollector_beamsearch.py` or "
            "`mpiexec -n $SLURM_NTASKS python DataCollector_beamsearch.py`."
        )

    chunk_paths = []
    if mpi_rank == 0:
        print(f"Running beam search with MPI size={mpi_size}")

    for selected_map in range(initial_map, initial_map + mapcount):
        if mpi_rank == 0:
            print(f"\nCollecting ranked CMA-ES data for map {selected_map}")
            map_dataset = make_empty_dataset()
        else:
            map_dataset = None

        pts = load_map(selected_map)
        _, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, _ = initialize_gp()

        N = X_test.shape[0]
        cx, cy = 20.0, 20.0
        mu = mean.copy()
        P = cov.copy()
        lateral_coverage = step * 2
        samplestep = step
        rng = np.random.default_rng(SENSORNOISE_SEED + selected_map)

        for _ in range(2):
            grad_x, grad_y, _ = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, step=step)
            cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
            mu, P = apply_measurement_update(
                cx,
                cy,
                mu,
                P,
                pts,
                X_test,
                N,
                lateral_coverage,
                step,
                rng,
            )

        beam = [
            {
                "cx": cx,
                "cy": cy,
                "mu": mu.copy(),
                "P": P.copy(),
                "beam_score": 0.0,
                "beam_id": 0,
            }
        ]

        for rank in range(RANKLIM):
            beam_contexts = []
            tasks = []

            for beam_index, state in enumerate(beam):
                utility = importance_filter(
                    state["mu"],
                    state["P"],
                    beta_values[-1],
                    threshold=utility_threshold,
                )
                baseline_rmse = compute_reconstruction_rmse(
                    mu=state["mu"],
                    pts=pts,
                    xs=xs,
                    ys=ys,
                    step=step,
                    xmin=xmin,
                    ymin=ymin,
                )["weighted_rmse"]
                record_state = {
                    "cx": state["cx"],
                    "cy": state["cy"],
                    "mu": state["mu"].copy(),
                    "P": state["P"].copy(),
                    "utility": utility.copy(),
                }
                beam_contexts.append(
                    {
                        "state": state,
                        "record_state": record_state,
                        "baseline_rmse": baseline_rmse,
                    }
                )

                candidate_seed = make_candidate_seed(selected_map, rank, beam_index)
                for beta in beta_values:
                    tasks.append((beam_index, float(beta), candidate_seed))

            local_candidates = []
            local_tasks = assigned_tasks_for_rank(tasks, mpi_rank, mpi_size)
            for beam_index, beta, candidate_seed in local_tasks:
                state = beam_contexts[beam_index]["state"]
                print(
                    f"[MPI {mpi_rank}/{mpi_size}] Map {selected_map}, rank {rank}, "
                    f"beam {beam_index}, simulating beta {beta:.1f}"
                )
                candidate = simulate_candidate(
                    beta,
                    state["cx"],
                    state["cy"],
                    state["mu"],
                    state["P"],
                    pts,
                    X_test,
                    N,
                    xs,
                    ys,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    lateral_coverage,
                    samplestep,
                    rng_seed=candidate_seed,
                )
                candidate["beam_score"] = state["beam_score"] + candidate["weighted_rmse"]
                candidate["baseline_rmse"] = beam_contexts[beam_index]["baseline_rmse"]
                candidate["rmse_correction"] = (
                    candidate["baseline_rmse"] - candidate["weighted_rmse"]
                )
                candidate["parent_beam_index"] = beam_index
                candidate["parent_beam_id"] = state["beam_id"]
                candidate["condition_id"] = condition_id_for(
                    selected_map,
                    rank,
                    state["beam_id"],
                )
                local_candidates.append(candidate)

            if comm is None:
                gathered_candidates = [local_candidates]
            else:
                gathered_candidates = comm.gather(local_candidates, root=0)

            if mpi_rank == 0:
                candidates = [
                    candidate
                    for rank_candidates in gathered_candidates
                    for candidate in rank_candidates
                ]
                recorded_count = 0
                for parent_beam_index, context in enumerate(beam_contexts):
                    parent_candidates = [
                        candidate
                        for candidate in candidates
                        if candidate["parent_beam_index"] == parent_beam_index
                    ]
                    record_candidates = select_record_candidates_for_parent(parent_candidates)
                    for candidate in record_candidates:
                        record_candidate(
                            map_dataset,
                            candidate,
                            context["record_state"],
                            selected_map,
                            rank,
                            X.shape,
                        )
                    recorded_count += len(record_candidates)

                next_candidates = select_next_beam(candidates, beam_width=BEAM_WIDTH)
                beam = []
                for next_beam_id, candidate in enumerate(next_candidates):
                    beam.append(
                        {
                            "cx": candidate["final_cx"],
                            "cy": candidate["final_cy"],
                            "mu": candidate["final_mu"],
                            "P": candidate["final_P"],
                            "beam_score": candidate["beam_score"],
                            "beam_id": next_beam_id,
                        }
                    )

                print(
                    f"Map {selected_map}, rank {rank}: "
                    f"best beta={next_candidates[0]['beta']:.1f}, "
                    f"parent beam={next_candidates[0]['parent_beam_id']}, "
                    f"weighted RMSE {next_candidates[0]['baseline_rmse']:.4f} "
                    f"-> {next_candidates[0]['weighted_rmse']:.4f}, "
                    f"beam size={len(beam)}, recorded samples={recorded_count}"
                )
                del candidates, next_candidates
            else:
                beam = None

            if comm is not None:
                beam = comm.bcast(beam, root=0)

            del local_candidates, local_tasks, beam_contexts, tasks

        if mpi_rank == 0:
            chunk_paths.append(save_dataset_chunk(map_dataset, selected_map))
            del map_dataset
        del pts, X_test, mean, cov, xs, ys, X, Y, mu, P, beam

    if mpi_rank == 0:
        consolidate_chunks(chunk_paths, final_dataset_path, delete_chunks=True)


if __name__ == "__main__":
    main()
