import numpy as np
import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import torch

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


step = 2.0
RANKLIM = 200
TOP_K = 4
beta_values = np.round(np.arange(0.1, 1.0, 0.1), 1)
alpha = 0.02
utility_threshold = 0.5
planning_horizon = 8
initial_map = 1
mapcount = 1
samples_per_segment = 5
execution_chunk = 20
SENSORNOISE_SEED = 123
R = 500
MAPTYPE = "multiblob"

target_trajectory_len = planning_horizon * samples_per_segment + 1
script_dir = Path(__file__).resolve().parent
final_dataset_path = script_dir / "CMAES_ranked_dataset.pt"
chunk_dir = script_dir / "CMAES_ranked_dataset_chunks"


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
        "final_mu": sim_mu,
        "final_P": sim_P,
        "occupied_rmse": rmse["occupied_rmse"],
        "global_rmse": rmse["global_rmse"],
    }


def record_candidate(dataset, candidate, state, selected_map, rank, map_shape, baseline_rmse):
    dataset["trajectories"].append(
        resample_trajectory(candidate["spline_path"], target_trajectory_len)
    )
    dataset["control_waypoints"].append(
        np.asarray(candidate["control_waypoints"], dtype=np.float32)
    )
    dataset["current_position"].append(
        np.asarray([state["cx"], state["cy"]], dtype=np.float32)
    )
    dataset["current_mean"].append(state["mu"].reshape(map_shape).astype(np.float32))
    dataset["current_var"].append(np.diag(state["P"]).reshape(map_shape).astype(np.float32))
    dataset["current_util"].append(state["utility"].reshape(map_shape).astype(np.float32))
    dataset["map_id"].append(selected_map)
    dataset["beta"].append(candidate["beta"])
    dataset["timestep"].append(rank)
    dataset["RMSE_correction"].append(baseline_rmse - candidate["occupied_rmse"])


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


def main():
    chunk_paths = []

    for selected_map in range(initial_map, initial_map + mapcount):
        print(f"\nCollecting ranked CMA-ES data for map {selected_map}")
        map_dataset = make_empty_dataset()
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

        for rank in range(RANKLIM):
            utility = importance_filter(mu, P, beta_values[-1], threshold=utility_threshold)
            baseline_rmse = compute_reconstruction_rmse(
                mu=mu,
                pts=pts,
                xs=xs,
                ys=ys,
                step=step,
                utility_threshold=utility_threshold,
                xmin=xmin,
                ymin=ymin,
            )["occupied_rmse"]
            state = {
                "cx": cx,
                "cy": cy,
                "mu": mu.copy(),
                "P": P.copy(),
                "utility": utility.copy(),
            }

            candidates = []
            # candidate_seed = SENSORNOISE_SEED + selected_map * 100_000 + rank
            for beta in beta_values:
                print(f"Map {selected_map}, rank {rank}, simulating beta {beta:.1f}")
                candidates.append(
                    simulate_candidate(
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
                        rng_seed=rng,
                    )
                )

            candidates.sort(key=lambda item: item["occupied_rmse"])
            for candidate in candidates[:TOP_K]:
                record_candidate(
                    map_dataset,
                    candidate,
                    state,
                    selected_map,
                    rank,
                    X.shape,
                    baseline_rmse,
                )

            best = candidates[0]
            cx = best["final_cx"]
            cy = best["final_cy"]
            mu = best["final_mu"]
            P = best["final_P"]
            print(
                f"Map {selected_map}, rank {rank}: best beta={best['beta']:.1f}, "
                f"occupied RMSE {baseline_rmse:.4f} -> {best['occupied_rmse']:.4f}"
            )
            del candidates

        chunk_paths.append(save_dataset_chunk(map_dataset, selected_map))
        del map_dataset, pts, X_test, mean, cov, xs, ys, X, Y, mu, P

    consolidate_chunks(chunk_paths, final_dataset_path, delete_chunks=True)


if __name__ == "__main__":
    main()
