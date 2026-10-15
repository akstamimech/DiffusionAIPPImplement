import argparse
import csv
from pathlib import Path

import numpy as np

from gaussianprocesstraining import (
    build_correlated_noise_covariance,
    build_sensor_matrix,
    fov_lateral_radius,
    fov_grid_points,
    initialize_gp,
    kalman_update,
    noise_model,
    sample_correlated_sensor_noise,
)
from evalmetrics import (
    compute_coverage_efficiency,
    compute_reconstruction_rmse,
    compute_rmse_time_metrics,
    compute_task_completion,
)


SCRIPT_DIR = Path(__file__).resolve().parent

step = 2.0
timealloted = 150
MAP_ID_START = 20
MAP_ID_END = 25
RUNS_PER_MAP = 1
MAPTYPE = "NAIP"
SENSORNOISE_SEED = 123
output_root = SCRIPT_DIR / "lawnmower_batch_metrics"
utility_threshold = 0.3
INIT_ALTITUDE = 10.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate lawnmower planner over an inclusive range of map ids, "
            "running each map multiple times with different sensor-noise seeds."
        )
    )
    parser.add_argument("--map-start", type=int, default=MAP_ID_START)
    parser.add_argument("--map-end", type=int, default=MAP_ID_END)
    parser.add_argument("--runs-per-map", type=int, default=RUNS_PER_MAP)
    parser.add_argument("--output-dir", type=Path, default=output_root)
    parser.add_argument("--summary-name", default="lawnmower_batch_rmse_summary.csv")
    return parser.parse_args()


def iter_map_runs(map_start, map_end, runs_per_map):
    if map_end < map_start:
        raise ValueError(f"map_end ({map_end}) must be >= map_start ({map_start})")
    if runs_per_map < 1:
        raise ValueError(f"runs_per_map must be >= 1, got {runs_per_map}")

    for selected_map in range(map_start, map_end + 1):
        for run_index in range(runs_per_map):
            yield selected_map, run_index


def write_metric_summary(metriclist, summary_path):
    if not metriclist:
        print("No metrics to write.")
        return

    fieldnames = list(metriclist[0].keys())
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metriclist)
    print(f"\nSaved summary metrics to {summary_path}")


def load_map_context(selected_map, xs, ys, xmin, ymin, gp_step):
    data = np.loadtxt(
        SCRIPT_DIR / "csv" / f"map_{selected_map}_{MAPTYPE}_grid_counts.csv",
        delimiter=",",
        skiprows=1,
    )

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], gp_step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], gp_step), 0.0, atol=tol)
    )
    pts = pts[mask]

    true_map = np.zeros((len(ys), len(xs)), dtype=np.float32)
    x_idx = np.rint((pts[:, 0] - xmin) / gp_step).astype(int)
    y_idx = np.rint((pts[:, 1] - ymin) / gp_step).astype(int)
    valid = (0 <= x_idx) & (x_idx < len(xs)) & (0 <= y_idx) & (y_idx < len(ys))
    true_map[y_idx[valid], x_idx[valid]] = pts[valid, 2]

    return {
        "pts": pts,
        "true_map": true_map,
    }


def grid_spaced_values(start, stop, spacing, gp_step):
    direction = 1.0 if stop >= start else -1.0
    values = list(np.arange(start, stop + direction * 1e-9, direction * spacing))
    if not values or not np.isclose(values[-1], stop):
        values.append(stop)
    return [float(gp_step * np.round(value / gp_step)) for value in values]


def lawnmower_planner(cx, cy, xmin, xmax, ymin, ymax, gp_step, fov_radius):
    buffer = gp_step * 2
    x_left = xmin + buffer
    x_right = xmax - buffer
    y_top = ymax - buffer
    sweep_spacing = max(gp_step, gp_step * np.round((2.0 * fov_radius) / gp_step))

    cx = gp_step * np.round(cx / gp_step)
    cy = gp_step * np.round(cy / gp_step)

    flight_plan = []
    y_values = grid_spaced_values(cy, y_top, sweep_spacing, gp_step)

    for row_idx, y in enumerate(y_values):
        if row_idx == 0:
            x_values = grid_spaced_values(cx, x_right, sweep_spacing, gp_step)
        elif row_idx % 2 == 1:
            x_values = grid_spaced_values(x_right, x_left, sweep_spacing, gp_step)
        else:
            x_values = grid_spaced_values(x_left, x_right, sweep_spacing, gp_step)

        for x in x_values:
            next_waypoint = (
                float(gp_step * np.round(x / gp_step)),
                float(gp_step * np.round(y / gp_step)),
            )
            if (
                len(flight_plan) == 0
                and np.isclose(next_waypoint[0], cx)
                and np.isclose(next_waypoint[1], cy)
            ):
                continue
            flight_plan.append(next_waypoint)

    return flight_plan


def waypoint(cx, cy, goal_x, goal_y, gp_step):
    dx = goal_x - cx
    dy = goal_y - cy
    dist = np.hypot(dx, dy)

    if dist <= gp_step:
        return 0.0, 0.0, True

    grad_x = dx / dist
    grad_y = dy / dist
    return grad_x, grad_y, False


def dynamics(cx, cy, grad_x, grad_y, gp_step, samplestep, xmin, xmax, ymin, ymax, buffer=None):
    if buffer is None:
        buffer = gp_step * 2

    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cx = gp_step * np.round(cx / gp_step)
    cy = gp_step * np.round(cy / gp_step)
    return cx, cy


def run_map(
    selected_map,
    map_context,
    gp_context,
    run_index=0,
    batch_output_root=output_root,
):
    output_dir = batch_output_root / f"lawnmower_map_{selected_map}_run_{run_index:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, mean, cov, xs, ys, _, _, xmin, xmax, ymin, ymax, gp_step = gp_context
    pts = map_context["pts"]
    true_map = map_context["true_map"]
    true_map_flat = true_map.ravel()

    mu = np.full_like(mean, utility_threshold, dtype=np.float32)
    P = cov.copy()
    R = noise_model(INIT_ALTITUDE)
    sensor_seed = SENSORNOISE_SEED + selected_map * 1000 + run_index
    rng = np.random.default_rng(sensor_seed)

    pos_history = []
    rmselist = []
    global_rmselist = []

    samplestep = gp_step

    cx, cy, cz = 4.0, 4.0, INIT_ALTITUDE
    sensor_footprint_radius = fov_lateral_radius(cz)
    grad_x, grad_y = 0.0, 0.0
    pos_history.append((cx, cy))

    initial_total_variance = np.sum(np.diag(P))
    print(
        f"Map {selected_map} run {run_index}: Initial total variance: "
        f"{initial_total_variance:.4f}, sensor_seed={sensor_seed}"
    )

    flight_plan = []

    for ts in range(0, timealloted):
        if ts <= 1:
            grad_x, grad_y, _ = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, gp_step=gp_step)
            cx, cy = dynamics(cx, cy, grad_x, grad_y, gp_step, samplestep, xmin, xmax, ymin, ymax)
            pos_history.append((cx, cy))
        else:
            if ts == 2:
                flight_plan = lawnmower_planner(
                    cx,
                    cy,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    gp_step=gp_step,
                    fov_radius=sensor_footprint_radius,
                )

            if flight_plan:
                grad_x, grad_y, waypoint_reached = waypoint(
                    cx,
                    cy,
                    goal_x=flight_plan[0][0],
                    goal_y=flight_plan[0][1],
                    gp_step=gp_step,
                )
                if waypoint_reached:
                    flight_plan.pop(0)
                    if flight_plan:
                        grad_x, grad_y, _ = waypoint(
                            cx,
                            cy,
                            goal_x=flight_plan[0][0],
                            goal_y=flight_plan[0][1],
                            gp_step=gp_step,
                        )
                    else:
                        grad_x, grad_y = 0.0, 0.0
            else:
                grad_x, grad_y = 0.0, 0.0

            cx, cy = dynamics(cx, cy, grad_x, grad_y, gp_step, samplestep, xmin, xmax, ymin, ymax)
            pos_history.append((cx, cy))

        fov = fov_grid_points(cx, cy, cz, xs, ys)
        sensor, sensor_block_ids = build_sensor_matrix(
            fov, cz, xs, ys, return_block_ids=True
        )
        R = noise_model(cz)

        z_meas = sensor @ true_map_flat
        z_meas += sample_correlated_sensor_noise(sensor_block_ids, R, rng)
        R_cov = build_correlated_noise_covariance(sensor_block_ids, R)

        mu, P = kalman_update(
            mu, P, sensor, z_meas, R_cov, block_ids=sensor_block_ids
        )

        reconstruction_metrics = compute_reconstruction_rmse(
            mu=mu,
            pts=pts,
            xs=xs,
            ys=ys,
            step=gp_step,
            utility_threshold=utility_threshold,
            xmin=xmin,
            ymin=ymin,
        )
        rmselist.append(reconstruction_metrics["occupied_rmse"])
        global_rmselist.append(reconstruction_metrics["global_rmse"])

    rmse_trace = np.column_stack([global_rmselist, rmselist])
    np.savetxt(
        output_dir / f"map_{selected_map}_run_{run_index:02d}_rmse_over_time.csv",
        rmse_trace,
        delimiter=",",
        header="global_rmse,occupied_rmse",
        comments="",
    )

    final_variance = np.sum(np.diag(P))
    variance_delta = initial_total_variance - final_variance

    print(f"Map {selected_map} run {run_index}: Final total variance: {final_variance:.4f}")
    print(f"Map {selected_map} run {run_index}: Variance reduction: {variance_delta:.4f}")

    metrics = compute_task_completion(
        pos_history=pos_history,
        pts=pts,
        xs=xs,
        ys=ys,
        step=gp_step,
        lateral_coverage=sensor_footprint_radius,
        xmin=xmin,
        ymin=ymin,
    )
    coverage_efficiency = compute_coverage_efficiency(metrics["task_completion"], timealloted + 1)

    global_rmse_time = compute_rmse_time_metrics(global_rmselist)
    occupied_rmse_time = compute_rmse_time_metrics(rmselist)

    print(
        f"Map {selected_map} run {run_index}: Gained Utility = {metrics['gained_true_utility']:.4f}, "
        f"Total Utility = {metrics['total_true_utility']:.4f}, "
        f"Task Completion = {metrics['task_completion']:.4%}"
    )
    print(f"Map {selected_map} run {run_index}: Coverage Efficiency = {coverage_efficiency:.4f}")
    print(
        f"Map {selected_map} run {run_index}: Global RMSE = {reconstruction_metrics['global_rmse']:.4f}, "
        f"Occupied RMSE = {reconstruction_metrics['occupied_rmse']:.4f}"
    )

    return {
        "map_id": selected_map,
        "run_index": run_index,
        "sensor_seed": sensor_seed,
        "gained_true_utility": metrics["gained_true_utility"],
        "total_true_utility": metrics["total_true_utility"],
        "task_completion": metrics["task_completion"],
        "coverage_efficiency": coverage_efficiency,
        "global_final_rmse": global_rmse_time["final_rmse"],
        "global_mean_rmse": global_rmse_time["mean_rmse"],
        "global_auc_rmse": global_rmse_time["auc_rmse"],
        "occupied_final_rmse": occupied_rmse_time["final_rmse"],
        "occupied_mean_rmse": occupied_rmse_time["mean_rmse"],
        "occupied_auc_rmse": occupied_rmse_time["auc_rmse"],
        "final_total_variance": final_variance,
        "variance_reduction": variance_delta,
    }


if __name__ == "__main__":
    args = parse_args()
    map_runs = list(iter_map_runs(args.map_start, args.map_end, args.runs_per_map))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    gp_context = initialize_gp()
    _, _, _, _, xs, ys, _, _, xmin, _, ymin, _, gp_step = gp_context
    map_contexts = {
        selected_map: load_map_context(selected_map, xs, ys, xmin, ymin, gp_step)
        for selected_map in sorted({selected_map for selected_map, _ in map_runs})
    }

    metriclist = []
    for selected_map, run_index in map_runs:
        metriclist.append(
            run_map(
                selected_map,
                map_contexts[selected_map],
                gp_context,
                run_index=run_index,
                batch_output_root=args.output_dir,
            )
        )

    print("\nSummary of results:")
    for metrics in metriclist:
        print(
            f"Map {metrics['map_id']} run {int(metrics['run_index'])}: "
            f"Task Completion = {metrics['task_completion']:.4%}, "
            f"Global RMSE-AUC = {metrics['global_auc_rmse']:.4f}, "
            f"Occupied RMSE-AUC = {metrics['occupied_auc_rmse']:.4f}"
        )

    summary_path = args.output_dir / args.summary_name
    write_metric_summary(metriclist, summary_path)

    if metriclist:
        avg_completion = np.mean([metrics["task_completion"] for metrics in metriclist])
        avg_global_auc = np.mean([metrics["global_auc_rmse"] for metrics in metriclist])
        avg_occupied_auc = np.mean([metrics["occupied_auc_rmse"] for metrics in metriclist])
        print(f"\nAverage Task Completion = {avg_completion:.4%}")
        print(f"Average Global RMSE-AUC = {avg_global_auc:.4f}")
        print(f"Average Occupied RMSE-AUC = {avg_occupied_auc:.4f}")
