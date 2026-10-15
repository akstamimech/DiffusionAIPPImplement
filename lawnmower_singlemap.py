import numpy as np
import matplotlib
matplotlib.use("Agg")
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
from pathlib import Path
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
from gaussianprocesstraining import utility_function, sampler, create_plots_and_gifs, kalman_update, initialize_gp, grid_search
from gaussianprocesstraining import build_sensor_matrix, fov_grid_points, fov_lateral_radius, noise_model
from evalmetrics import compute_task_completion, compute_reconstruction_rmse, compute_rmse_time_metrics
from CMAES_classic_singlemap import dynamics_3d, waypoint_3d
import torch
import time

SCRIPT_DIR = Path(__file__).resolve().parent

step = 2.0
timealloted = int(os.environ.get("TIMEALLOTED", "400"))
#beta, alpha, are just for vizualisation. utility is for GP
beta = 1.5
alpha = 1.0
utility_threshold = float(os.environ.get("UTILITY_THRESHOLD", "0.3"))
planning_horizon = int(os.environ.get("PLANNING_HORIZON", "8"))
action_horizon = 3  # this is more like replanning horizon
selected_map = int(os.environ.get("SELECTED_MAP", 55))
lateral_coverage = step * 2
SENSORNOISE_SEED = int(os.environ.get("SENSORNOISE_SEED", "123"))
MAPTYPE = os.environ.get("MAPTYPE", "NAIP") #choose between "multiblob" and "halffield" or "blob" or "(nothing)" or "NAIP"
INIT_ALTITUDE = 10
ZMIN = 10.0
ZMAX = 40.0
WALLCLOCK_SECONDS = float(os.environ.get("WALLCLOCK_SECONDS", "150"))  # <=0 = unconstrained (timestep
                                                                      # loop runs to completion); otherwise
                                                                      # the flight ends at timealloted
                                                                      # timesteps OR this many real
                                                                      # seconds, whichever comes first

# Each simulation step is 1 vector step, treated as 1 meter of real flight. At
# a flight speed of FLIGHT_SPEED_MPS, that step physically takes at least
# STEP_DISTANCE_METERS / FLIGHT_SPEED_MPS seconds. When ENFORCE_MIN_STEP_TIME
# is on, a step that computed (dynamics + sensor update) faster than that gets
# padded with a sleep up to the floor; a step that already took longer is left
# alone - this is a floor, not a fixed duration.
ENFORCE_MIN_STEP_TIME = os.environ.get("ENFORCE_MIN_STEP_TIME", "1") == "1"
STEP_DISTANCE_METERS = 1.0
FLIGHT_SPEED_MPS = 3
MIN_STEP_SECONDS = STEP_DISTANCE_METERS / FLIGHT_SPEED_MPS


"""
Single-map classic grid-search control script.
Set `selected_map` above to run one map and generate visualizations afterward.
"""


# alpha is weight for how costly distance is



def grid_spaced_values(start, stop, spacing, step=step):
    direction = 1.0 if stop >= start else -1.0
    values = list(np.arange(start, stop + direction * 1e-9, direction * spacing))
    if not values or not np.isclose(values[-1], stop):
        values.append(stop)
    return [float(step * np.round(value / step)) for value in values]


def lawnmower_planner(cx, cy, xmin, xmax, ymin, ymax, step=step, buffer=step * 2, fov_radius=lateral_coverage):
    x_left = xmin + buffer
    x_right = xmax - buffer
    y_bottom = ymin + buffer
    y_top = ymax - buffer
    sweep_spacing = max(step, step * np.round((2.0 * fov_radius) / step))

    # Snap current position to grid
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)

    flight_plan = []

    # Build all sweep rows from current y upward
    y_values = grid_spaced_values(cy, y_top, sweep_spacing, step=step)

    for row_idx, y in enumerate(y_values):
        if row_idx == 0:
            x_values = grid_spaced_values(cx, x_right, sweep_spacing, step=step)
        elif row_idx % 2 == 1:
            x_values = grid_spaced_values(x_right, x_left, sweep_spacing, step=step)
        else:
            x_values = grid_spaced_values(x_left, x_right, sweep_spacing, step=step)

        for x in x_values:
            waypoint = (float(step * np.round(x / step)), float(step * np.round(y / step)))

            # Avoid immediately adding current position as first target
            if len(flight_plan) == 0 and np.isclose(waypoint[0], cx) and np.isclose(waypoint[1], cy):
                continue

            flight_plan.append(waypoint)

    return flight_plan





"""
Assume a very simple waypoint based receding horizon. We aren't even considering dynamics yet.

This is wrong! This is just a greedy planner. Receding horizon needs to consider total gain over n steps.
"""
# def true_receding_horizon(cx, cy, sorted_util_values, alpha=alpha, horizon=planning_horizon):
#     ...


def waypoint(cx, cy, goal_x, goal_y, step):
    dx = goal_x - cx
    dy = goal_y - cy
    dist = np.hypot(dx, dy)

    if dist <= step:
        return 0.0, 0.0, True

    grad_x = dx / dist
    grad_y = dy / dist

    return grad_x, grad_y, False


def dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax, buffer=step * 2):
    # noise = np.random.randint(-1, 2, size=2)
    # cx = np.clip(cx + grad_x * samplestep + noise[0], xmin + buffer, xmax - buffer)
    # cy = np.clip(cy + grad_y * samplestep + noise[1], ymin + buffer, ymax - buffer)
    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    return cx, cy


if __name__ == "__main__":
    csv_path = Path(os.environ.get("CSV_DIR", str(SCRIPT_DIR / "csv")))
    output_name = f"lawnmower_map_{MAPTYPE}_{selected_map}_viz"
    run_output_tag = os.environ.get("RUN_OUTPUT_TAG")
    if run_output_tag:
        output_name = f"{output_name}_{run_output_tag}"
    results_root = Path(os.environ.get("RESULTS_ROOT", str(SCRIPT_DIR / "Vizualization")))
    output_dir = results_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(csv_path / f"map_{selected_map}_{MAPTYPE}_grid_counts.csv", delimiter=",", skiprows=1)

    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()
    lateral_coverage = step * 2

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    N = X_test.shape[0]
    true_map_flat = np.zeros(N, dtype=float)
    for x_true, y_true, value in pts:
        idx = np.where(
            np.isclose(X_test[:, 0], x_true)
            & np.isclose(X_test[:, 1], y_true)
        )[0]
        if idx.size > 0:
            true_map_flat[idx[0]] = value

    rng = np.random.default_rng(SENSORNOISE_SEED + selected_map)

    mean = np.full(X_test.shape[0], utility_threshold - 0.1) #LCB/NAIP: pessimistic prior, everywhere starts unimportant
    mu = mean.copy()
    P = cov.copy()
    R = noise_model(INIT_ALTITUDE)

    mu_history = []
    P_history = []
    step_numbers = []
    grad_history = []
    pos_history = []
    pose_history = []
    utility_history = []
    sorted_util_values_list = []

    collect_viz_history = os.environ.get("SKIP_VIZ", "0") != "1"
    if collect_viz_history:
        mu_history.append(mu.copy())
        P_history.append(P.copy())
    step_numbers.append(0)

    initial_utility = utility_function(mu, P, utility_threshold, beta)
    if collect_viz_history:
        utility_history.append(initial_utility.copy())

    save_every = 5
    samplestep = step

    cx, cy, cz = 4.0, 4.0, INIT_ALTITUDE
    sensor_footprint_radius = fov_lateral_radius(cz)
    grad_x, grad_y = 0.0, 0.0
    pos_history.append((cx, cy))
    pose_history.append((cx, cy, cz))

    initial_var_field = np.diag(P).reshape(X.shape)
    if collect_viz_history:
        gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
        grad_history.append((gx0, gy0))
        sorted_util_values_list.append(grid_search(X, Y, cx, cy, [initial_utility]))

    initial_total_variance = np.sum(np.diag(P))
    variance_history = [initial_total_variance]
    print(f"Map {selected_map}: Initial total variance: {initial_total_variance:.4f}")

    utility = initial_utility

    rmselist = []
    global_rmselist = []
    wall_time_history = []

    # WALLCLOCK_SECONDS<=0 means unconstrained: the loop below only ever stops
    # because ts reached timealloted. Otherwise this flight ends either when
    # the timestep loop naturally finishes or when WALLCLOCK_SECONDS of real
    # time have elapsed since the flight started, whichever comes first.
    unconstrained = WALLCLOCK_SECONDS <= 0
    flight_start_time = time.time()
    stopped_early = False

    for ts in range(0, timealloted):
        if not unconstrained and (time.time() - flight_start_time) >= WALLCLOCK_SECONDS:
            stopped_early = True
            print(
                f"Map {selected_map}: wall-clock budget reached at timestep {ts} "
                f"(of {timealloted}); ending the flight early."
            )
            break

        step_start_time = time.time()

        if ts <= 1:
            grad_x, grad_y, grad_z, waypoint_reached = waypoint_3d(
                cx, cy, cz, goal_x=80.0, goal_y=80.0, goal_z=INIT_ALTITUDE, step=step
            )
            cx, cy, cz = dynamics_3d(
                cx, cy, cz, grad_x, grad_y, grad_z, samplestep, xmin, xmax, ymin, ymax, ZMIN, ZMAX, buffer=step * 2
            )
            pos_history.append((cx, cy))
            pose_history.append((cx, cy, cz))
            step_numbers.append(ts + 1)
        else:
            if ts == 2:
                flight_plan = lawnmower_planner(
                    cx,
                    cy,
                    xmin,
                    xmax,
                    ymin,
                    ymax,
                    step=step,
                    fov_radius=sensor_footprint_radius,
                )

            if os.environ.get("PRINT_FLIGHT_PLAN", "0") == "1":
                print("Current flight plan:", flight_plan)
            if len(flight_plan) > 0:
                grad_x, grad_y, grad_z, waypoint_reached = waypoint_3d(
                    cx, cy, cz, goal_x=flight_plan[0][0], goal_y=flight_plan[0][1], goal_z=INIT_ALTITUDE, step=step
                )
                if waypoint_reached:
                    flight_plan.pop(0)
                    if os.environ.get("PRINT_WAYPOINT_EVENTS", "0") == "1":
                        print("Waypoint reached.")
                    if len(flight_plan) > 0:
                        grad_x, grad_y, grad_z, _ = waypoint_3d(
                            cx, cy, cz, goal_x=flight_plan[0][0], goal_y=flight_plan[0][1], goal_z=INIT_ALTITUDE, step=step
                        )
                    else:
                        grad_x, grad_y, grad_z = 0.0, 0.0, 0.0
            else:
                grad_x, grad_y, grad_z = 0.0, 0.0, 0.0



            cx, cy, cz = dynamics_3d(
                cx, cy, cz, grad_x, grad_y, grad_z, samplestep, xmin, xmax, ymin, ymax, ZMIN, ZMAX, buffer=step * 2
            )
            pos_history.append((cx, cy))
            pose_history.append((cx, cy, cz))
            step_numbers.append(ts + 1)

        fov = fov_grid_points(cx, cy, cz, xs, ys)
        sensor = build_sensor_matrix(fov, cz, xs, ys)
        R = noise_model(cz)

        z_meas = sensor @ true_map_flat
        z_meas += rng.normal(0, np.sqrt(R), size=sensor.shape[0])

        mu, P = kalman_update(mu, P, sensor, z_meas, R)
        utility = utility_function(mu, P, utility_threshold, beta=beta)
        variance_history.append(float(np.sum(np.diag(P))))

        if collect_viz_history:
            mu_history.append(mu.copy())
            P_history.append(P.copy())
            utility_history.append(utility.copy())
            sorted_util_values_list.append(grid_search(X, Y, cx, cy, [utility]))

            var_field = np.diag(P).reshape(X.shape)
            gy, gx = np.gradient(var_field, Y[:, 0], X[0, :])
            grad_history.append((gx, gy))

        util = utility.reshape(len(ys), len(xs))
        # print(util, "shape:", util.shape)

        reconstruction_metrics = compute_reconstruction_rmse(
            mu=mu,
            pts=pts,
            xs=xs,
            ys=ys,
            step=step,
            utility_threshold=utility_threshold,
            xmin=xmin,
            ymin=ymin,
        )

        rmselist.append(reconstruction_metrics["occupied_rmse"])
        global_rmselist.append(reconstruction_metrics["global_rmse"])
        if ENFORCE_MIN_STEP_TIME:
            step_elapsed = time.time() - step_start_time
            if step_elapsed < MIN_STEP_SECONDS:
                time.sleep(MIN_STEP_SECONDS - step_elapsed)

        wall_time_history.append(time.time() - flight_start_time)

    if not rmselist:
        raise SystemExit(
            f"Map {selected_map}: WALLCLOCK_SECONDS={WALLCLOCK_SECONDS} was too small "
            f"for even one timestep to complete; nothing to plot or save."
        )

    timestep_index = np.asarray(step_numbers[1:], dtype=int)
    wall_time_arr = np.asarray(wall_time_history, dtype=float)
    variancelist = variance_history
    variance_per_step = np.asarray(variancelist[1:], dtype=float)

    metrics_trace = np.column_stack(
        [timestep_index, wall_time_arr, global_rmselist, rmselist, variance_per_step]
    )
    np.savetxt(
        output_dir / f"map_{selected_map}_rmse_over_time.csv",
        metrics_trace,
        delimiter=",",
        header="timestep,wall_time_seconds,global_rmse,occupied_rmse,global_variance",
        comments="",
    )
    pose_trace = np.column_stack(
        [
            np.asarray(step_numbers, dtype=int),
            np.concatenate(([0.0], wall_time_arr)),
            np.asarray(pose_history, dtype=float),
        ]
    )
    np.savetxt(
        output_dir / f"map_{selected_map}_executed_trajectory.csv",
        pose_trace,
        delimiter=",",
        header="timestep,wall_time_seconds,x,y,z",
        comments="",
    )

    plt.figure()
    plt.plot(timestep_index, global_rmselist, label="Global RMSE")
    plt.plot(timestep_index, rmselist, label="Occupied RMSE")
    plt.xlabel("Timestep")
    plt.ylabel("RMSE")
    plt.title(f"Map {selected_map} - RMSE over Timesteps")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_rmse_over_time.png")
    plt.close()

    plt.figure()
    plt.plot(wall_time_arr, global_rmselist, label="Global RMSE")
    plt.plot(wall_time_arr, rmselist, label="Occupied RMSE")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("RMSE")
    plt.title(f"Map {selected_map} - RMSE over Wall Time")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_rmse_over_walltime.png")
    plt.close()

    plt.figure()
    plt.plot(variancelist, label="Global Variance")
    plt.xlabel("Timestep")
    plt.ylabel("Global Variance")
    plt.title(f"Map {selected_map} - Lawnmower Variance over Timesteps")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_var.png")
    plt.close()

    plt.figure()
    plt.plot(wall_time_arr, variance_per_step, label="Global Variance")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("Global Variance")
    plt.title(f"Map {selected_map} - Lawnmower Variance over Wall Time")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_var_walltime.png")
    plt.close()

    final_variance = np.sum(np.diag(P))
    variance_delta = initial_total_variance - final_variance
    print(f"Map {selected_map}: Final total variance: {final_variance:.4f}")
    print(f"Map {selected_map}: Variance reduction: {variance_delta:.4f}")
    if stopped_early:
        print(
            f"Map {selected_map}: flight ended early after "
            f"{time.time() - flight_start_time:.1f}s due to the wall-clock budget "
            f"(completed {len(rmselist)} of {timealloted} timesteps)."
        )

    if os.environ.get("SKIP_VIZ", "0") != "1":
        create_plots_and_gifs(
            str(output_dir),
            mu_history,
            P_history,
            step_numbers,
            grad_history,
            pos_history,
            utility_history,
            sorted_util_values_list,
            X,
            Y,
            xs,
            ys,
            cx,
            cy,
            sensor_footprint_radius,
            xmin,
            xmax,
            ymin,
            ymax,
            plot_utility=True,
            plot_grad=False,
        )

    metrics = compute_task_completion(
        pos_history=pos_history,
        pts=pts,
        xs=xs,
        ys=ys,
        step=step,
        lateral_coverage=sensor_footprint_radius,
        xmin=xmin,
        ymin=ymin,
    )
    print(
        f"Map {selected_map}: Gained Utility = {metrics['gained_true_utility']:.4f}, "
        f"Total Utility = {metrics['total_true_utility']:.4f}, "
        f"Task Completion = {metrics['task_completion']:.4%}"
    )

    print(
        f"Map {selected_map}: Global RMSE = {reconstruction_metrics['global_rmse']:.4f}, "
        f"Occupied RMSE = {reconstruction_metrics['occupied_rmse']:.4f}"
    )

    global_rmse_time = compute_rmse_time_metrics(global_rmselist)
    occupied_rmse_time = compute_rmse_time_metrics(rmselist)
    print(
        f"Map {selected_map}: Global RMSE AUC = {global_rmse_time['auc_rmse']:.4f}, "
        f"Mean = {global_rmse_time['mean_rmse']:.4f}"
    )
    print(
        f"Map {selected_map}: Occupied RMSE AUC = {occupied_rmse_time['auc_rmse']:.4f}, "
        f"Mean = {occupied_rmse_time['mean_rmse']:.4f}"
    )
