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
from gaussianprocesstraining import (
    noise_model,
    grid_search_3d,
    create_plots_and_gifs,
    kalman_update,
    initialize_gp,
    importance_filter,
    grid_measure,
    build_correlated_noise_covariance,
    build_sensor_matrix,
    sample_correlated_sensor_noise,
)
from evalmetrics import compute_task_completion, compute_reconstruction_rmse, compute_rmse_time_metrics
from CMAES_classic_singlemap import compute_fov, dynamics_3d, waypoint_3d
import time

SCRIPT_DIR = Path(__file__).resolve().parent

step = 2.0
timealloted = int(os.environ.get("TIMEALLOTED", "3000"))
beta = 1.0
utility_threshold = float(os.environ.get("UTILITY_THRESHOLD", "0.3"))
selected_map = int(os.environ.get("SELECTED_MAP", 53))
SENSORNOISE_SEED = int(os.environ.get("SENSORNOISE_SEED", "123"))
MAPTYPE = os.environ.get("MAPTYPE", "NAIP")  # "multiblob" or "NAIP"
WALLCLOCK_SECONDS = float(os.environ.get("WALLCLOCK_SECONDS", "300"))  # <=0 = unconstrained (timestep
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
FLIGHT_SPEED_MPS = 3.0
MIN_STEP_SECONDS = STEP_DISTANCE_METERS / FLIGHT_SPEED_MPS


##MAKING ALTITUDE A THING
INIT_ALTITUDE = 10.0  # Starting altitude only - grid_search_3d's lattice spans
                      # ZMIN..ZMAX, so altitude can change once planning starts.
ZMIN = 10.0
ZMAX = 40.0


"""
Single-map greedy-gradient control script.
Same dynamics, sensor/noise model, Kalman update, and map/settings as
CMAES_classic_singlemap.py. The planner reuses CMAES_classic_singlemap.py's
3D pyramid-lattice grid search (gaussianprocesstraining.grid_search_3d) but
with planning_horizon=1, so each replan only scores the single next-best
grid point (gain/distance, same scoring grid_search_3d always uses) rather
than an 8-waypoint flight plan - and unlike CMAES_classic_singlemap.py, that
one waypoint is never refined by CMA-ES (no cma_es_refine_waypoints_3d, no
spline trajectory - just a straight run at the chosen point).

Replanning is event-triggered, not per-timestep: the agent flies straight at
its current target grid point every timestep via waypoint_3d/dynamics_3d,
and only calls grid_search_3d again once that target is reached. Because
grid_search_3d's lattice spans multiple altitude tiers, this planner (unlike
the old sampler()-based version) does control height.
Set `selected_map` above to run one map and generate visualizations afterward.
"""


'''
the waypoint() function converts the current position and a goal position into a unit vector direction for the dynamics() function
'''

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
    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    return cx, cy


def dynamics_3d_with_progress(
    cx,
    cy,
    cz,
    grad_x,
    grad_y,
    grad_z,
    target_x,
    target_y,
    target_z,
    samplestep,
    xmin,
    xmax,
    ymin,
    ymax,
    zmin,
    zmax,
    buffer,
):
    next_x, next_y, next_z = dynamics_3d(
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
        zmin,
        zmax,
        buffer=buffer,
    )
    if (
        np.isclose(next_x, cx)
        and np.isclose(next_y, cy)
        and np.isclose(next_z, cz)
        and np.linalg.norm(np.array([target_x - cx, target_y - cy, target_z - cz])) > step
    ):
        delta = np.array([target_x - cx, target_y - cy, target_z - cz], dtype=float)
        axis = int(np.argmax(np.abs(delta)))
        fallback = np.array([cx, cy, cz], dtype=float)
        fallback[axis] += step * np.sign(delta[axis])
        fallback[0] = np.clip(fallback[0], xmin + buffer, xmax - buffer)
        fallback[1] = np.clip(fallback[1], ymin + buffer, ymax - buffer)
        fallback[2] = np.clip(fallback[2], zmin, zmax)
        fallback = step * np.round(fallback / step)
        next_x, next_y, next_z = fallback.tolist()
    return next_x, next_y, next_z


def replan_target(cx, cy, cz, mu, P, xs, ys):
    start_time = time.time()
    control_waypoints = grid_search_3d(
        mu,
        P,
        xs,
        ys,
        start_pose=(cx, cy, cz),
        beta=beta,
        utility_threshold=utility_threshold,
        planning_horizon=2,
        zmin=ZMIN,
        zmax=ZMAX,
    )
    end_time = time.time()
    current_pose = np.array([cx, cy, cz], dtype=float)
    reachable_now_tolerance = step
    target_x, target_y, target_z = control_waypoints[-1]
    for candidate in control_waypoints:
        candidate_pose = np.asarray(candidate, dtype=float)
        if np.linalg.norm(candidate_pose - current_pose) > reachable_now_tolerance:
            target_x, target_y, target_z = candidate
            break

    print("Replanning...")
    print(f"Replanning time: {end_time - start_time:.4f} seconds")
    print(f"Next grid point: ({target_x:.2f}, {target_y:.2f}, {target_z:.2f})")

    return target_x, target_y, target_z


if __name__ == "__main__":
    csv_path = Path(os.environ.get("CSV_DIR", str(SCRIPT_DIR / "csv")))
    output_name = f"greedygradient_map_{MAPTYPE}_{selected_map}_viz"
    run_output_tag = os.environ.get("RUN_OUTPUT_TAG")
    if run_output_tag:
        output_name = f"{output_name}_{run_output_tag}"
    results_root = Path(os.environ.get("RESULTS_ROOT", str(SCRIPT_DIR / "Vizualization")))
    output_dir = results_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(csv_path / f"map_{selected_map}_{MAPTYPE}_grid_counts.csv", delimiter=",", skiprows=1)
    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

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

    # mean = np.full(X_test.shape[0], utility_threshold + 0.1) #UCB variant (grf)
    mean = np.full(X_test.shape[0], utility_threshold - 0.1) #pessimistic prior for LCB/NAIP: everywhere starts important
    mu = mean.copy()
    P = cov.copy()

    mu_history = []
    P_history = []
    step_numbers = []
    grad_history = []
    pos_history = []
    utility_history = []
    sorted_util_values_list = []
    planned_path_history = []
    control_waypoint_history = []
    pose_history = []

    collect_viz_history = os.environ.get("SKIP_VIZ", "0") != "1"
    if collect_viz_history:
        mu_history.append(mu.copy())
        P_history.append(P.copy())
    step_numbers.append(0)

    initial_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    if collect_viz_history:
        utility_history.append(initial_utility.copy())
    planned_path_history.append([])
    control_waypoint_history.append([])

    save_every = 5
    lateral_coverage = step * 2
    samplestep = step

    cx, cy, cz = 4.0, 4.0, INIT_ALTITUDE
    grad_x, grad_y, grad_z = 0.0, 0.0, 0.0
    target_x, target_y, target_z = cx, cy, cz
    pos_history.append((cx, cy))
    pose_history.append((cx, cy, cz))

    initial_var_field = np.diag(P).reshape(X.shape)
    if collect_viz_history:
        gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
        grad_history.append((gx0, gy0))
        sorted_util_values_list.append(grid_measure(initial_utility, xs, ys))

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
            grad_x, grad_y, waypoint_reached = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, step=step)
            cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
            pos_history.append((cx, cy))
            pose_history.append((cx, cy, cz))
            step_numbers.append(ts + 1)
            planned_path_history.append([])
            control_waypoint_history.append([])
        else:
            if ts == 2:
                target_x, target_y, target_z = replan_target(cx, cy, cz, mu, P, xs, ys)

            grad_x, grad_y, grad_z, target_reached = waypoint_3d(
                cx, cy, cz, goal_x=target_x, goal_y=target_y, goal_z=target_z, step=step
            )

            if target_reached:
                target_x, target_y, target_z = replan_target(cx, cy, cz, mu, P, xs, ys)
                grad_x, grad_y, grad_z, _ = waypoint_3d(
                    cx, cy, cz, goal_x=target_x, goal_y=target_y, goal_z=target_z, step=step
                )

            prev_cx, prev_cy = cx, cy
            cx, cy, cz = dynamics_3d_with_progress(
                cx,
                cy,
                cz,
                grad_x,
                grad_y,
                grad_z,
                target_x,
                target_y,
                target_z,
                samplestep,
                xmin,
                xmax,
                ymin,
                ymax,
                ZMIN,
                ZMAX,
                buffer=step / 2,
            )
            pos_history.append((cx, cy))
            pose_history.append((cx, cy, cz))
            step_numbers.append(ts + 1)
            planned_path_history.append([(prev_cx, prev_cy), (target_x, target_y)])
            control_waypoint_history.append([(target_x, target_y)])

        fov = compute_fov(cz=cz, xs=xs, ys=ys, angle_of_view=60, step=step, cx=cx, cy=cy)

        sensor, sensor_block_ids = build_sensor_matrix(
            fov, cz, xs, ys, return_block_ids=True
        )

        R = noise_model(cz)  # Update measurement noise based on current altitude
        z_meas = sensor @ true_map_flat
        z_meas += sample_correlated_sensor_noise(sensor_block_ids, R, rng)
        R_cov = build_correlated_noise_covariance(sensor_block_ids, R)
        mu, P = kalman_update(
            mu, P, sensor, z_meas, R_cov, block_ids=sensor_block_ids
        )
        utility = importance_filter(mu, P, beta, threshold=utility_threshold)
        variance_history.append(float(np.sum(np.diag(P))))

        if collect_viz_history:
            mu_history.append(mu.copy())
            P_history.append(P.copy())
            utility_history.append(utility.copy())
            sorted_util_values_list.append(grid_measure(utility, xs, ys))

            var_field = np.diag(P).reshape(X.shape)
            gy, gx = np.gradient(var_field, Y[:, 0], X[0, :])
            grad_history.append((gx, gy))

        util = utility.reshape(len(ys), len(xs))

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

        rmselist.append(reconstruction_metrics['occupied_rmse'])
        global_rmselist.append(reconstruction_metrics['global_rmse'])
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
    plt.title(f"Map {selected_map} - Greedy Gradient RMSE over Timesteps")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_rmse_over_time.png")
    plt.close()

    plt.figure()
    plt.plot(wall_time_arr, global_rmselist, label="Global RMSE")
    plt.plot(wall_time_arr, rmselist, label="Occupied RMSE")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("RMSE")
    plt.title(f"Map {selected_map} - Greedy Gradient RMSE over Wall Time")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_rmse_over_walltime.png")
    plt.close()

    plt.figure()
    plt.plot(variancelist, label="Global Variance")
    plt.xlabel("Timestep")
    plt.ylabel("Global Variance")
    plt.title(f"Map {selected_map} - Greedy Gradient Variance over Timesteps")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_var.png")
    plt.close()

    plt.figure()
    plt.plot(wall_time_arr, variance_per_step, label="Global Variance")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("Global Variance")
    plt.title(f"Map {selected_map} - Greedy Gradient Variance over Wall Time")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_var_walltime.png")
    plt.close()

    final_variance = np.sum(np.diag(P))
    variance_delta = initial_total_variance - final_variance

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
            lateral_coverage,
            xmin,
            xmax,
            ymin,
            ymax,
            plot_utility=True,
            plot_grad=False,
            planned_path_history=planned_path_history,
            control_waypoint_history=control_waypoint_history,
            pose_history=pose_history,
            angle_of_view=60.0,
        )

    metrics = compute_task_completion(
        pos_history=pos_history,
        pts=pts,
        xs=xs,
        ys=ys,
        step=step,
        lateral_coverage=lateral_coverage,
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
    print(
        f"Map {selected_map}: Final total variance: {final_variance:.4f}, "
        f"Variance reduction: {variance_delta:.4f}"
    )
    if stopped_early:
        print(
            f"Map {selected_map}: flight ended early after "
            f"{time.time() - flight_start_time:.1f}s due to the wall-clock budget "
            f"(completed {len(rmselist)} of {timealloted} timesteps)."
        )
