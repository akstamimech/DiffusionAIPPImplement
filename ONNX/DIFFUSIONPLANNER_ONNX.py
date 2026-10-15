import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent   # scripts/: gaussianprocesstraining.py, evalmetrics.py, CMAES_classic_singlemap.py, csv/, Vizualization/
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
from tqdm import tqdm
from gaussianprocesstraining import (
    build_correlated_noise_covariance,
    build_sensor_matrix,
    create_plots_and_gifs,
    grid_measure,
    importance_filter,
    initialize_gp,
    kalman_update,
    noise_model,
    sample_correlated_sensor_noise,
)
from evalmetrics import compute_task_completion, compute_coverage_efficiency, compute_reconstruction_rmse, compute_rmse_time_metrics
import json
import time
from CMAES_classic_singlemap import compute_fov, dynamics_3d, waypoint_3d

ONNX_DIR = Path(__file__).resolve().parent   # this folder: exported models + deploy JSON + onnx_sample_reference.py
if str(ONNX_DIR) not in sys.path:
    sys.path.insert(0, str(ONNX_DIR))

# The network runs through onnxruntime (no PyTorch, no training module, no dataset). Everything the
# training code used to derive from its dataset at import time (belief normalization constants, noise
# schedule, coordinate bounds) is read from the deploy JSON that was exported with the model.
from onnx_sample_reference import Planner


step = 2.0
timealloted = int(os.environ.get("TIMEALLOTED", "4000"))
beta = 1.0
alpha = 0.02
utility_threshold = float(os.environ.get("UTILITY_THRESHOLD", "0.3"))
planning_horizon = int(os.environ.get("PLANNING_HORIZON", "8"))
selected_map = int(os.environ.get("SELECTED_MAP", 26))
samples_per_segment = 5
execution_chunk = int(os.environ.get("EXECUTION_CHUNK", 40))
SENSORNOISE_SEED = int(os.environ.get("SENSORNOISE_SEED", "123"))
MAPTYPE = os.environ.get("MAPTYPE", "NAIP")
CHUNK_SIZE = 256
CHUNK_PREFIX = "./trajectory_dataset_chunk"
ETA = float(os.environ.get("ETA", "0.0"))
RUN_SEED_ENV = os.environ.get("RUN_SEED", "1121")
if RUN_SEED_ENV is not None:
    RUN_SEED = int(RUN_SEED_ENV)
    np.random.seed(RUN_SEED)
WALLCLOCK_SECONDS = float(os.environ.get("WALLCLOCK_SECONDS", "150"))  # <=0 = unconstrained (timestep
                                                                      # loop runs to completion); otherwise
                                                                      # the flight ends at timealloted
                                                                      # timesteps OR this many real
                                                                      # seconds, whichever comes first

# Each simulation step is 1 vector step, treated as 1 meter of real flight. At
# a flight speed of FLIGHT_SPEED_MPS, that step physically takes at least
# STEP_DISTANCE_METERS / FLIGHT_SPEED_MPS seconds. When ENFORCE_MIN_STEP_TIME
# is on, a step that computed (dynamics + any replan + sensor update) faster
# than that gets padded with a sleep up to the floor; a step that already took
# longer (e.g. one that triggered a diffusion replan) is left alone - this is
# a floor, not a fixed duration.
ENFORCE_MIN_STEP_TIME = os.environ.get("ENFORCE_MIN_STEP_TIME", "1") == "1"
STEP_DISTANCE_METERS = 1.0
FLIGHT_SPEED_MPS = 3.0
MIN_STEP_SECONDS = STEP_DISTANCE_METERS / FLIGHT_SPEED_MPS

ONNX_MODEL = Path(os.environ.get("ONNX_MODEL", str(ONNX_DIR / "new_1200_int8.onnx")))
ONNX_DEPLOY_JSON = Path(os.environ.get("ONNX_DEPLOY_JSON", str(ONNX_DIR / "new_1200_deploy.json")))
DDIM_STEPS = int(os.environ["DDIM_STEPS"]) if os.environ.get("DDIM_STEPS") else None  # None = all T steps
ORT_THREADS = int(os.environ["ORT_THREADS"]) if os.environ.get("ORT_THREADS") else None  # None = onnxruntime default

INIT_ALTITUDE = 10.0
_deploy_world = json.loads(ONNX_DEPLOY_JSON.read_text())["world"]
ZMIN = float(_deploy_world["z_min_m"])
ZMAX = float(_deploy_world["z_max_m"])
ANGLE_OF_VIEW = 60.0
START_X = float(os.environ.get("START_X", "4.0"))
START_Y = float(os.environ.get("START_Y", "4.0"))

"""
Single-map diffusion planner running the exported ONNX network (ONNX/DIFFUSIONPLANNER_ONNX.py), a copy of
Diffusionplanner_singlemap_headingremoved.py with the PyTorch model swapped for onnxruntime.
This script runs one chosen map and reports task completion after a single execution.
"""


def load_diffusion_model(onnx_path=None, deploy_json=None):
    return Planner(onnx_path or ONNX_MODEL, deploy_json or ONNX_DEPLOY_JSON, threads=ORT_THREADS, seed=RUN_SEED)


def sample_diffusion_trajectory(
    model,
    current_position,
    current_mean,
    current_var,
    grid_step=step,
    bounds=None,
    num_steps=None,
    clip_x0=True
):
    """Same contract as the PyTorch planner: returns (dense_traj [3, 41], control_waypoints [3, 8]) in meters."""
    position = np.asarray(current_position, dtype=np.float64).reshape(3)
    maps, position_normalized, total_variance = model.build_inputs(current_mean, current_var, position)
    sample = model.sample_normalized(
        maps,
        position_normalized,
        total_variance,
        eta=ETA,
        clip_x0=clip_x0,
        num_steps=num_steps or DDIM_STEPS,
    )
    control_waypoints = model.denormalize(sample)                      # [8, 3] meters, clipped to the field and altitude band
    dense_traj = model.cubic_spline(position, control_waypoints).T.copy()   # [3, 41]
    control_waypoints = control_waypoints.T.copy()                      # [3, 8]
    if bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        padding = grid_step * 2
        dense_traj[0] = np.clip(dense_traj[0], xmin + padding, xmax - padding)
        dense_traj[1] = np.clip(dense_traj[1], ymin + padding, ymax - padding)
        dense_traj[2] = np.clip(dense_traj[2], zmin, zmax)
        control_waypoints[0] = np.clip(control_waypoints[0], xmin + padding, xmax - padding)
        control_waypoints[1] = np.clip(control_waypoints[1], ymin + padding, ymax - padding)
        control_waypoints[2] = np.clip(control_waypoints[2], zmin, zmax)
    return dense_traj, control_waypoints


def build_true_map_flat(pts, X_test):
    true_map_flat = np.zeros(X_test.shape[0], dtype=float)
    for x_true, y_true, value in pts:
        idx = np.where(
            np.isclose(X_test[:, 0], x_true)
            & np.isclose(X_test[:, 1], y_true)
        )[0]
        if idx.size:
            true_map_flat[idx[0]] = value
    return true_map_flat


def apply_measurement_update_3d(cx, cy, cz, mu, P, true_map_flat, xs, ys, rng):
    fov = compute_fov(
        cz=cz,
        xs=xs,
        ys=ys,
        angle_of_view=ANGLE_OF_VIEW,
        step=step,
        cx=cx,
        cy=cy,
    )
    sensor, block_ids = build_sensor_matrix(
        fov,
        cz,
        xs,
        ys,
        return_block_ids=True,
    )
    sensor_variance = noise_model(cz)
    z_meas = sensor @ true_map_flat
    z_meas += sample_correlated_sensor_noise(
        block_ids,
        sensor_variance,
        rng,
    )
    noise_covariance = build_correlated_noise_covariance(
        block_ids,
        sensor_variance,
    )
    return kalman_update(
        mu,
        P,
        sensor,
        z_meas,
        noise_covariance,
        block_ids=block_ids,
    )


# def receding_horizon_planner(cx, cy, sorted_util_values, alpha, horizon=planning_horizon):
#     flight_plan = []
#     for _ in range(horizon + 1):
#         updated_utils = []
#         for util, (x, y) in sorted_util_values:
#             dist = np.hypot(x - cx, y - cy)

#             if dist == 0:
#                 dist = 1e-6

#             new_util = util * np.exp(-alpha * dist)
#             updated_utils.append((new_util, (x, y)))

#         sorted_util_values = sorted(updated_utils, key=lambda x: x[0], reverse=True)

#         best_coord = sorted_util_values[0][1]
#         best_coord = (
#             step * np.round(best_coord[0] / step),
#             step * np.round(best_coord[1] / step),
#         )

#         flight_plan.append(best_coord)

#         cx, cy = best_coord
#         sorted_util_values.pop(0)

#     return flight_plan



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
    cx = np.clip(cx + grad_x * samplestep, xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep, ymin + buffer, ymax - buffer)
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    return cx, cy


if __name__ == "__main__":
    csv_path = Path(os.environ.get("CSV_DIR", str(SCRIPT_DIR / "csv")))
    output_name = f"diffusion_onnx_map_{MAPTYPE}_{selected_map}_viz"
    run_output_tag = os.environ.get("RUN_OUTPUT_TAG")
    if run_output_tag:
        output_name = f"{output_name}_{run_output_tag}"
    results_root = Path(os.environ.get("RESULTS_ROOT", str(SCRIPT_DIR / "Vizualization")))
    output_dir = results_root / output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    diffusion_model = load_diffusion_model()
    print(f"ONNX planner: model {ONNX_MODEL.name}, deploy JSON {ONNX_DEPLOY_JSON.name}, T={diffusion_model.T}, eta={ETA}, steps={DDIM_STEPS or diffusion_model.T}", flush=True)

    data = np.loadtxt(
        csv_path / f"map_{selected_map}_{MAPTYPE}_grid_counts.csv",
        delimiter=",",
        skiprows=1,
    )

    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    N = X_test.shape[0]
    true_map_flat = build_true_map_flat(pts, X_test)

    rng = np.random.default_rng(SENSORNOISE_SEED + selected_map)

    # Under LCB/NAIP, the pessimistic prior starts below the importance threshold.
    mean = np.full(X_test.shape[0], utility_threshold - 0.1)
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
    altitude_history = []

    collect_viz_history = os.environ.get("SKIP_VIZ", "0") != "1"
    if collect_viz_history:
        mu_history.append(mu.copy())
        P_history.append(P.copy())
    step_numbers.append(0)
    planned_path_history.append([])
    control_waypoint_history.append([])

    initial_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    if collect_viz_history:
        utility_history.append(initial_utility.copy())

    save_every = 5
    lateral_coverage = step * 2
    samplestep = step

    cx, cy, cz = START_X, START_Y, INIT_ALTITUDE
    grad_x, grad_y, grad_z = 0.0, 0.0, 0.0
    pos_history.append((cx, cy))
    pose_history.append((cx, cy, cz))
    altitude_history.append(cz)

    initial_var_field = np.diag(P).reshape(X.shape)
    if collect_viz_history:
        gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
        grad_history.append((gx0, gy0))
        sorted_util_values_list.append(grid_measure(initial_utility, xs, ys))

    initial_total_variance = np.sum(np.diag(P))
    variance_history = [initial_total_variance]
    print(f"Map {selected_map}: Initial total variance: {initial_total_variance:.4f}")

    utility = initial_utility
    spline_path = []
    spline_idx = 0
    control_waypoints = []
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
            grad_x, grad_y, grad_z, _ = waypoint_3d(
                cx,
                cy,
                cz,
                goal_x=80.0,
                goal_y=80.0,
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
                buffer=step * 2,
            )
            pos_history.append((cx, cy))
            pose_history.append((cx, cy, cz))
            altitude_history.append(cz)
            step_numbers.append(ts + 1)
            planned_path_history.append([])
            control_waypoint_history.append([])
        else:
            if ts == 2 or spline_idx >= execution_chunk or spline_idx >= len(spline_path):
                current_mean = mu.reshape(X.shape)
                current_var = np.diag(P).reshape(X.shape)
                start_time = time.time()
                dense_traj, sparse_controls = sample_diffusion_trajectory(
                    diffusion_model,
                    current_position=(cx, cy, cz),
                    current_mean=current_mean,
                    current_var=current_var,
                    grid_step=step,
                    bounds=(xmin, xmax, ymin, ymax, ZMIN, ZMAX),
                )
                end_time = time.time()
                spline_path = dense_traj.T.tolist()
                spline_idx = 0
                control_waypoints = sparse_controls.T.tolist()
                print("Replanning...")
                print("Control waypoints:", control_waypoints)
                print(f"Replanning time: {end_time - start_time:.4f} seconds")

            if spline_idx < len(spline_path):
                goal_x, goal_y, goal_z = spline_path[spline_idx]
                grad_x, grad_y, grad_z, waypoint_reached = waypoint_3d(
                    cx,
                    cy,
                    cz,
                    goal_x=goal_x,
                    goal_y=goal_y,
                    goal_z=goal_z,
                    step=step,
                )

                if waypoint_reached:
                    spline_idx += 1
                    if spline_idx < len(spline_path):
                        goal_x, goal_y, goal_z = spline_path[spline_idx]
                        grad_x, grad_y, grad_z, _ = waypoint_3d(
                            cx,
                            cy,
                            cz,
                            goal_x=goal_x,
                            goal_y=goal_y,
                            goal_z=goal_z,
                            step=step,
                        )
                    else:
                        grad_x, grad_y, grad_z = 0.0, 0.0, 0.0

                if spline_idx < len(spline_path):
                    x_next, y_next, z_next = spline_path[spline_idx]
                    padding = step * 2
                    clamped_x = min(max(x_next, xmin + padding), xmax - padding)
                    clamped_y = min(max(y_next, ymin + padding), ymax - padding)
                    clamped_z = min(max(z_next, ZMIN), ZMAX)
                    if (
                        clamped_x != x_next
                        or clamped_y != y_next
                        or clamped_z != z_next
                    ):
                        spline_path[spline_idx] = [
                            clamped_x,
                            clamped_y,
                            clamped_z,
                        ]
                        print("Flight plan waypoint was out of bounds and was clamped back into the padded map region.")

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
                pos_history.append((cx, cy))
                pose_history.append((cx, cy, cz))
                altitude_history.append(cz)
                step_numbers.append(ts + 1)
                planned_path_history.append(
                    [(x, y) for x, y, _ in spline_path[spline_idx:]]
                )
                control_waypoint_history.append(
                    [(x, y) for x, y, _ in control_waypoints]
                )
            else:
                pos_history.append((cx, cy))
                pose_history.append((cx, cy, cz))
                altitude_history.append(cz)
                step_numbers.append(ts + 1)
                planned_path_history.append([])
                control_waypoint_history.append([])

        mu, P = apply_measurement_update_3d(
            cx,
            cy,
            cz,
            mu,
            P,
            true_map_flat,
            xs,
            ys,
            rng,
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
    plt.title(f"Map {selected_map} - Diffusion RMSE over Timesteps")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_rmse_over_time.png")
    plt.close()

    plt.figure()
    plt.plot(wall_time_arr, global_rmselist, label="Global RMSE")
    plt.plot(wall_time_arr, rmselist, label="Occupied RMSE")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("RMSE")
    plt.title(f"Map {selected_map} - Diffusion RMSE over Wall Time")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_rmse_over_walltime.png")
    plt.close()

    plt.figure()
    plt.plot(variancelist, label="Global Variance")
    plt.xlabel("Timestep")
    plt.ylabel("Global Variance")
    plt.title(f"Map {selected_map} - Diffusion Variance over Timesteps")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_var.png")
    plt.close()

    plt.figure()
    plt.plot(wall_time_arr, variance_per_step, label="Global Variance")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("Global Variance")
    plt.title(f"Map {selected_map} - Diffusion Variance over Wall Time")
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
            angle_of_view=ANGLE_OF_VIEW,
        )

    metrics = compute_task_completion(
        pos_history=pos_history,
        pose_history=pose_history,
        pts=pts,
        xs=xs,
        ys=ys,
        step=step,
        lateral_coverage=lateral_coverage,
        xmin=xmin,
        ymin=ymin,
    )

    coverage_efficiency = compute_coverage_efficiency(metrics["task_completion"], timealloted+ 1)
    print(f"Map {selected_map}: Gained Utility = {metrics['gained_true_utility']:.4f}, Total Utility = {metrics['total_true_utility']:.4f}, Task Completion = {metrics['task_completion']:.4%}")
    print(f"Map {selected_map}: Coverage Efficiency = {coverage_efficiency:.4f}")
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
