import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent   # scripts/: functions.py, evalmetrics.py, CMAES_classic_singlemap.py, csv/, Vizualization/
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
from tqdm import tqdm
from functions import (
    create_plots_and_gifs,
    grid_measure,
    importance_filter,
)
from evalmetrics import compute_task_completion, compute_coverage_efficiency, compute_reconstruction_rmse, compute_rmse_time_metrics
import json
import time

ONNX_DIR = Path(__file__).resolve().parent   # this folder: exported models + deploy JSON + onnx_sample_reference.py
if str(ONNX_DIR) not in sys.path:
    sys.path.insert(0, str(ONNX_DIR))

# The belief, the planner, the path follower and the mission loop are the aipp core (the planner needs
# onnxruntime, the rest is numpy only). Only the simulated camera and vehicle (aipp.sim) are simulator-specific.
from aipp import GPBelief, GridSpec, KernelSpec, Mission, MissionConfig, PathFollower, SensorSpec
from aipp.planner import DiffusionPlanner, PlannerConfig
from aipp.sim import PointMassVehicle, SimSensor

# The network runs through onnxruntime (no PyTorch, no training module, no dataset). Everything the
# training code used to derive from its dataset at import time (belief normalization constants, noise
# schedule, coordinate bounds) is read from the deploy JSON that was exported with the model.

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

# Each simulation step moves the vehicle one grid step, which is 2 m of real
# flight (the 51 x 51 grid represents a 100 m x 100 m field at 2 m spacing). At
# a flight speed of FLIGHT_SPEED_MPS, that step physically takes at least
# STEP_DISTANCE_METERS / FLIGHT_SPEED_MPS seconds. When ENFORCE_MIN_STEP_TIME
# is on, a step that computed (dynamics + any replan + sensor update) faster
# than that gets padded with a sleep up to the floor; a step that already took
# longer (e.g. one that triggered a diffusion replan) is left alone - this is
# a floor, not a fixed duration.
ENFORCE_MIN_STEP_TIME = os.environ.get("ENFORCE_MIN_STEP_TIME", "1") == "1"
STEP_DISTANCE_METERS = step   # one grid step per simulation step (2 m)
FLIGHT_SPEED_MPS = 3.0
MIN_STEP_SECONDS = STEP_DISTANCE_METERS / FLIGHT_SPEED_MPS

ONNX_MODEL = Path(os.environ.get("ONNX_MODEL", str(ONNX_DIR / "new_1200_int8.onnx")))
ONNX_DEPLOY_JSON = Path(os.environ.get("ONNX_DEPLOY_JSON", str(ONNX_DIR / "new_1200_deploy.json")))
DDIM_STEPS = int(os.environ["DDIM_STEPS"]) if os.environ.get("DDIM_STEPS") else None  # None = all T steps
ORT_THREADS = int(os.environ["ORT_THREADS"]) if os.environ.get("ORT_THREADS") else None  # None = onnxruntime default

INIT_ALTITUDE = 10.0
GRID = GridSpec(step=step)
KERNEL = KernelSpec()   # NAIP-fitted Matern 3/2 prior
SENSOR = SensorSpec()
ANGLE_OF_VIEW = SENSOR.angle_of_view_deg
START_X = float(os.environ.get("START_X", "4.0"))
START_Y = float(os.environ.get("START_Y", "4.0"))

"""
Single-map diffusion planner running the exported ONNX network (ONNX/DIFFUSIONPLANNER_ONNX.py), a copy of
Diffusionplanner_singlemap_headingremoved.py with the PyTorch model swapped for onnxruntime.
This script runs one chosen map and reports task completion after a single execution.
"""


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

    planner = DiffusionPlanner(
        ONNX_MODEL, ONNX_DEPLOY_JSON, GRID,
        PlannerConfig(eta=ETA, ddim_steps=DDIM_STEPS),
        threads=ORT_THREADS, seed=RUN_SEED,
    )
    ZMIN, ZMAX = planner.z_range
    print(f"ONNX planner: model {ONNX_MODEL.name}, deploy JSON {ONNX_DEPLOY_JSON.name}, T={planner.T}, eta={ETA}, steps={DDIM_STEPS or planner.T}", flush=True)

    data = np.loadtxt(
        csv_path / f"map_{selected_map}_{MAPTYPE}_grid_counts.csv",
        delimiter=",",
        skiprows=1,
    )

    xs, ys = GRID.xs, GRID.ys
    X, Y = GRID.meshgrid()
    X_test = GRID.cell_coordinates()
    xmin, xmax, ymin, ymax = GRID.xmin, GRID.xmax, GRID.ymin, GRID.ymax

    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    true_map_flat = build_true_map_flat(pts, X_test)

    rng = np.random.default_rng(SENSORNOISE_SEED + selected_map)

    # The mission: simulated camera and vehicle around the deployable core (belief, planner, follower, loop).
    # Under LCB/NAIP, the pessimistic prior starts below the importance threshold.
    belief = GPBelief(GRID, KERNEL, prior_mean=utility_threshold - 0.1)
    sim_sensor = SimSensor(true_map_flat, GRID, SENSOR, rng)
    vehicle = PointMassVehicle((START_X, START_Y, INIT_ALTITUDE), GRID, (ZMIN, ZMAX))
    follower = PathFollower(GRID, (ZMIN, ZMAX), execution_chunk)
    mission = Mission(
        belief, sim_sensor, planner, follower, vehicle,
        MissionConfig(
            warmup_goal=(80.0, 80.0, INIT_ALTITUDE),
            wallclock_seconds=WALLCLOCK_SECONDS,
            min_step_seconds=MIN_STEP_SECONDS if ENFORCE_MIN_STEP_TIME else 0.0,
        ),
    )

    mu, P = belief.mu, belief.P

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

    lateral_coverage = step * 2

    cx, cy, cz = vehicle.pose
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

    rmselist = []
    global_rmselist = []

    def observe(rec):
        """Everything the simulator wants to know about a step: path histories for the plots, and the ground-truth
        reconstruction error. Runs after the belief update; the mission loop knows nothing about it."""
        if rec.replanned:
            print("Replanning...")
            print("Control waypoints:", rec.plan.waypoints.tolist())
            print(f"Replanning time: {rec.plan.seconds:.4f} seconds")
        if rec.clamped:
            print("Flight plan waypoint was out of bounds and was clamped back into the padded map region.")

        pos_history.append(rec.pose[:2])
        pose_history.append(rec.pose)
        altitude_history.append(rec.pose[2])
        step_numbers.append(rec.step_number)
        planned_path_history.append(rec.planned_path_xy)
        control_waypoint_history.append(rec.control_waypoints_xy)

        mu, P = belief.mu, belief.P
        utility = importance_filter(mu, P, beta, threshold=utility_threshold)
        variance_history.append(rec.total_variance)
        if collect_viz_history:
            mu_history.append(mu.copy())
            P_history.append(P.copy())
            utility_history.append(utility.copy())
            sorted_util_values_list.append(grid_measure(utility, xs, ys))

            var_field = np.diag(P).reshape(X.shape)
            gy, gx = np.gradient(var_field, Y[:, 0], X[0, :])
            grad_history.append((gx, gy))

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

    # WALLCLOCK_SECONDS<=0 means unconstrained: the loop only ever stops because timealloted steps were flown.
    # Otherwise the flight ends when those steps are done or WALLCLOCK_SECONDS of real time have elapsed,
    # whichever comes first.
    result = mission.run(timealloted, observers=[observe])
    stopped_early = result.stopped_early
    wall_time_history = [rec.wall_time for rec in result.records]
    mu, P = belief.mu, belief.P
    cx, cy, cz = vehicle.pose
    if stopped_early:
        print(
            f"Map {selected_map}: wall-clock budget reached at timestep {len(result.records)} "
            f"(of {timealloted}); ending the flight early."
        )

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
            f"{result.flight_seconds:.1f}s due to the wall-clock budget "
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
        f"Map {selected_map}: Global RMSE = {global_rmselist[-1]:.4f}, "
        f"Occupied RMSE = {rmselist[-1]:.4f}"
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
