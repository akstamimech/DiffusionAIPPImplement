"""
One simulated flight: ground-truth map, belief, noisy camera, point-mass drone, a planner, and the scoring.

Every planner runs through this same function, so the simulator and robot settings (grid, camera, noise, drone,
warm-up, pacing, metrics, output files) are identical across planners; only the planner differs:

    settings = FlightSettings.from_env(output_prefix="classic", plot_label="CMA-ES", timealloted=300, ...)
    result = run_flight(my_planner, settings)

A planner is anything with `plan(belief, pose) -> FlightPlan` and a `z_range` (see aipp.interfaces.PathPlanner).
Settings come from environment variables (SELECTED_MAP, TIMEALLOTED, ...), with defaults the planner's run script
passes in, so a sweep can launch any planner the same way.
"""
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from aipp import GPBelief, GridSpec, KernelSpec, Mission, MissionConfig, MissionResult, PathFollower, SensorSpec
from functions import create_plots_and_gifs, grid_measure, importance_filter
from .metrics import (
    compute_coverage_efficiency,
    compute_reconstruction_rmse,
    compute_rmse_time_metrics,
    compute_task_completion,
)

from .sensor import SimSensor
from .vehicle import PointMassVehicle
from .world import load_ground_truth

SCRIPTS_DIR = Path(__file__).resolve().parent.parent

GRID = GridSpec()
KERNEL = KernelSpec()   # NAIP-fitted Matern 3/2 prior
SENSOR = SensorSpec()
FLIGHT_SPEED_MPS = 3.0  # one simulation step is one grid step (2 m); at this speed it takes at least 2/3 s


def _env(name, cast, default):
    value = os.environ.get(name)
    return default if value is None or value == "" else cast(value)


@dataclass
class FlightSettings:
    """Everything about the simulated flight that is not the planner."""
    selected_map: int = 26
    maptype: str = "NAIP"
    timealloted: int = 4000
    utility_threshold: float = 0.3          # a cell is important when its lower confidence bound is below this
    beta: float = 1.0
    sensornoise_seed: int = 123
    start_x: float = 4.0
    start_y: float = 4.0
    init_altitude: float = 10.0
    warmup_goal_xy: tuple = (80.0, 80.0)
    execution_chunk: int = 40               # plan points flown before replanning
    wallclock_seconds: float = 150.0        # <= 0: no budget, run all steps
    enforce_min_step_time: bool = True      # pad every step to the time a real flight step would take
    skip_viz: bool = False                  # True: no videos (RMSE and variance plots are always written)
    csv_dir: Path = SCRIPTS_DIR / "csv"
    results_root: Path = SCRIPTS_DIR / "Vizualization"
    run_output_tag: Optional[str] = None
    output_prefix: str = "flight"           # output folder: <prefix>_map_<MAPTYPE>_<map>_viz[_<tag>]
    plot_label: str = "Planner"

    @classmethod
    def from_env(cls, output_prefix, plot_label, **defaults):
        """Defaults (from the planner's run script) are overridden by environment variables."""
        base = cls(output_prefix=output_prefix, plot_label=plot_label, **defaults)
        return replace(
            base,
            selected_map=_env("SELECTED_MAP", int, base.selected_map),
            maptype=_env("MAPTYPE", str, base.maptype),
            timealloted=_env("TIMEALLOTED", int, base.timealloted),
            utility_threshold=_env("UTILITY_THRESHOLD", float, base.utility_threshold),
            beta=_env("BETA", float, base.beta),
            sensornoise_seed=_env("SENSORNOISE_SEED", int, base.sensornoise_seed),
            start_x=_env("START_X", float, base.start_x),
            start_y=_env("START_Y", float, base.start_y),
            execution_chunk=_env("EXECUTION_CHUNK", int, base.execution_chunk),
            wallclock_seconds=_env("WALLCLOCK_SECONDS", float, base.wallclock_seconds),
            enforce_min_step_time=_env("ENFORCE_MIN_STEP_TIME", str, "1" if base.enforce_min_step_time else "0") == "1",
            skip_viz=_env("SKIP_VIZ", str, "1" if base.skip_viz else "0") == "1",
            csv_dir=Path(_env("CSV_DIR", str, str(base.csv_dir))),
            results_root=Path(_env("RESULTS_ROOT", str, str(base.results_root))),
            run_output_tag=_env("RUN_OUTPUT_TAG", str, base.run_output_tag),
        )

    @property
    def output_name(self):
        name = f"{self.output_prefix}_map_{self.maptype}_{self.selected_map}_viz"
        return f"{name}_{self.run_output_tag}" if self.run_output_tag else name

    @property
    def min_step_seconds(self):
        return GRID.step / FLIGHT_SPEED_MPS if self.enforce_min_step_time else 0.0


@dataclass
class FlightResult:
    mission: MissionResult
    output_dir: Path
    global_rmse: List[float] = field(default_factory=list)
    occupied_rmse: List[float] = field(default_factory=list)
    total_variance: List[float] = field(default_factory=list)    # entry 0 is the prior, then one per step
    wall_time: List[float] = field(default_factory=list)
    pose_history: List[tuple] = field(default_factory=list)      # entry 0 is the start pose
    step_numbers: List[int] = field(default_factory=list)
    task_metrics: dict = field(default_factory=dict)


def run_flight(planner, settings: FlightSettings, follower: Optional[PathFollower] = None) -> FlightResult:
    grid, step = GRID, GRID.step
    output_dir = settings.results_root / settings.output_name
    output_dir.mkdir(parents=True, exist_ok=True)

    xs, ys = grid.xs, grid.ys
    X, Y = grid.meshgrid()
    xmin, xmax, ymin, ymax = grid.xmin, grid.xmax, grid.ymin, grid.ymax
    selected_map, utility_threshold, beta = settings.selected_map, settings.utility_threshold, settings.beta

    pts, true_map_flat = load_ground_truth(settings.csv_dir, selected_map, settings.maptype, grid)
    rng = np.random.default_rng(settings.sensornoise_seed + selected_map)

    z_range = tuple(planner.z_range)
    # Under LCB/NAIP, the pessimistic prior starts below the importance threshold.
    belief = GPBelief(grid, KERNEL, prior_mean=utility_threshold - 0.1)
    sim_sensor = SimSensor(true_map_flat, grid, SENSOR, rng)
    vehicle = PointMassVehicle((settings.start_x, settings.start_y, settings.init_altitude), grid, z_range)
    follower = follower or PathFollower(grid, z_range, settings.execution_chunk)
    mission = Mission(
        belief, sim_sensor, planner, follower, vehicle,
        MissionConfig(
            warmup_goal=(*settings.warmup_goal_xy, settings.init_altitude),
            wallclock_seconds=settings.wallclock_seconds,
            min_step_seconds=settings.min_step_seconds,
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

    collect_viz_history = not settings.skip_viz
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

    # wallclock_seconds <= 0 means unconstrained: the loop only ever stops because timealloted steps were flown.
    # Otherwise the flight ends when those steps are done or the budget of real time has elapsed, whichever
    # comes first.
    result = mission.run(settings.timealloted, observers=[observe])
    stopped_early = result.stopped_early
    wall_time_history = [rec.wall_time for rec in result.records]
    mu, P = belief.mu, belief.P
    cx, cy, cz = vehicle.pose
    if stopped_early:
        print(
            f"Map {selected_map}: wall-clock budget reached at timestep {len(result.records)} "
            f"(of {settings.timealloted}); ending the flight early."
        )

    if not rmselist:
        raise SystemExit(
            f"Map {selected_map}: WALLCLOCK_SECONDS={settings.wallclock_seconds} was too small "
            f"for even one timestep to complete; nothing to plot or save."
        )

    label = f"{settings.plot_label} " if settings.plot_label else ""
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
    plt.title(f"Map {selected_map} - {label}RMSE over Timesteps")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_rmse_over_time.png")
    plt.close()

    plt.figure()
    plt.plot(wall_time_arr, global_rmselist, label="Global RMSE")
    plt.plot(wall_time_arr, rmselist, label="Occupied RMSE")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("RMSE")
    plt.title(f"Map {selected_map} - {label}RMSE over Wall Time")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_rmse_over_walltime.png")
    plt.close()

    plt.figure()
    plt.plot(variancelist, label="Global Variance")
    plt.xlabel("Timestep")
    plt.ylabel("Global Variance")
    plt.title(f"Map {selected_map} - {label}Variance over Timesteps")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_var.png")
    plt.close()

    plt.figure()
    plt.plot(wall_time_arr, variance_per_step, label="Global Variance")
    plt.xlabel("Wall-clock time (s)")
    plt.ylabel("Global Variance")
    plt.title(f"Map {selected_map} - {label}Variance over Wall Time")
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
            f"(completed {len(rmselist)} of {settings.timealloted} timesteps)."
        )

    if collect_viz_history:
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
            angle_of_view=SENSOR.angle_of_view_deg,
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

    coverage_efficiency = compute_coverage_efficiency(metrics["task_completion"], settings.timealloted + 1)
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

    return FlightResult(
        mission=result, output_dir=output_dir, global_rmse=global_rmselist, occupied_rmse=rmselist,
        total_variance=variancelist, wall_time=wall_time_history, pose_history=pose_history,
        step_numbers=step_numbers, task_metrics=metrics,
    )
