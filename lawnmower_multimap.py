import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
from pathlib import Path
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
from gaussianprocesstraining import utility_function, sampler, create_plots_and_gifs, kalman_update, initialize_gp, grid_search
from evalmetrics import compute_task_completion
import torch


step = 2.0
timealloted = 150
beta = 0.1
alpha = 1.0
utility_threshold = 0.0
planning_horizon = 8
action_horizon = 3  # this is more like replanning horizon
initial_map = 31
mapcount = 19
lateral_coverage = step * 2


"""
Multi-map lawnmower coverage script.
Set `initial_map` and `mapcount` above to choose the map range.
"""


def receding_horizon_planner(cx, cy, sorted_util_values, alpha, horizon=planning_horizon):
    flight_plan = []
    for _ in range(horizon + 1):
        updated_utils = []
        for util, (x, y) in sorted_util_values:
            dist = np.hypot(x - cx, y - cy)

            if dist == 0:
                dist = 1e-6

            new_util = util * np.exp(-alpha * dist)
            updated_utils.append((new_util, (x, y)))

        sorted_util_values = sorted(updated_utils, key=lambda x: x[0], reverse=True)

        best_coord = sorted_util_values[0][1]
        best_coord = (
            step * np.round(best_coord[0] / step),
            step * np.round(best_coord[1] / step),
        )

        flight_plan.append(best_coord)

        cx, cy = best_coord
        sorted_util_values.pop(0)

    return flight_plan


def lawnmower_planner(cx, cy, xmin, xmax, ymin, ymax, step=step, buffer=step * 2, lateral_coverage=lateral_coverage):
    x_left = xmin + buffer
    x_right = xmax - buffer
    y_top = ymax - buffer

    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)

    flight_plan = []
    y_values = np.arange(cy, y_top + 1e-9, lateral_coverage)

    for row_idx, y in enumerate(y_values):
        if row_idx % 2 == 0:
            x_values = np.arange(x_left, x_right + 1e-9, lateral_coverage)
        else:
            x_values = np.arange(x_right, x_left - 1e-9, -lateral_coverage)

        for x in x_values:
            waypoint = (float(step * np.round(x / step)), float(step * np.round(y / step)))

            if len(flight_plan) == 0 and np.isclose(waypoint[0], cx) and np.isclose(waypoint[1], cy):
                continue

            flight_plan.append(waypoint)

    return flight_plan


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
    csv_path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"
    metriclist = []
    total_gained_utility = 0.0
    total_available_utility = 0.0

    for selected_map in range(initial_map, initial_map + mapcount):
        # output_dir = Path(__file__).resolve().parent / f"lawnmower_map_{selected_map}_viz"
        # output_dir.mkdir(parents=True, exist_ok=True)

        data = np.loadtxt(rf"{csv_path}/map_{selected_map}_halffield_grid_counts.csv", delimiter=",", skiprows=1)

        gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()

        pts = data[:, 0:3]
        tol = 1e-9
        mask = (
            np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
            & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
        )
        pts = pts[mask]

        N = X_test.shape[0]
        mu = mean.copy()
        P = cov.copy()
        R = 1e-6

        mu_history = []
        P_history = []
        step_numbers = []
        grad_history = []
        pos_history = []
        utility_history = []
        sorted_util_values_list = []

        mu_history.append(mu.copy())
        P_history.append(P.copy())
        step_numbers.append(0)

        initial_utility = utility_function(mu, P, utility_threshold, beta)
        utility_history.append(initial_utility.copy())

        samplestep = 4.0

        cx, cy = 4.0, 4.0
        grad_x, grad_y = 0.0, 0.0
        pos_history.append((cx, cy))

        initial_var_field = np.diag(P).reshape(X.shape)
        gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
        grad_history.append((gx0, gy0))
        sorted_util_values_list.append(grid_search(X, Y, cx, cy, [initial_utility]))

        initial_total_variance = np.sum(np.diag(P))
        print(f"Initial total variance: {initial_total_variance:.4f}")

        utility = initial_utility

        for ts in range(0, timealloted):
            if ts <= 2:
                grad_x, grad_y, waypoint_reached = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, step=step)
                cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
                pos_history.append((cx, cy))
                step_numbers.append(ts + 1)
            else:
                if ts == 3:
                    flight_plan = lawnmower_planner(
                        cx, cy, xmin, xmax, ymin, ymax, step=step, lateral_coverage=lateral_coverage
                    )

                print("Current flight plan:", flight_plan)

                if len(flight_plan) == 0:
                    grad_x, grad_y = 0.0, 0.0
                else:
                    grad_x, grad_y, waypoint_reached = waypoint(
                        cx, cy, goal_x=flight_plan[0][0], goal_y=flight_plan[0][1], step=step
                    )
                    if waypoint_reached:
                        flight_plan.pop(0)
                        print("Waypoint reached.")
                        if len(flight_plan) > 0:
                            grad_x, grad_y, _ = waypoint(
                                cx, cy, goal_x=flight_plan[0][0], goal_y=flight_plan[0][1], step=step
                            )
                        else:
                            grad_x, grad_y = 0.0, 0.0

                cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
                pos_history.append((cx, cy))
                step_numbers.append(ts + 1)

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

                if len(idx) == 0:
                    continue

                idx = idx[0]
                sensor[i, idx] = 1.0

            measurement_list = []
            for x_fov, y_fov in fov:
                idx = np.where(
                    np.isclose(pts[:, 0], x_fov) & np.isclose(pts[:, 1], y_fov)
                )[0]

                if idx.size > 0:
                    measurement_list.append(pts[idx[0], 2])
                else:
                    measurement_list.append(0.0)

            z_meas = np.array(measurement_list)

            mu, P = kalman_update(mu, P, sensor, z_meas, R)
            utility = utility_function(mu, P, utility_threshold, beta=beta)

            mu_history.append(mu.copy())
            P_history.append(P.copy())
            utility_history.append(utility.copy())
            sorted_util_values_list.append(grid_search(X, Y, cx, cy, [utility]))

            var_field = np.diag(P).reshape(X.shape)
            gy, gx = np.gradient(var_field, Y[:, 0], X[0, :])
            grad_history.append((gx, gy))

            util = utility.reshape(len(ys), len(xs))
            print(util, "shape:", util.shape)

        final_variance = np.sum(np.diag(P))
        print(f"Final total variance: {final_variance:.4f}")
        variance_delta = initial_total_variance - final_variance
        print(f"Variance reduction: {variance_delta:.4f}")

        # create_plots_and_gifs(
        #     str(output_dir),
        #     mu_history,
        #     P_history,
        #     step_numbers,
        #     grad_history,
        #     pos_history,
        #     utility_history,
        #     sorted_util_values_list,
        #     X,
        #     Y,
        #     xs,
        #     ys,
        #     cx,
        #     cy,
        #     lateral_coverage,
        #     xmin,
        #     xmax,
        #     ymin,
        #     ymax,
        #     plot_utility=True,
        #     plot_grad=False,
        # )

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
        metriclist.append(
            (selected_map, metrics["gained_true_utility"], metrics["total_true_utility"], metrics["task_completion"])
        )
        total_gained_utility += metrics["gained_true_utility"]
        total_available_utility += metrics["total_true_utility"]
        print(
            f"Map {selected_map}: Gained Utility = {metrics['gained_true_utility']:.4f}, "
            f"Total Utility = {metrics['total_true_utility']:.4f}, "
            f"Task Completion = {metrics['task_completion']:.4%}"
        )

    print("\nSummary of results:")
    for map_id, gained_utility, total_utility, completion in metriclist:
        print(
            f"Map {map_id}: Gained Utility = {gained_utility:.4f}, "
            f"Total Utility = {total_utility:.4f}, "
            f"Task Completion = {completion:.4%}"
        )

    overall_completion = (
        total_gained_utility / total_available_utility if total_available_utility > 0 else 0.0
    )
    print("\nOverall totals:")
    print(f"Total Gained Utility = {total_gained_utility:.4f}")
    print(f"Total Utility = {total_available_utility:.4f}")
    print(f"Overall Task Completion = {overall_completion:.4%}")
