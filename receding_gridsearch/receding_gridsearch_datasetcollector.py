import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
from gaussianprocesstraining import utility_function, sampler, create_plots_and_gifs, kalman_update, initialize_gp, grid_search
import torch


step = 2.0
timealloted = 1000
beta = 0.1
alpha = 0.1
utility_threshold = 0.0
planning_horizon = 8
action_horizon = 3 #this is more like replanning horizon
mapcount = 30
initial_map = 3
CHUNK_SIZE = 256
CHUNK_PREFIX = "./trajectory_dataset_chunk"


"""
CAREFUL: gaussianprocesstraining.py has a very simple gradient based planner that I commented out for now.
"""


# alpha is weight for how costly distance is
def receding_horizon_planner(cx, cy, sorted_util_values, alpha, horizon=planning_horizon):
    # sorted_util_values is sorted by estimated utility over map
    flight_plan = []
    for _ in range(horizon + 1):
        updated_utils = []
        for util, (x, y) in sorted_util_values:
            dist = np.hypot(x - cx, y - cy)

            if dist == 0:
                dist = 1e-6

            new_util = util * np.exp(-alpha * dist) # exp changes the order
            updated_utils.append((new_util, (x, y)))

        sorted_util_values = sorted(updated_utils, key=lambda x: x[0], reverse=True)

        best_coord = sorted_util_values[0][1]
        best_coord = (
            step * np.round(best_coord[0] / step),
            step * np.round(best_coord[1] / step),
        )

        flight_plan.append(best_coord)

        # update current position and avoid selecting the same point repeatedly
        cx, cy = best_coord
        sorted_util_values.pop(0)

    return flight_plan


"""
Assume a very simple waypoint based receding horizon. We aren't even considering dynamics yet.

This is wrong! This is just a greedy planner. Receding horizon needs to consider total gain over n steps.
"""
# def true_receding_horizon(cx, cy, sorted_util_values, alpha, horizon = planning_horizon):
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
    noise = np.random.randint(-1, 2, size=2)
    cx = np.clip(cx + grad_x * samplestep + noise[0], xmin + buffer, xmax - buffer)
    cy = np.clip(cy + grad_y * samplestep + noise[1], ymin + buffer, ymax - buffer)
    cx = step * np.round(cx / step)
    cy = step * np.round(cy / step)
    return cx, cy


def flush_chunk(chunk_idx, traj_buffer, cond_buffer, mean_buffer, var_buffer):
    if not traj_buffer:
        return chunk_idx

    payload = {
        "trajectories": torch.tensor(np.asarray(traj_buffer, dtype=np.float32)).permute(0, 2, 1),
        "current_position": torch.tensor(np.asarray(cond_buffer, dtype=np.float32)),
        "current_mean": torch.tensor(np.asarray(mean_buffer, dtype=np.float32)),
        "current_var": torch.tensor(np.asarray(var_buffer, dtype=np.float32)),
    }
    chunk_path = f"{CHUNK_PREFIX}_{chunk_idx:04d}.pt"
    torch.save(payload, chunk_path)
    print(f"Saved chunk {chunk_idx} to {chunk_path} with {len(traj_buffer)} samples")

    traj_buffer.clear()
    cond_buffer.clear()
    mean_buffer.clear()
    var_buffer.clear()

    return chunk_idx + 1


def finalize_chunks(num_chunks, final_path="./trajectory_dataset.pt"):
    trajectories = []
    current_positions = []
    current_means = []
    current_vars = []

    for chunk_idx in range(num_chunks):
        chunk_path = f"{CHUNK_PREFIX}_{chunk_idx:04d}.pt"
        chunk = torch.load(chunk_path, map_location="cpu")
        trajectories.append(chunk["trajectories"])
        current_positions.append(chunk["current_position"])
        current_means.append(chunk["current_mean"])
        current_vars.append(chunk["current_var"])

    final_payload = {
        "trajectories": torch.cat(trajectories, dim=0),
        "current_position": torch.cat(current_positions, dim=0),
        "current_mean": torch.cat(current_means, dim=0),
        "current_var": torch.cat(current_vars, dim=0),
    }
    torch.save(final_payload, final_path)

    for chunk_idx in range(num_chunks):
        os.remove(f"{CHUNK_PREFIX}_{chunk_idx:04d}.pt")


if __name__ == "__main__":
    path = r"/scratch/ajain3/aipp/csv"

    flight_plan_buffer = []
    condition_buffer = []
    mean_buffer = []
    var_buffer = []
    chunk_idx = 0

    for map in range(initial_map, initial_map + mapcount):
        data = np.loadtxt(rf"{path}/map_{map}_blob_grid_counts.csv", delimiter=",", skiprows=1)

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

        step_numbers = []
        grad_history = []
        pos_history = []

        step_numbers.append(0)

        initial_utility = utility_function(mu, P, utility_threshold, beta)

        save_every = 5
        lateral_coverage = step * 2
        samplestep = 4.0

        cx, cy = 20.0, 20.0
        grad_x, grad_y = 0.0, 0.0
        pos_history.append((cx, cy))

        initial_var_field = np.diag(P).reshape(X.shape)
        gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
        grad_history.append((gx0, gy0))

        initial_total_variance = np.sum(np.diag(P))
        print(f"Initial total variance: {initial_total_variance:.4f}")

        utility = initial_utility

        for ts in range(0, timealloted):
            if ts <= 10:
                grad_x, grad_y, waypoint_reached = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, step=step)
                cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
                pos_history.append((cx, cy))
                step_numbers.append(ts + 1)
            else:
                if ts == 11:
                    flight_plan = receding_horizon_planner(
                        cx, cy, grid_search(X, Y, cx, cy, [utility]), alpha=alpha, horizon=planning_horizon
                    )

                print("Current flight plan:", flight_plan)
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

                if len(flight_plan) <= action_horizon:
                    flight_plan = receding_horizon_planner(
                        cx, cy, grid_search(X, Y, cx, cy, [utility]), alpha=alpha, horizon=planning_horizon
                    )
                    print("Replanning...")

                if len(flight_plan) == planning_horizon + 1:
                    flight_plan_buffer.append(np.asarray(flight_plan, dtype=np.float32))
                    condition_buffer.append(np.asarray([cx, cy], dtype=np.float32))
                    mean_buffer.append(mu.reshape(len(ys), len(xs)).astype(np.float32))
                    var_buffer.append(np.diag(P).reshape(len(ys), len(xs)).astype(np.float32))

                    if len(flight_plan_buffer) >= CHUNK_SIZE:
                        chunk_idx = flush_chunk(
                            chunk_idx,
                            flight_plan_buffer,
                            condition_buffer,
                            mean_buffer,
                            var_buffer,
                        )

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
            util = utility.reshape(len(ys), len(xs))
            print(util, "shape:", util.shape)

        final_variance = np.sum(np.diag(P))
        print(f"Final total variance: {final_variance:.4f}")
        variance_delta = initial_total_variance - final_variance
        print(f"Variance reduction: {variance_delta:.4f}")

    chunk_idx = flush_chunk(
        chunk_idx,
        flight_plan_buffer,
        condition_buffer,
        mean_buffer,
        var_buffer,
    )
    finalize_chunks(chunk_idx, "./trajectory_dataset.pt")
    print(f"Saved final dataset with {chunk_idx} chunks to ./trajectory_dataset.pt")
