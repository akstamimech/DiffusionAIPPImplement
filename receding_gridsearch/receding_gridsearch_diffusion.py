import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
from gaussianprocesstraining import utility_function, sampler, create_plots_and_gifs, kalman_update, initialize_gp, grid_search
from evalmetrics import compute_task_completion, compute_coverage_efficiency, compute_spatial_efficiency
import torch


SCRIPT_DIR = Path(__file__).resolve().parent
DIFFUSION_DIR = SCRIPT_DIR / "Diffusion"
if str(DIFFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_DIR))

import MLPdiffusion as diffusion

###NOT UPDATED YET

step = 2.0
GP_STEP = 4.0 #NEED TO CHANGE THIS IN gaussianprocesstraining.py AS WELL
timealloted = 150
beta = 0.1
alpha = 0.1
utility_threshold = 0.0
planning_horizon = 8
action_horizon = 3 #this is more like replanning horizon
mapcount = 19
initial_map =31
CHUNK_SIZE = 256
CHUNK_PREFIX = "./trajectory_dataset_chunk"
selected_map = initial_map
DIFF_ITER = 5

MAP_DISC = int(100/GP_STEP) + 1

diffusion_path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\Diffusion\checkpoints\mlp_conditional_diffusion_traj_final.pth"

"""
Single-map evaluation copy of receding_gridsearch.py.
All default settings remain aligned with receding_gridsearch.py, but this file
runs one chosen map and reports task completion at the end.
"""


def load_diffusion_model(checkpoint_path=None):
    model = diffusion.NoisePredictor().to(diffusion.device)
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location=diffusion.device)
        model.load_state_dict(state_dict)
    model.eval()
    diffusion.model = model
    return model


@torch.no_grad()
def sample_diffusion_trajectory(model, current_position, current_mean, current_var):
    current_position = torch.as_tensor(current_position, dtype=torch.float32, device=diffusion.device).view(1, 2)
    current_position = current_position / diffusion.SCALE_FACTOR

    current_mean = torch.as_tensor(current_mean, dtype=torch.float32, device=diffusion.device)
    current_var = torch.as_tensor(current_var, dtype=torch.float32, device=diffusion.device)

    mean_map = (current_mean - diffusion.mean_center.to(diffusion.device)) / diffusion.mean_scale.to(diffusion.device)
    log_var_map = torch.log1p(current_var)
    log_var_map = (log_var_map - diffusion.log_var_center.to(diffusion.device)) / diffusion.log_var_scale.to(diffusion.device)
    meanvar_map = torch.stack([mean_map, log_var_map], dim=0).unsqueeze(0)

    traj = torch.randn((1, diffusion.NUM_COORDS, diffusion.TRAJ_SIZE), device=diffusion.device)
    for i in range(diffusion.T - 1, -1, -1):
        t = torch.full((1,), i, dtype=torch.long, device=diffusion.device)
        traj = diffusion.sample_timestep(traj, t, meanvar_map=meanvar_map, current_position=current_position)

    traj = (traj[0].detach().cpu().numpy() * diffusion.SCALE_FACTOR)
    return traj




def receding_horizon_planner(cx, cy, sorted_util_values, alpha, horizon=planning_horizon):
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

        cx, cy = best_coord
        sorted_util_values.pop(0)

    return flight_plan


"""
Assume a very simple waypoint based receding horizon. We aren't even considering dynamics yet.

This is wrong! This is just a greedy planner. Receding horizon needs to consider total gain over n steps.
"""
# def true_receding_horizon(cx, cy, sorted_util_values, alpha = alpha, horizon = planning_horizon):
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

    metriclist = []
    for selected_map in range(initial_map, initial_map + mapcount):
        path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"

        data = np.loadtxt(rf"{path}/map_{selected_map}_halffield_grid_counts.csv", delimiter=",", skiprows=1)

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
        sorted_util_values_list = []

        mu_history.append(mu.copy())
        P_history.append(P.copy())
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
        sorted_util_values_list.append(grid_search(X, Y, cx, cy, [initial_utility]))

        initial_total_variance = np.sum(np.diag(P))
        print(f"Initial total variance: {initial_total_variance:.4f}")

        utility = initial_utility



        #diffusion model loading
        diffusion_model = load_diffusion_model(checkpoint_path=diffusion_path)


        for ts in range(0, timealloted):
            if ts <= 10:
                grad_x, grad_y, waypoint_reached = waypoint(cx, cy, goal_x=80.0, goal_y=80.0, step=step)
                cx, cy = dynamics(cx, cy, grad_x, grad_y, step, samplestep, xmin, xmax, ymin, ymax)
                pos_history.append((cx, cy))
                step_numbers.append(ts + 1)
            else:
                if ts == 11:
                    current_mean = mu.reshape(MAP_DISC, MAP_DISC)
                    current_var = np.diag(P).reshape(MAP_DISC, MAP_DISC)
                    flight_plan = sample_diffusion_trajectory(diffusion_model, current_position=(cx, cy), current_mean=current_mean, current_var=current_var)
                    flight_plan = flight_plan.T.tolist()
                    # flight_plan = flight_plan[1:].tolist()
                

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
                
                if len(flight_plan) > 0:
                    x_next, y_next = flight_plan[0]
                    padding = step * 2
                    clamped_x = min(max(x_next, xmin + padding), xmax - padding)
                    clamped_y = min(max(y_next, ymin + padding), ymax - padding)
                    if clamped_x != x_next or clamped_y != y_next:
                        flight_plan[0] = [clamped_x, clamped_y]
                        print("Flight plan waypoint was out of bounds and was clamped back into the padded map region.")

                
                if len(flight_plan) <= action_horizon:
                    current_mean = mu.reshape(MAP_DISC, MAP_DISC)
                    current_var = np.diag(P).reshape(MAP_DISC, MAP_DISC)
                    flight_plan = sample_diffusion_trajectory(diffusion_model, current_position=(cx, cy), current_mean=current_mean, current_var=current_var)
                    flight_plan = flight_plan.T.tolist()
                    # flight_plan = flight_plan[1:].tolist()
                    print("Replanning...")

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
            sorted_util_values_list.append(grid_search(X, Y, cx, cy, [utility]))
            util = utility.reshape(len(ys), len(xs))
            print(util, "shape:", util.shape)

        final_variance = np.sum(np.diag(P))
        print(f"Final total variance: {final_variance:.4f}")
        variance_delta = initial_total_variance - final_variance
        print(f"Variance reduction: {variance_delta:.4f}")

        # create_plots_and_gifs(
        #     path,
        #     mu_history,
        #     P_history,
        #     step_numbers,
        #     grad_history,
        #     pos_history,
        #     [],
        #     [],
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
        #     plot_utility=False,
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
        # print(f"Map {selected_map} true blob utility captured after {timealloted} timesteps: {metrics['gained_true_utility']:.4f}")
        # print(f"Map {selected_map} total true blob utility: {metrics['total_true_utility']:.4f}")
        # print(f"Task completion after {timealloted} timesteps: {metrics['task_completion']:.4%}")

        metriclist.append((selected_map, metrics['gained_true_utility'], metrics['total_true_utility'], metrics['task_completion']))
    print("\nSummary of results:")
    for map_id, gained_utility, total_utility, completion in metriclist:
        print(f"Map {map_id}: Gained Utility = {gained_utility:.4f}, Total Utility = {total_utility:.4f}, Task Completion = {completion:.4%}")

    
