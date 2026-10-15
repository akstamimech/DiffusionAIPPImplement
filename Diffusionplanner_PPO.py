import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
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
import torch
import time
from CMAES_classic_singlemap import compute_fov, dynamics_3d, waypoint_3d

SCRIPT_DIR = Path(__file__).resolve().parent
DIFFUSION_DIR = SCRIPT_DIR / "Diffusion"
if str(DIFFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_DIR))

from sample_3d_sparse_trans_diffusion import diffusion

step = 2.0
timealloted = 150
beta = 1.0
alpha = 0.02
utility_threshold = 0.5
planning_horizon = 8
selected_map = int(os.environ.get("SELECTED_MAP", 22))
samples_per_segment = 5
execution_chunk = 10
SENSORNOISE_SEED = 123
rng = np.random.default_rng(SENSORNOISE_SEED + selected_map)
MAPTYPE = os.environ.get("MAPTYPE", "grf")
CHUNK_SIZE = 256
CHUNK_PREFIX = "./trajectory_dataset_chunk"

diffusion_path = DIFFUSION_DIR / "checkpoints" / "sparse_trans_waypoints_epoch_1900_multimodal_full.pth"
INIT_ALTITUDE = 10.0
ZMIN = diffusion.Z_MIN
ZMAX = diffusion.Z_MAX
ANGLE_OF_VIEW = 60.0

"""
Single-map diffusion verification copy of receding_gridsearch_diffusion.py.
This script runs one chosen map and reports task completion after a single execution.
"""


def load_diffusion_model(checkpoint_path=None):
    model = diffusion.NoisePredictor().to(diffusion.device)
    if checkpoint_path is not None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=diffusion.device,
            weights_only=True,
        )
        state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        state_dict = diffusion.remap_legacy_state_dict_keys(state_dict)
        model.load_state_dict(state_dict)
    model.eval()
    diffusion.model = model
    return model


@torch.no_grad()
def sample_diffusion_trajectory(
    model,
    current_position,
    current_mean,
    current_var,
    current_heading_velocity=None,
    grid_step=step,
    bounds=None,
    num_steps=None,
    clip_x0=True
):
    model.eval()
    diffusion.model = model

    current_position_world = torch.as_tensor(
        current_position, dtype=torch.float32, device=diffusion.device
    ).view(1, 3)
    current_position_model = diffusion.normalize_xyz(current_position_world)

    current_mean = torch.tensor(current_mean, dtype=torch.float32, device=diffusion.device)
    current_var = torch.tensor(current_var, dtype=torch.float32, device=diffusion.device)

    mean_map = (current_mean - diffusion.mean_center.to(diffusion.device)) / diffusion.mean_scale.to(diffusion.device)
    var_map = (current_var - diffusion.var_center.to(diffusion.device)) / diffusion.var_scale.to(diffusion.device)
    marker_map = diffusion.make_position_marker_maps(
        current_position_world[:, :2].cpu(),
        grid_size=51,
        marker_radius=2,
    ).to(diffusion.device)[0, 0]
    meanvarmarker_map = torch.stack([mean_map, var_map, marker_map], dim=0).unsqueeze(0)

    if current_heading_velocity is None:
        initial_heading_velocity = torch.zeros(
            (1, 3),
            dtype=torch.float32,
            device=diffusion.device,
        )
    else:
        current_heading_velocity = torch.as_tensor(
            current_heading_velocity,
            dtype=torch.float32,
            device=diffusion.device,
        ).view(1, 3)
        initial_heading_velocity = diffusion.normalize_xyz_displacement(
            current_heading_velocity
        )

    sparse_noise = torch.randn((1, *diffusion.TARGET_SHAPE), device=diffusion.device)
    sparse_sample = diffusion.ddim_sample(
        sparse_noise,
        meanvarmarker_map,
        current_position=current_position_model,
        initial_heading_velocity=initial_heading_velocity,
        num_steps=num_steps,
        clip_x0=clip_x0,
    )

    control_waypoints = diffusion.extract_control_waypoints(sparse_sample[0])
    dense_traj = diffusion.pytorch_cubic_spline(
        control_waypoints,
        current_position=current_position_world[0],
    )[0]
    dense_traj = grid_step * torch.round(dense_traj / grid_step)
    if bounds is not None:
        xmin, xmax, ymin, ymax, zmin, zmax = bounds
        padding = grid_step * 2
        dense_traj[0] = dense_traj[0].clamp(xmin + padding, xmax - padding)
        dense_traj[1] = dense_traj[1].clamp(ymin + padding, ymax - padding)
        dense_traj[2] = dense_traj[2].clamp(zmin, zmax)
        control_waypoints[0] = control_waypoints[0].clamp(
            xmin + padding, xmax - padding
        )
        control_waypoints[1] = control_waypoints[1].clamp(
            ymin + padding, ymax - padding
        )
        control_waypoints[2] = control_waypoints[2].clamp(zmin, zmax)
    return dense_traj.detach().cpu().numpy(), control_waypoints.detach().cpu().numpy()


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


def flush_chunk(chunk_idx, traj_buffer, cond_buffer, mean_buffer, var_buffer):
    if not traj_buffer:
        return chunk_idx

    payload = {
        "trajectories": torch.tensor(np.asarray(traj_buffer, dtype=np.float32)).permute(0, 2, 1),
        "current_position": torch.tensor(np.asarray(cond_buffer, dtype=torch.float32)),
        "current_mean": torch.tensor(np.asarray(mean_buffer, dtype=torch.float32)),
        "current_var": torch.tensor(np.asarray(var_buffer, dtype=torch.float32)),
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

    """
    Map loading
    """

    csv_path = r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts\csv"
    # output_dir = Path(__file__).resolve().parent / "Vizualization" / f"diffusion_map_{selected_map}_viz"
    output_dir = Path(__file__).resolve().parent / "Vizualization" / f"diffusion_map_{MAPTYPE}_{selected_map}_viz"
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.loadtxt(
        rf"{csv_path}/map_{selected_map}_{MAPTYPE}_grid_counts.csv",
        delimiter=",",
        skiprows=1,
    )


    """
    Internal map representation initialization + data collection lists
    """
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
    mean = np.full(X_test.shape[0], utility_threshold + 0.1)  # UCB/grf: optimistic prior
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

    mu_history.append(mu.copy())
    P_history.append(P.copy())
    step_numbers.append(0)
    planned_path_history.append([])
    control_waypoint_history.append([])

    initial_utility = importance_filter(mu, P, beta, threshold=utility_threshold)
    utility_history.append(initial_utility.copy())

    save_every = 5
    lateral_coverage = step * 2
    samplestep = step

    cx, cy, cz = 4.0, 4.0, INIT_ALTITUDE
    grad_x, grad_y, grad_z = 0.0, 0.0, 0.0
    current_heading_velocity = np.zeros(3, dtype=np.float32)
    pos_history.append((cx, cy))
    pose_history.append((cx, cy, cz))
    altitude_history.append(cz)

    initial_var_field = np.diag(P).reshape(X.shape)
    gy0, gx0 = np.gradient(initial_var_field, Y[:, 0], X[0, :])
    grad_history.append((gx0, gy0))
    sorted_util_values_list.append(grid_measure(initial_utility, xs, ys))

    initial_total_variance = np.sum(np.diag(P))
    print(f"Initial total variance: {initial_total_variance:.4f}")

    utility = initial_utility
    diffusion_model = load_diffusion_model(checkpoint_path=diffusion_path)
    spline_path = []
    spline_idx = 0
    control_waypoints = []
    rmselist = []
    global_rmselist = []


    """
    Simulation loop
    """

    for ts in range(0, timealloted):
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
            previous_pose = np.array([cx, cy, cz], dtype=np.float32)
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
            current_heading_velocity = (
                np.array([cx, cy, cz], dtype=np.float32) - previous_pose
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
                """
                Diffusion sampling done here
                """
                dense_traj, sparse_controls = sample_diffusion_trajectory(
                    diffusion_model,
                    current_position=(cx, cy, cz),
                    current_mean=current_mean,
                    current_var=current_var,
                    current_heading_velocity=current_heading_velocity,
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

                previous_pose = np.array([cx, cy, cz], dtype=np.float32)
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
                current_heading_velocity = (
                    np.array([cx, cy, cz], dtype=np.float32) - previous_pose
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

    rmse_trace = np.column_stack([global_rmselist, rmselist])
    np.savetxt(
        output_dir / f"map_{selected_map}_rmse_over_time.csv",
        rmse_trace,
        delimiter=",",
        header="global_rmse,occupied_rmse",
        comments="",
    )

    plt.figure()
    plt.plot(global_rmselist, label="Global RMSE")
    plt.plot(rmselist, label="Occupied RMSE")
    plt.xlabel("Timestep")
    plt.ylabel("RMSE")
    plt.title(f"Map {selected_map} - Diffusion RMSE over Time")
    plt.legend()
    plt.savefig(output_dir / f"map_{selected_map}_rmse_over_time.png")
    plt.close()

    final_variance = np.sum(np.diag(P))
    print(f"Final total variance: {final_variance:.4f}")
    variance_delta = initial_total_variance - final_variance
    print(f"Variance reduction: {variance_delta:.4f}")

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
