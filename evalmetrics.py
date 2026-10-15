import numpy as np
LCB = True  # must stay in sync with gaussianprocesstraining.py's LCB - unsynced copy, no shared import

def compute_task_completion(
    pos_history,
    pts,
    xs,
    ys,
    step,
    lateral_coverage,
    xmin=None,
    ymin=None,
    pose_history=None,
    angle_of_view=60.0,
):
    """Compute unique true utility coverage over a rollout.

    Args:
        pos_history: Sequence of sampled positions [(x, y), ...].
        pts: Array with columns [x, y, true_count].
        xs, ys: Grid coordinate axes used by the planner.
        step: Grid spacing.
        lateral_coverage: Half-width of the square sensor footprint.
        xmin, ymin: Optional grid origin. Defaults to xs.min(), ys.min().
        pose_history: Optional sequence of sampled poses [(x, y, z), ...].
            When supplied, altitude determines the square FoV radius and
            lateral_coverage is ignored.
        angle_of_view: Sensor angle of view used with pose_history.

    Returns:
        dict with gained_true_utility, total_true_utility, task_completion,
        true_map, and observed_mask.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    pts = np.asarray(pts)

    if xmin is None:
        xmin = float(xs.min())
    if ymin is None:
        ymin = float(ys.min())

    true_map = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for x, y, count in pts:
        xi = int(round((x - xmin) / step))
        yi = int(round((y - ymin) / step))
        if 0 <= xi < len(xs) and 0 <= yi < len(ys):
            true_map[yi, xi] = count

    observed_mask = np.zeros_like(true_map, dtype=bool)

    if pose_history is None:
        coverage_poses = [
            (cx, cy, float(lateral_coverage))
            for cx, cy in pos_history
        ]
    else:
        half_angle = np.deg2rad(angle_of_view) / 2.0
        coverage_poses = [
            (cx, cy, float(cz) * np.tan(half_angle))
            for cx, cy, cz in pose_history
        ]

    for cx, cy, coverage_radius in coverage_poses:
        for x in np.arange(cx - coverage_radius, cx + coverage_radius + 1e-9, step):
            for y in np.arange(cy - coverage_radius, cy + coverage_radius + 1e-9, step):
                xi = int(round((x - xmin) / step))
                yi = int(round((y - ymin) / step))
                if 0 <= xi < len(xs) and 0 <= yi < len(ys):
                    observed_mask[yi, xi] = True

    gained_true_utility = float(true_map[observed_mask].sum())
    total_true_utility = float(true_map.sum())
    task_completion = gained_true_utility / total_true_utility if total_true_utility > 0 else 0.0

    return {
        "gained_true_utility": gained_true_utility,
        "total_true_utility": total_true_utility,
        "task_completion": task_completion,
        "true_map": true_map,
        "observed_mask": observed_mask,
    }


def compute_reconstruction_rmse(
    mu, pts, xs, ys, step, utility_threshold, xmin=None, ymin=None
):
    """Compare a posterior mean map against true grid counts.

    Args:
        mu: Posterior mean as either a flat vector or a grid-shaped array.
        pts: Array with columns [x, y, true_count].
        xs, ys: Grid coordinate axes used by the planner.
        step: Grid spacing.
        utility_threshold: Ground-truth cutoff used with the configured LCB/UCB
            direction for the targeted reconstruction metric.
        xmin, ymin: Optional grid origin. Defaults to xs.min(), ys.min().

    Returns:
        dict with global_rmse, occupied_rmse, true_map, mean_map, and
        occupied_mask.
    """
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    pts = np.asarray(pts)

    if xmin is None:
        xmin = float(xs.min())
    if ymin is None:
        ymin = float(ys.min())

    true_map = np.zeros((len(ys), len(xs)), dtype=np.float32)
    for x, y, count in pts:
        xi = int(round((x - xmin) / step))
        yi = int(round((y - ymin) / step))
        if 0 <= xi < len(xs) and 0 <= yi < len(ys):
            true_map[yi, xi] = count

    mean_map = np.asarray(mu, dtype=np.float32)
    if mean_map.ndim == 1:
        mean_map = mean_map.reshape(len(ys), len(xs))
    elif mean_map.shape != true_map.shape:
        raise ValueError(
            f"mu must be flat or have shape {true_map.shape}, got {mean_map.shape}"
        )

    error = mean_map - true_map
    global_rmse = float(np.sqrt(np.mean(error ** 2)))

    if LCB == True: 
        occupied_mask = true_map <= utility_threshold
    
    elif LCB == False:
        occupied_mask = true_map >= utility_threshold
    if np.any(occupied_mask):
        occupied_rmse = float(np.sqrt(np.mean(error[occupied_mask] ** 2)))
    else:
        occupied_rmse = 0.0

    return {
        "global_rmse": global_rmse,
        "occupied_rmse": occupied_rmse,
        "true_map": true_map,
        "mean_map": mean_map,
        "occupied_mask": occupied_mask,
    }


def compute_rmse_time_metrics(rmse_values, dt=1.0):
    """Summarize an RMSE-over-time curve.

    Lower mean_rmse and auc_rmse indicate faster reconstruction error reduction.
    """
    rmse_values = np.asarray(rmse_values, dtype=float)
    if rmse_values.size == 0:
        return {
            "final_rmse": 0.0,
            "mean_rmse": 0.0,
            "auc_rmse": 0.0,
        }

    return {
        "final_rmse": float(rmse_values[-1]),
        "mean_rmse": float(np.mean(rmse_values)),
        "auc_rmse": float(np.trapezoid(rmse_values, dx=dt)),
    }


def compute_variance_time_metrics(variance_values, dt=1.0):
    """Summarize a variance-over-time curve (e.g. occupied-region Tr(P)).

    Same convention as compute_rmse_time_metrics: auc_variance is the raw
    trapezoidal integral over elapsed time (variance * time units), NOT
    divided by the time span - it is not normalized, so runs covering more
    timesteps naturally accumulate a larger value. Lower still means
    variance stayed lower for more of the flight.
    """
    variance_values = np.asarray(variance_values, dtype=float)
    if variance_values.size == 0:
        return {
            "final_variance": 0.0,
            "mean_variance": 0.0,
            "auc_variance": 0.0,
        }

    return {
        "final_variance": float(variance_values[-1]),
        "mean_variance": float(np.mean(variance_values)),
        "auc_variance": float(np.trapezoid(variance_values, dx=dt)),
    }


def compute_coverage_efficiency(task_completion, ts):
    """Compute coverage efficiency as task completion divided by time steps."""
    if ts > 0:
        return task_completion / ts
    else:
        return 0.0
    
def compute_spatial_efficiency(pos_history, task_completion, ):
    """Compute spatial efficiency as task completion divided by unique positions."""
    unique_positions = set((round(x, 2), round(y, 2)) for x, y in pos_history)
    num_unique_positions = len(unique_positions)
    if num_unique_positions > 0:
        return task_completion / num_unique_positions
    else:
        return 0.0
