"""
dppo_rollout_worker.py

Lightweight, CUDA-free reimplementation of DPPOTRAINING_batched.py's per-episode chunk rollout
(_rollout_trajectory + Diffusionplanner_singlemap.apply_measurement_update_3d), used ONLY by
collect_rollouts_batched's multiprocessing worker pool.

WHY THIS FILE EXISTS SEPARATELY: apply_measurement_update_3d normally lives in
Diffusionplanner_singlemap.py, but importing that file pulls in the entire diffusion/DPPO stack
(sample_3d_sparse_trans_diffusion -> threeDSparseTransDiffusion), which at import time both loads
a large training dataset AND calls torch.cuda.is_available() (initializing a CUDA context) at
module level - confirmed by measurement: ~6.4 sec per import, plus an unwanted CUDA context per
process. A multiprocessing worker that only needs to run the Kalman-filter/dynamics part of the
simulation (pure NumPy/SciPy - the part collect_rollouts_batched's module docstring already
identifies as "can't be GPU-batched") has no use for any of that. This module imports only the
plain-NumPy pieces (gaussianprocesstraining, CMAES_classic_singlemap) and reimplements the same
measurement-update math, so worker processes start fast and never touch CUDA.

Measured with this module (16 worker processes, spawn context, BLAS pinned to 1 thread/worker):
the per-episode chunk-rollout workload that takes ~37s single-threaded-sequential drops to
~12s wall-clock - a real ~3x, not a naive 16x (this workload's dominant cost, updating a full
2601x2601 dense covariance matrix per measurement, is memory-bandwidth-bound, not purely
core-count-bound).

Kept deliberately in sync with DPPOTRAINING_batched.py's _rollout_trajectory and
Diffusionplanner_singlemap.py's apply_measurement_update_3d - if either changes, mirror the change
here too.
"""

import sys
from pathlib import Path

# Self-sufficient sys.path setup, independent of how the parent process's sys.path was built -
# a spawned multiprocessing worker is a fresh Python interpreter and isn't guaranteed to inherit
# path entries the parent added via `-m` invocation semantics rather than explicit sys.path
# edits. gaussianprocesstraining/CMAES_classic_singlemap live one level up from this file.
_SCRIPTS_DIR = str(Path(__file__).resolve().parent.parent)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

import numpy as np
import threadpoolctl

from gaussianprocesstraining import (
    build_spline_trajectory_3d,
    build_sensor_matrix,
    noise_model,
    sample_correlated_sensor_noise,
    build_correlated_noise_covariance,
    kalman_update,
)
from CMAES_classic_singlemap import dynamics_3d, waypoint_3d, compute_fov

ANGLE_OF_VIEW = 60.0


def pool_worker_init():
    """multiprocessing.Pool(initializer=pool_worker_init) - call once per worker process.

    A pool of N worker processes, each running a multi-threaded BLAS (this codebase's NumPy is
    OpenBLAS-backed, MAX_THREADS=24) internally, oversubscribes the machine's real core count
    N-fold and was measured to meaningfully hurt throughput. threadpoolctl (not env vars) is used
    because it constrains the ALREADY-LOADED BLAS library's runtime thread pool - OPENBLAS_NUM_THREADS
    only takes effect if set before the library is first loaded, which has usually already happened
    by the time an initializer callback runs (importing this module to resolve the callback
    reference already imports numpy).
    """
    threadpoolctl.threadpool_limits(limits=1)


def apply_measurement_update_3d(cx, cy, cz, mu, P, true_map_flat, xs, ys, rng):
    """Reimplementation of Diffusionplanner_singlemap.apply_measurement_update_3d - kept
    byte-for-byte equivalent deliberately; mirror any change made there here too."""
    fov = compute_fov(cz=cz, xs=xs, ys=ys, angle_of_view=ANGLE_OF_VIEW, cx=cx, cy=cy)
    sensor, block_ids = build_sensor_matrix(fov, cz, xs, ys, return_block_ids=True)
    sensor_variance = noise_model(cz)
    z_meas = sensor @ true_map_flat
    z_meas += sample_correlated_sensor_noise(block_ids, sensor_variance, rng)
    noise_covariance = build_correlated_noise_covariance(block_ids, sensor_variance)
    return kalman_update(mu, P, sensor, z_meas, noise_covariance, block_ids=block_ids)


def rollout_chunk(traj_world, mu, P, cx, cy, cz, grid_step, xmin, xmax, ymin, ymax,
                   true_map_flat, xs, ys, z_min, z_max, sample_step, rng,
                   chunk_size, record_positions=False, max_measurement_safety=1000):
    """One environment step's chunk (T_a in the DPPO paper). Kept congruent with the deployment
    loop in Diffusionplanner_singlemap.py - if that loop changes, mirror the change here.

    Three things match deployment exactly (all were divergent before and caused a train/deploy gap
    that val-eval could not detect, since val-eval reuses this same training rollout):

      1. Replan cadence is keyed to SPLINE INDEX, not measurement count. Deployment replans when
         `spline_idx >= execution_chunk`, taking one measurement per step until then - NOT a fixed
         number of measurements. The drone takes ~1.9 measurements per spline point, so the old
         `for _ in range(chunk_size)` (a measurement-count cap) executed only ~half of each planned
         trajectory before replanning, while deployment executes `chunk_size` spline points (~all
         of a 41-point plan at chunk_size>=40). `chunk_size` is now a SPLINE-INDEX threshold,
         matching deployment's execution_chunk. max_measurement_safety only guards against a goal
         that is never 'reached' looping forever.
      2. The dense path is clamped to the padded map region (padding = grid_step * 2), matching
         sample_diffusion_trajectory's clamp of dense_traj to [xmin+2*step, xmax-2*step] / z-bounds.
      3. The returned heading is the LAST single (~sample_step) step before replan, matching
         deployment's `current_heading_velocity = pos - previous_pose` at replan time - NOT the net
         displacement over the whole chunk (which was ~18x larger in normalized magnitude and threw
         the model's heading conditioning far out of distribution at deployment).

    Returns (mu, P, cx, cy, cz, position_history, last_step_heading). position_history is a list of
    (cx,cy,cz) tuples if record_positions=True else None. last_step_heading is the world-frame
    (dx,dy,dz) of the final executed step (zeros if no step was taken).
    """
    dense_path = build_spline_trajectory_3d(cx, cy, cz, traj_world, samples_per_segment=5)
    # (2) clamp dense path to the padded map region, matching sample_diffusion_trajectory.
    padding = grid_step * 2
    dense_path = [
        (min(max(px, xmin + padding), xmax - padding),
         min(max(py, ymin + padding), ymax - padding),
         min(max(pz, z_min), z_max))
        for (px, py, pz) in dense_path
    ]
    spline_idx = 0
    position_history = [] if record_positions else None
    prev_pose = (cx, cy, cz)
    measurements = 0

    # (1) advance until spline_idx reaches chunk_size (or the spline is exhausted), matching
    # deployment's `spline_idx >= execution_chunk` replan trigger.
    while spline_idx < chunk_size and measurements < max_measurement_safety:
        if spline_idx >= len(dense_path):
            break

        goal_x, goal_y, goal_z = dense_path[spline_idx]
        grad_x, grad_y, grad_z, waypoint_reached = waypoint_3d(
            cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=grid_step,
        )
        if waypoint_reached:
            spline_idx += 1
            if spline_idx < len(dense_path):
                goal_x, goal_y, goal_z = dense_path[spline_idx]
                grad_x, grad_y, grad_z, _ = waypoint_3d(
                    cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=grid_step,
                )
            else:
                grad_x, grad_y, grad_z = 0.0, 0.0, 0.0

        prev_pose = (cx, cy, cz)
        cx, cy, cz = dynamics_3d(
            cx, cy, cz, grad_x, grad_y, grad_z, sample_step,
            xmin, xmax, ymin, ymax, z_min, z_max,
            buffer=grid_step / 2,
        )

        mu, P = apply_measurement_update_3d(cx, cy, cz, mu, P, true_map_flat, xs, ys, rng)
        measurements += 1

        if record_positions:
            position_history.append((cx, cy, cz))

    # (3) last single step before replan, matching deployment's per-step heading at replan time.
    last_step_heading = (cx - prev_pose[0], cy - prev_pose[1], cz - prev_pose[2])

    return mu, P, cx, cy, cz, position_history, last_step_heading


def warmup_chunk(mu, P, cx, cy, cz, goal_x, goal_y, goal_z, grid_step, xmin, xmax, ymin, ymax,
                  true_map_flat, xs, ys, z_min, z_max, sample_step, rng, num_steps,
                  record_positions=False):
    """Replicates Diffusionplanner_singlemap.py's ts<=1 warmup: before the first diffusion replan,
    fly toward a fixed goal (world (80,80,INIT_ALTITUDE) there) for `num_steps` steps, taking one
    measurement per step. This makes the first replan's belief (prior + num_steps measurements),
    position, and heading (last warmup step) match deployment instead of starting the first replan
    from the raw prior at the corner with an arbitrary heading.

    Note the buffer is grid_step * 2 here (not grid_step / 2 as in rollout_chunk) - deployment's
    warmup branch uses buffer=step*2 while its main flight loop uses buffer=step/2.

    Unlike pool_task, this runs in the MAIN process (it's cheap - num_steps * num_episodes
    measurements), so `rng` is mutated in place and does NOT need to be returned/reassigned.

    Returns (mu, P, cx, cy, cz, position_history, last_step_heading).
    """
    position_history = [] if record_positions else None
    prev_pose = (cx, cy, cz)
    for _ in range(num_steps):
        grad_x, grad_y, grad_z, _ = waypoint_3d(
            cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=grid_step,
        )
        prev_pose = (cx, cy, cz)
        cx, cy, cz = dynamics_3d(
            cx, cy, cz, grad_x, grad_y, grad_z, sample_step,
            xmin, xmax, ymin, ymax, z_min, z_max,
            buffer=grid_step * 2,
        )
        mu, P = apply_measurement_update_3d(cx, cy, cz, mu, P, true_map_flat, xs, ys, rng)
        if record_positions:
            position_history.append((cx, cy, cz))

    last_step_heading = (cx - prev_pose[0], cy - prev_pose[1], cz - prev_pose[2])
    return mu, P, cx, cy, cz, position_history, last_step_heading


def pool_task(args):
    """Entry point dispatched via Pool.map - one argument tuple in, one result tuple out (both
    plain picklable data: no torch tensors, no SingleEnvironment). `rng` is a numpy Generator,
    which pickles its full internal state - the caller must keep using the RETURNED rng (its
    state has advanced) for that same episode's next replan, not the one it sent in."""
    (traj_world, mu, P, cx, cy, cz, grid_step, xmin, xmax, ymin, ymax,
     true_map_flat, xs, ys, z_min, z_max, sample_step, rng,
     chunk_size, record_positions) = args

    mu, P, cx, cy, cz, position_history, last_step_heading = rollout_chunk(
        traj_world, mu, P, cx, cy, cz, grid_step, xmin, xmax, ymin, ymax,
        true_map_flat, xs, ys, z_min, z_max, sample_step, rng,
        chunk_size, record_positions,
    )
    return mu, P, cx, cy, cz, position_history, last_step_heading, rng
