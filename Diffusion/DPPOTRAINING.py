"""
DPPOTRAINING.py

Fine-tunes the pre-trained 3D Sparse-Trans diffusion policy (threeDSparseTransDiffusion.NoisePredictor)
with Diffusion Policy Policy Optimization (DPPO, Ren et al. 2024), restricted to a single, fixed
environment: one map (chosen via MAP_ID) with one fixed starting position/heading, and the
initial belief state built directly from initialize_gp()'s raw GP prior.

Implemented so far:
    Step 1 (build_environment): resolve the single fixed environment - map csv, starting
        position/heading, and the GP prior/baseline RMSE every episode is scored against.
    Step 2 (compute_step_reward): score one environment step's executed chunk by how much
        reconstruction RMSE improved during that chunk, via the GP/Kalman measurement pipeline,
        minus a penalty proportional to the environment step number (encourages reducing RMSE
        as early in the episode as possible).

    step 3 (DPPO rollout/update loop): output the loglikelihoods of each denoising step

    Step 4/5 (collect_rollouts / RolloutBuffer): run one full multi-step episode (T
        execute-then-replan environment steps, each with a K-step denoising replan) under
        pi_theta_old, score every step's chunk with compute_step_reward, and pack every
        (environment step, denoising step) transition - x_k, x_k_minus_1, k, k_prev,
        old_log_prob, reward - into a flat RolloutBuffer ready for PPO minibatching.

    Step 6 (compute_advantages): real GAE across environment steps (paper Eq. A2) - V_phi is
        evaluated at every environment step's own (evolving) belief state, not just the
        episode's initial one, bootstrapping V(s_{t+1}) into each TD residual. Also returns the
        discounted return-to-go per step, which update_value now regresses toward (Eq. A4)
        instead of the raw one-step reward.

    Step 7 (collect_pooled_episodes / MPI): each training iteration pools NUM_EPISODES_PER_ITERATION
        independent episodes (paper Algorithm 1's "N environments in parallel") against the same
        fixed map, rather than training off one noisy episode per iteration - a single episode's
        reward is dominated by sampling noise (eta=1 stochastic denoising + sensor noise), and
        reusing it for many PPO epochs was previously overfitting to that noise instead of
        learning real signal. Distributed round-robin across MPI ranks if launched under
        mpirun/srun/mpiexec (see get_mpi_context/assigned_tasks_for_rank, same conventions as
        DataCollector_beamsearch.py); falls back to one serial process otherwise.

    Step 8 (visualization / checkpointing): after every iteration, if VISUALIZE is True, saves a
        top-down + 3D trajectory plot of one representative pooled episode to
        `rl correction images/` (plot_iteration_trajectory), so training instability can be
        diagnosed visually instead of only from aggregate RMSE/reward numbers. Always saves the
        current model/value_fn weights to `dppo_checkpoints/` (_save_checkpoint), so a specific
        iteration's policy can be inspected or reloaded later instead of being lost once training
        regresses past it.

    Step 9 (stability guards): collect_pooled_episodes standardizes the pooled row_advantage
        (zero mean, unit std across the whole batch, not per-episode) before PPO uses it, so
        gradient step size stays consistent iteration to iteration regardless of that
        iteration's particular raw advantage scale. _ppo_step/_value_step clip gradient norm to
        MAX_GRAD_NORM on both optimizers - clip_epsilon alone only bounds how much of a step
        counts toward the loss, not how large the underlying gradient is. update_policy also
        tracks an approx_kl per epoch and stops that iteration's remaining epochs early once it
        exceeds TARGET_KL (paper Appendix B's KL-divergence termination criterion) - added
        because clip_fraction telemetry showed clip_epsilon=0.1 still saturating near 100% most
        iterations (see CLIP_EPSILON's comment), meaning multiple epochs per iteration were
        running well past the point where the clipped objective is still trustworthy.

HOW TO RUN:
    Run as a module from the `scripts` directory (not `python DPPOTRAINING.py` from inside
    Diffusion/) so that both the `Diffusion` package and its sibling modules at the scripts
    root (gaussianprocesstraining, evalmetrics, CMAES_classic_singlemap,
    Diffusionplanner_singlemap) are importable:
        python -m Diffusion.DPPOTRAINING

    To parallelize episode collection across multiple processes (e.g. on an HPC node), launch
    under MPI instead - every rank must have the same map_id/env config, and rank 0 is the only
    one that ever performs the policy/value update (others just collect episodes and receive the
    updated weights each iteration):
        mpiexec -n <N> python -m Diffusion.DPPOTRAINING
        srun --mpi=pmix python -m Diffusion.DPPOTRAINING
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3D projection

try:
    from mpi4py import MPI
except ImportError:
    MPI = None

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from gaussianprocesstraining import build_spline_trajectory_3d, initialize_gp
from evalmetrics import compute_reconstruction_rmse
from CMAES_classic_singlemap import dynamics_3d, waypoint_3d
from Diffusionplanner_singlemap import apply_measurement_update_3d, build_true_map_flat

from DPPOvaluefunction import ValueFunction
from threeDSparseTransDiffusion import NoisePredictor
import threeDSparseTransDiffusion as diffusion


# ---------------------------------------------------------------------------
# MPI helpers (same conventions as DataCollector_beamsearch.py). Everything below degrades to a
# single serial process if mpi4py isn't installed or the script isn't launched under
# mpirun/srun/mpiexec (MPI is None, or comm is None) - no separate code path needed for local
# testing vs an HPC job.
# ---------------------------------------------------------------------------

def get_mpi_context():
    if MPI is None:
        return None, 0, 1
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size()


def detect_mpi_launch_issue(env, mpi_size):
    slurm_ntasks = int(env.get("SLURM_NTASKS", "1"))
    return slurm_ntasks > 1 and mpi_size == 1


def assigned_tasks_for_rank(tasks, rank, size):
    return tasks[rank::size]


def _broadcast_module_state(module, comm, mpi_rank):
    """Keeps `module`'s weights identical across all MPI ranks - broadcasts rank 0's state_dict
    to everyone else. No-op if MPI isn't in use (comm is None)."""
    if comm is None:
        return
    state_dict = module.state_dict() if mpi_rank == 0 else None
    state_dict = comm.bcast(state_dict, root=0)
    if mpi_rank != 0:
        module.load_state_dict(state_dict)


# ---------------------------------------------------------------------------
# Step 1: the single, fixed environment
# ---------------------------------------------------------------------------

SENSORNOISE_SEED = 123
MAPTYPE = os.environ.get("MAPTYPE", "grf")
CSV_PATH = REPO_DIR / "csv"
UTILITY_THRESHOLD = 0.5
SAMPLESTEP = 2.0

# T_a in the paper's terms: measurement updates executed per environment step before replanning.
EXECUTION_CHUNK = 20
# T in the paper's terms: number of execute-then-replan environment steps per episode.
# Deliberately not 30, so it's never confused with K (denoising steps, also 30 right now).
ENV_HORIZON = 30

# K' in the paper's terms (Section 4.3, "Fine-tune only the last few denoising steps"): the
# full K=diffusion.T=30-step denoising chain still runs every replan (needed to get a coherent
# sample at all), but only the transitions from the LAST NUM_FINE_TUNE_STEPS steps (closest to
# k=0, the terminal/least-noisy step) get stored in the buffer and trained on - the early,
# noisiest steps are treated as frozen/untouched, matching the paper's theta vs theta_FT split
# (though we don't maintain a literal separate frozen copy of the weights, just skip recording
# and gradient-updating on those early steps).
NUM_FINE_TUNE_STEPS = 10

# Reward weighting. Both RMSE terms are expressed as a *fraction* of their own no-flight
# baseline (e.g. 0.3 = flying reduced that RMSE by 30%) rather than raw differences, so they
# land on a comparable ~[-1, 1] scale despite occupied_rmse's baseline (0.0934) being smaller
# than global_rmse's (0.1043) - a weight of 1.0 on each now genuinely means "equally important",
# instead of also having to compensate for the two baselines' different absolute scales.
GLOBAL_RMSE_WEIGHT = 1.0
OCCUPIED_RMSE_WEIGHT = 1.0
# Penalty proportional to the environment step number (0-indexed) - encourages reducing RMSE as
# early in the episode as possible rather than spreading improvement out over many steps. Not a
# discount (that's ENV_GAMMA/GAE_LAMBDA further down, which weights how much future reward
# counts toward the return) - this is a direct per-step cost that grows with elapsed steps, so
# the same RMSE improvement is worth strictly less the later it happens.
STEP_PENALTY_WEIGHT = 0.003

NUM_PPO_EPOCHS = 10
# Scales with NUM_EPISODES_PER_ITERATION (64 at 6 episodes/iteration -> 96 at 9) - DPPO's
# gradient averages over environment steps x denoising steps jointly, so minibatch size should
# track how much pooled data is actually available per iteration (paper Appendix B, "Large
# batch size"), not stay fixed while NUM_EPISODES_PER_ITERATION changes.
MINIBATCH_SIZE = 512
# Paper (Table A7/A9) tunes epsilon per task in {0.1, 0.01, 0.001}, picking the highest value
# that keeps the batch's clipping fraction (see _ppo_step's returned clip_fraction) in the
# 10-20% range. 0.1 measured 60-100% clip_fraction in practice (slurm-10340607.out) - moved to
# the paper's actual DPPO value of 0.01. Watch the printed clip_fraction below; drop to 0.001
# next if it's still above ~20%.
CLIP_EPSILON = 0.01

# Paper (Appendix B): "the batch updates in an iteration terminate when the KL divergence
# between pi_theta and pi_theta_old reaches 1, although in practice we find this never
# happens" (under their tuned hyperparameters). Ours isn't necessarily tuned enough yet for that
# to hold, so this is a live circuit breaker on update_policy's epoch loop, not decoration - see
# _ppo_step's returned approx_kl.
TARGET_KL = 1.0

# Standard PPO safety net (e.g. OpenAI baselines' PPO2 default) independent of clip_epsilon -
# clip_epsilon only stops the LOSS from rewarding ratio moving past the threshold, it doesn't
# bound how large a single minibatch's gradient can be. A handful of outlier-advantage rows
# could still yank the shared weights hard even after advantage normalization (see
# collect_pooled_episodes) reduces how often that happens.
MAX_GRAD_NORM = 0.5

NUM_VALUE_EPOCHS = 10
VALUE_LR = 1e-3

# GAE across environment steps (paper Eq. A2): gamma_ENV discounts reward across steps t (NOT
# denoising steps k - that's still denoising_gamma, passed separately to compute_advantages),
# lambda trades off bias/variance in the advantage estimate the usual GAE way.
ENV_GAMMA = 0.99
GAE_LAMBDA = 0.95

# Number of DPPO training iterations (each iteration = pool NUM_EPISODES_PER_ITERATION episodes
# via collect_pooled_episodes, then update_policy + update_value off the pooled buffer).
NUM_DPPO_ITERATIONS = 10

# N in the paper's "N environments in parallel" (Algorithm 1): independent episodes collected
# and pooled together before each policy/value update. A single episode's reward is dominated by
# sampling noise (stochastic eta=1 denoising + sensor measurement noise) - pooling several
# independent episodes and averaging their advantage estimates is what actually reduces that
# noise, rather than reusing one noisy episode for many PPO epochs (NUM_PPO_EPOCHS) and
# overfitting to whatever that one episode happened to look like. Distributed round-robin across
# MPI ranks if launched under mpirun/srun/mpiexec; collected serially by one process otherwise.
NUM_EPISODES_PER_ITERATION = 16

# Which map csv every DPPO episode in this file trains against.
MAP_ID = 11
# Starting heading/velocity for every episode - a raw (unnormalized) world-frame direction
# vector, not literally unit-length; normalize_xyz_displacement handles scaling it into the
# model's conditioning space.
INITIAL_HEADING_VELOCITY = (1.0, 1.0, 1.0)

#SIGMA PROB MIN FOR LIKELIHOODS
SIGMA_PROB_MIN = 0.3

DEFAULT_CHECKPOINT = (
    SCRIPT_DIR / "behaviouralcloninginstance" / "sparse_trans_waypoints_epoch_1900_multimodal_3d.pth"
)

# Toggle: dump a top-down + 3D trajectory plot of one representative pooled episode after every
# iteration (see plot_iteration_trajectory) - lets training instability be diagnosed visually
# (orbiting in place, leaving the map, etc.) instead of only from aggregate RMSE/reward numbers.
VISUALIZE = True
IMAGE_DIR = SCRIPT_DIR / "rl_correction_images"

# Every iteration's fine-tuned policy/value weights, so a specific iteration can be reloaded
# later (e.g. to inspect why training regressed past it) - distinct from the pretrained BC
# checkpoint DEFAULT_CHECKPOINT loads from.
CHECKPOINT_DIR = SCRIPT_DIR / "dppo_checkpoints"


@dataclass
class SingleEnvironment:
    """Everything needed to sample from, and score trajectories against, one fixed
    (map, current_position, heading) scenario."""

    map_id: int

    # Diffusion conditioning tensors, batch dim = 1, ready to feed into NoisePredictor.
    meanvarmarker_map: torch.Tensor
    current_position: torch.Tensor
    initial_heading_velocity: torch.Tensor
    # Raw (unnormalized) world-frame version of the above - the starting point collect_rollouts
    # tracks and recomputes from actual position deltas every environment step (heading
    # continuity), since initial_heading_velocity itself is fixed/normalized once and never
    # reflects how the drone actually moved.
    initial_heading_velocity_world: tuple

    # World-frame starting pose used to simulate flight.
    cx0: float
    cy0: float
    cz0: float

    # GP / grid setup, shared by every episode sampled against this map.
    gp: object
    X_test: np.ndarray
    xs: np.ndarray
    ys: np.ndarray
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    grid_step: float

    # Ground truth for this map, and the GP prior before any flight.
    true_map_flat: np.ndarray
    pts: np.ndarray
    mu0: np.ndarray
    P0: np.ndarray

    # RMSE of the GP prior mean against ground truth, before any measurements are taken.
    # This is the baseline every sampled trajectory is scored against.
    baseline_global_rmse: float
    baseline_occupied_rmse: float


def _load_true_map(map_id, X_test, step):
    data = np.loadtxt(
        CSV_PATH / f"map_{map_id}_{MAPTYPE}_grid_counts.csv",
        delimiter=",",
        skiprows=1,
    )
    pts = data[:, 0:3]

    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    true_map_flat = build_true_map_flat(pts, X_test)
    return true_map_flat, pts


def _build_belief_conditioning(mu, P, cx, cy, cz, xs, ys, device):
    """Builds (meanvarmarker_map, current_position) model-conditioning tensors from a raw belief
    state (mu, P, position) - the same construction Diffusionplanner_singlemap.py's
    sample_diffusion_trajectory uses for an arbitrary (non-dataset-indexed) state. Reusable both
    for the initial state (build_environment) and every subsequent environment step's replan,
    since the belief evolves every step once measurements start coming in."""
    current_position_world = torch.tensor([[cx, cy, cz]], dtype=torch.float32, device=device)
    current_position = diffusion.normalize_xyz(current_position_world)

    grid_shape = (len(ys), len(xs))
    mean_grid = torch.tensor(mu.reshape(grid_shape), dtype=torch.float32, device=device)
    var_grid = torch.tensor(np.diag(P).reshape(grid_shape), dtype=torch.float32, device=device)
    mean_map = (mean_grid - diffusion.mean_center.to(device)) / diffusion.mean_scale.to(device)
    var_map = (var_grid - diffusion.var_center.to(device)) / diffusion.var_scale.to(device)
    marker_map = diffusion.make_position_marker_maps(
        current_position_world[:, :2].cpu(), grid_size=51, marker_radius=2,
    ).to(device)[0, 0]
    meanvarmarker_map = torch.stack([mean_map, var_map, marker_map], dim=0).unsqueeze(0)

    return meanvarmarker_map, current_position


def build_environment(map_id=MAP_ID, cx0=None, cy0=None, cz0=None, heading_velocity=INITIAL_HEADING_VELOCITY):
    """Step 1: resolve the single fixed (map, starting position, heading) scenario that every
    DPPO episode in this file samples against. The initial belief state comes directly from
    initialize_gp()'s raw GP prior (zero mean, kernel-induced covariance) - the same starting
    point Diffusionplanner_singlemap.py's receding-horizon loop uses - rather than a
    diffusion-dataset row, since map_id/starting pose are now chosen directly instead of looked
    up from a precomputed condition_index."""

    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, grid_step = initialize_gp()

    cx0 = diffusion.XY_SCALE / 2.0 if cx0 is None else cx0
    cy0 = diffusion.XY_SCALE / 2.0 if cy0 is None else cy0
    cz0 = diffusion.Z_MIN if cz0 is None else cz0

    true_map_flat, pts = _load_true_map(map_id, X_test, grid_step)

    # mean/cov are already raw units (matching true_map_flat) since they come straight from the
    # GP prior, not a normalized diffusion-dataset row - no de-normalization needed anymore.
    mu0 = mean.copy()
    P0 = cov.copy()

    device = diffusion.device
    meanvarmarker_map, current_position = _build_belief_conditioning(
        mu0, P0, cx0, cy0, cz0, xs, ys, device,
    )

    raw_heading = torch.tensor([list(heading_velocity)], dtype=torch.float32, device=device)
    initial_heading_velocity = diffusion.normalize_xyz_displacement(raw_heading)

    baseline_metrics = compute_reconstruction_rmse(
        mu=mu0,
        pts=pts,
        xs=xs,
        ys=ys,
        step=grid_step,
        utility_threshold=UTILITY_THRESHOLD,
        xmin=xmin,
        ymin=ymin,
    )

    return SingleEnvironment(
        map_id=map_id,
        meanvarmarker_map=meanvarmarker_map,
        current_position=current_position,
        initial_heading_velocity=initial_heading_velocity,
        initial_heading_velocity_world=tuple(heading_velocity),
        cx0=cx0,
        cy0=cy0,
        cz0=cz0,
        gp=gp,
        X_test=X_test,
        xs=xs,
        ys=ys,
        xmin=xmin,
        xmax=xmax,
        ymin=ymin,
        ymax=ymax,
        grid_step=grid_step,
        true_map_flat=true_map_flat,
        pts=pts,
        mu0=mu0,
        P0=P0,
        baseline_global_rmse=baseline_metrics["global_rmse"],
        baseline_occupied_rmse=baseline_metrics["occupied_rmse"],
    )


# ---------------------------------------------------------------------------
# Step 2: execute one environment step's chunk and score it
# ---------------------------------------------------------------------------

def _rollout_trajectory(traj_world, mu, P, cx, cy, cz, env, rng, max_measurement_updates=EXECUTION_CHUNK,
                         position_history=None):
    """Fly `traj_world` (world-frame control waypoints, shape [NUM_CONTROL_WAYPOINTS, 3]) for up
    to `max_measurement_updates` GP measurement updates (one environment step's chunk, T_a in
    the paper), starting from the given belief state and position - NOT env.mu0/env.cx0 anymore,
    since this now gets called once per environment step with the belief as it stood at the
    start of that step, not always the episode's initial state. Returns the belief/position as
    they stand after this chunk, i.e. s_{t+1}.

    Matches Diffusionplanner_singlemap.py's receding-horizon loop's cadence exactly: ONE
    dynamics_3d call and ONE apply_measurement_update_3d call per iteration, unconditionally -
    unlike a naive dense-path walk that skips both when a waypoint is already reached, this
    advances spline_idx (at most one extra waypoint per iteration) and keeps measuring at the
    same cadence regardless, so a chunk always costs exactly `max_measurement_updates` real
    measurement updates (or fewer only if the plan runs out first, matching the reference's
    `spline_idx >= len(spline_path)` early-replan trigger).

    position_history: if a list is passed, every (cx, cy, cz) visited during this chunk is
    appended to it in place - used by collect_rollouts(record_trajectory=True) to build one
    episode's full flown path for plot_iteration_trajectory."""

    dense_path = build_spline_trajectory_3d(
        cx, cy, cz, traj_world, samples_per_segment=5,
    )
    spline_idx = 0

    for _ in range(max_measurement_updates):
        if spline_idx >= len(dense_path):
            break

        goal_x, goal_y, goal_z = dense_path[spline_idx]
        grad_x, grad_y, grad_z, waypoint_reached = waypoint_3d(
            cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=env.grid_step,
        )
        if waypoint_reached:
            spline_idx += 1
            if spline_idx < len(dense_path):
                goal_x, goal_y, goal_z = dense_path[spline_idx]
                grad_x, grad_y, grad_z, _ = waypoint_3d(
                    cx, cy, cz, goal_x=goal_x, goal_y=goal_y, goal_z=goal_z, step=env.grid_step,
                )
            else:
                grad_x, grad_y, grad_z = 0.0, 0.0, 0.0

        cx, cy, cz = dynamics_3d(
            cx, cy, cz, grad_x, grad_y, grad_z, SAMPLESTEP,
            env.xmin, env.xmax, env.ymin, env.ymax,
            diffusion.Z_MIN, diffusion.Z_MAX,
            buffer=env.grid_step / 2,
        )

        mu, P = apply_measurement_update_3d(
            cx, cy, cz, mu, P, env.true_map_flat, env.xs, env.ys, rng,
        )

        if position_history is not None:
            position_history.append((cx, cy, cz))

    return mu, P, cx, cy, cz


def _rmse_metrics(mu, env):
    return compute_reconstruction_rmse(
        mu=mu,
        pts=env.pts,
        xs=env.xs,
        ys=env.ys,
        step=env.grid_step,
        utility_threshold=UTILITY_THRESHOLD,
        xmin=env.xmin,
        ymin=env.ymin,
    )


def compute_step_reward(mu_before, mu_after, env, env_step):
    """Step 2: score one environment step's executed chunk - R(s_t, a_t) - by how much
    reconstruction RMSE improved *during this chunk* (mu_before -> mu_after), as a fraction of
    the episode's fixed no-flight baseline, minus a penalty proportional to how late in the
    episode this step is - pushes the policy toward reducing RMSE as fast as possible rather
    than spreading the same total improvement out over more steps.

    mu_before, mu_after: the posterior mean grid immediately before and after this step's chunk
        was flown (i.e. what _rollout_trajectory returned at the start/end of this step).
    env_step: this step's index (t in the paper), 0 for the episode's first replan.

    Returns (reward, global_rmse, occupied_rmse) - the post-chunk RMSE values are returned
    alongside the scalar reward so callers (collect_rollouts) can track them without re-simulating.
    """

    before_metrics = _rmse_metrics(mu_before, env)
    after_metrics = _rmse_metrics(mu_after, env)
    global_rmse = after_metrics["global_rmse"]
    occupied_rmse = after_metrics["occupied_rmse"]

    global_rmse_improvement_pct = (
        (before_metrics["global_rmse"] - global_rmse) / env.baseline_global_rmse
    )
    occupied_rmse_improvement_pct = (
        (before_metrics["occupied_rmse"] - occupied_rmse) / env.baseline_occupied_rmse
    )

    reward = (
        GLOBAL_RMSE_WEIGHT * global_rmse_improvement_pct
        + OCCUPIED_RMSE_WEIGHT * occupied_rmse_improvement_pct
        - STEP_PENALTY_WEIGHT * env_step
    )

    return float(reward), float(global_rmse), float(occupied_rmse)


# ---------------------------------------------------------------------------
# Step 4: rollout buffer
# ---------------------------------------------------------------------------

@dataclass
class RolloutBuffer:
    """Flat buffer of denoising-step transitions across all `T` environment steps of one
    episode. One row per (environment step, denoising step) pair - i.e. per (t, k) in the
    paper's two-layer indexing - so PPO minibatches can sample rows uniformly across both."""

    env_step: torch.Tensor       # [T*K] long  - which environment step (t in the paper) this
                                  #               row belongs to; 0 is the episode's first replan
    k: torch.Tensor              # [T*K] long  - denoising step index at this row (this is the
                                  #               paper's k; k=0 is the terminal, least-noisy step
                                  #               whose action actually gets executed)
    k_prev: torch.Tensor         # [T*K] long  - denoising step index of the *next* row (-1 at
                                  #               the terminal step)
    x_k: torch.Tensor            # [T*K, 3, 8] - noisy action fed into the model this row
                                  #               (a_t^{k+1} in the paper)
    x_k_minus_1: torch.Tensor    # [T*K, 3, 8] - action actually sampled this row (a_t^k)
    old_log_prob: torch.Tensor   # [T*K]       - log pi_theta_old(x_k_minus_1 | x_k), summed
                                  #               over the 3x8 action dims, under the frozen
                                  #               rollout-time weights
    reward: torch.Tensor         # [T*K]       - 0 everywhere except the terminal (k=0) row of
                                  #               each environment step, which holds that step's
                                  #               chunk reward R(s_t, a_t)
    global_rmse: torch.Tensor    # [T*K]       - 0 everywhere except the terminal (k=0) row;
                                  #               post-chunk global RMSE for that environment step
    occupied_rmse: torch.Tensor  # [T*K]       - same, but the occupied-cells RMSE

    # Per-environment-step (T-length, NOT T*K) conditioning tensors - one entry per env_step
    # value 0..T-1, indexed directly. Needed so compute_advantages/update_value can evaluate
    # V_phi at each step's own (evolving) belief state for the paper's GAE (Eq. A2) and value
    # target (Eq. A4), instead of only ever seeing the episode's fixed initial state.
    step_meanvarmarker_map: torch.Tensor  # [T, 3, 51, 51]
    step_current_position: torch.Tensor   # [T, 3]
    step_heading_velocity: torch.Tensor   # [T, 3]  - normalized heading actually used to
                                           #           condition this step's replan (heading
                                           #           continuity - recomputed from real position
                                           #           deltas, not the fixed episode-start value)

    def __len__(self):
        return self.env_step.shape[0]


def _offset_env_step(buffer, offset):
    """Returns a copy of `buffer` with env_step shifted by `offset`, so it can be concatenated
    with other episodes' buffers without their per-step state tensors (step_meanvarmarker_map
    etc.) aliasing - env_step doubles as the index into those per-step tensors."""
    return RolloutBuffer(
        env_step=buffer.env_step + offset,
        k=buffer.k,
        k_prev=buffer.k_prev,
        x_k=buffer.x_k,
        x_k_minus_1=buffer.x_k_minus_1,
        old_log_prob=buffer.old_log_prob,
        reward=buffer.reward,
        global_rmse=buffer.global_rmse,
        occupied_rmse=buffer.occupied_rmse,
        step_meanvarmarker_map=buffer.step_meanvarmarker_map,
        step_current_position=buffer.step_current_position,
        step_heading_velocity=buffer.step_heading_velocity,
    )


def _concat_rollout_buffers(buffers):
    """Pools multiple independent episodes' RolloutBuffers (each its own T-step episode) into
    one combined buffer. Re-indexes env_step across episodes first so per-step state lookups
    (buffer.step_*[buffer.env_step[idx]]) stay correct once concatenated - episode i's env_step
    values get offset by the total step-count of episodes 0..i-1, so they never collide with
    another episode's steps."""
    offset = 0
    shifted = []
    for buf in buffers:
        shifted.append(_offset_env_step(buf, offset))
        offset += buf.step_meanvarmarker_map.shape[0]

    return RolloutBuffer(
        env_step=torch.cat([b.env_step for b in shifted]),
        k=torch.cat([b.k for b in shifted]),
        k_prev=torch.cat([b.k_prev for b in shifted]),
        x_k=torch.cat([b.x_k for b in shifted]),
        x_k_minus_1=torch.cat([b.x_k_minus_1 for b in shifted]),
        old_log_prob=torch.cat([b.old_log_prob for b in shifted]),
        reward=torch.cat([b.reward for b in shifted]),
        global_rmse=torch.cat([b.global_rmse for b in shifted]),
        occupied_rmse=torch.cat([b.occupied_rmse for b in shifted]),
        step_meanvarmarker_map=torch.cat([b.step_meanvarmarker_map for b in shifted]),
        step_current_position=torch.cat([b.step_current_position for b in shifted]),
        step_heading_velocity=torch.cat([b.step_heading_velocity for b in shifted]),
    )


def _denoising_schedule(num_steps=None):
    """Same schedule construction as diffusion.ddpo_ddim_sample. Reimplemented here (rather
    than reused) because the rollout loop below drives diffusion.ddpo_ddim_sample_timestep one
    step at a time so it can keep every intermediate x_k - ddpo_ddim_sample itself only returns
    the final sample, which isn't enough to rebuild the log-prob under new weights later."""

    if num_steps is None or num_steps >= diffusion.T:
        return list(range(diffusion.T - 1, -1, -1))
    return (
        torch.linspace(diffusion.T - 1, 0, steps=num_steps)
        .long()
        .unique_consecutive()
        .tolist()
    )


@torch.no_grad()
def collect_rollouts(env, num_steps=None, clip_x0=True, eta=1.0, base_seed=None,
                      horizon=ENV_HORIZON, chunk_size=EXECUTION_CHUNK,
                      num_fine_tune_steps=NUM_FINE_TUNE_STEPS, record_trajectory=False):
    """Step 4/5: run ONE full episode of `horizon` execute-then-replan environment steps against
    `env`, under the current (frozen) policy weights - i.e. pi_theta_old. At each environment
    step t: rebuild the belief-conditioning tensors from the CURRENT (mu, P, position), run the
    FULL K-step denoising loop once to sample a fresh action (needed for a coherent sample), but
    only record transitions from the last `num_fine_tune_steps` steps (k < num_fine_tune_steps)
    into the buffer, execute `chunk_size` measurement updates of the resulting action, score the
    chunk with compute_step_reward, and carry the resulting belief state forward as s_{t+1}.
    Every recorded (t, k) denoising-step transition across the whole episode gets packed into
    one flat RolloutBuffer.

    Assumes diffusion.model already holds the weights to roll out with (see _load_checkpoint).

    Runs under torch.no_grad() deliberately: pi_theta_old must stay fixed for the whole PPO
    update phase (step 5), so rollout collection never needs gradients here - the update step
    will later re-run diffusion.ddpo_ddim_sample_timestep WITH grad at the stored
    (x_k, x_k_minus_1) pairs under the *current* weights to get the new log-probs for the PPO
    ratio.

    record_trajectory: if True, also returns this episode's full flown (cx, cy, cz) history
    (starting position plus every position visited during every chunk) as a second return value,
    for plot_iteration_trajectory. None when False.
    """

    schedule = _denoising_schedule(num_steps)
    device = env.meanvarmarker_map.device

    env_steps, ks, k_prevs = [], [], []
    x_ks, x_k_minus_1s, old_log_probs, rewards = [], [], [], []
    global_rmses, occupied_rmses = [], []
    step_meanvarmarker_maps, step_current_positions, step_heading_velocities = [], [], []

    if base_seed is not None:
        torch.manual_seed(base_seed)

    mu, P, cx, cy, cz = env.mu0.copy(), env.P0.copy(), env.cx0, env.cy0, env.cz0
    position_history = [(cx, cy, cz)] if record_trajectory else None
    # Heading continuity (matches Diffusionplanner_singlemap.py): starts from the episode's
    # fixed initial heading, then gets recomputed from actual position deltas after every
    # chunk, so each replan conditions on the direction the drone is really moving in rather
    # than a value fixed for the whole episode.
    heading_velocity_world = np.array(env.initial_heading_velocity_world, dtype=np.float32)
    rng_seed = SENSORNOISE_SEED + env.map_id if base_seed is None else SENSORNOISE_SEED + env.map_id + base_seed
    rng = np.random.default_rng(rng_seed)

    for t in range(horizon):
        # Belief evolves every step once measurements start coming in, so this has to be
        # rebuilt from the CURRENT (mu, P, cx, cy, cz), not env's fixed initial-state tensors.
        meanvarmarker_map, current_position = _build_belief_conditioning(
            mu, P, cx, cy, cz, env.xs, env.ys, device,
        )
        heading_velocity = diffusion.normalize_xyz_displacement(
            torch.tensor(heading_velocity_world[None, :], dtype=torch.float32, device=device)
        )
        step_meanvarmarker_maps.append(meanvarmarker_map[0])
        step_current_positions.append(current_position[0])
        step_heading_velocities.append(heading_velocity[0])

        x = torch.randn((1, *diffusion.TARGET_SHAPE), device=device)
        for step_idx, step_k in enumerate(schedule):
            prev_k = schedule[step_idx + 1] if step_idx + 1 < len(schedule) else -1
            k = torch.full((1,), step_k, dtype=torch.long, device=device)
            k_prev = torch.full((1,), prev_k, dtype=torch.long, device=device)

            x_prev, mean_theta, var = diffusion.ddpo_ddim_sample_timestep(
                x, k, k_prev,
                meanvarmarker_map, current_position, heading_velocity,
                clip_x0=clip_x0, eta=eta, sigma_prob_min=SIGMA_PROB_MIN,
            )

            if step_k < num_fine_tune_steps:
                var = var.clamp_min(1e-8).expand_as(mean_theta)
                log_prob = -diffusion.gaussian_nll(input=mean_theta, target=x_prev, var=var)
                log_prob = log_prob.sum(dim=(1, 2))  # joint log-prob over 24 action dims -> [1]

                env_steps.append(t)
                ks.append(step_k)
                k_prevs.append(prev_k)
                x_ks.append(x[0])
                x_k_minus_1s.append(x_prev[0])
                old_log_probs.append(log_prob[0])

            x = x_prev

        # x now holds this step's fully-denoised (k=0) action; execute one chunk of it, then
        # score that chunk and carry the resulting belief state forward as s_{t+1}.
        traj_world = diffusion.extract_control_waypoints(x[0]).detach().cpu().numpy().T
        mu_before = mu
        previous_pose = np.array([cx, cy, cz], dtype=np.float32)
        mu, P, cx, cy, cz = _rollout_trajectory(
            traj_world, mu, P, cx, cy, cz, env, rng, chunk_size, position_history=position_history,
        )
        heading_velocity_world = np.array([cx, cy, cz], dtype=np.float32) - previous_pose
        reward, global_rmse, occupied_rmse = compute_step_reward(
            mu_before, mu, env, t,
        )
        rows_this_step = min(num_fine_tune_steps, len(schedule))
        rewards.extend([0.0] * (rows_this_step - 1) + [reward])
        global_rmses.extend([0.0] * (rows_this_step - 1) + [global_rmse])
        occupied_rmses.extend([0.0] * (rows_this_step - 1) + [occupied_rmse])

    buffer = RolloutBuffer(
        env_step=torch.tensor(env_steps, dtype=torch.long),
        k=torch.tensor(ks, dtype=torch.long),
        k_prev=torch.tensor(k_prevs, dtype=torch.long),
        x_k=torch.stack(x_ks),
        x_k_minus_1=torch.stack(x_k_minus_1s),
        old_log_prob=torch.stack(old_log_probs),
        reward=torch.tensor(rewards, dtype=torch.float32),
        global_rmse=torch.tensor(global_rmses, dtype=torch.float32),
        occupied_rmse=torch.tensor(occupied_rmses, dtype=torch.float32),
        step_meanvarmarker_map=torch.stack(step_meanvarmarker_maps),
        step_current_position=torch.stack(step_current_positions),
        step_heading_velocity=torch.stack(step_heading_velocities),
    )
    return buffer, position_history


def _load_checkpoint(checkpoint_path, device):
    model = NoisePredictor().to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = (
        payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload
        else payload
    )
    state_dict = diffusion.remap_legacy_state_dict_keys(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    # ddim_sample_timestep/ddpo_ddim_sample_timestep read the module-level `model` global
    # directly (it isn't passed as a parameter), so it must be reassigned here or the
    # sampler will silently keep using the randomly-initialized model from import time.
    diffusion.model = model
    return model


def compute_advantages(buffer, value_fn, env, denoising_gamma=0.99, env_gamma=ENV_GAMMA, gae_lambda=GAE_LAMBDA):
    """GAE across environment steps (paper Eq. A2), then broadcast to every denoising row and
    discount by denoising_gamma^k same as before. Also returns the discounted return-to-go per
    environment step, which update_value needs as its regression target (Eq. A4) instead of the
    old one-step-reward target.

    Unlike the old single-step version, V_phi is evaluated at EVERY environment step's own
    (evolving) belief state - buffer.step_meanvarmarker_map/step_current_position - not just the
    episode's fixed initial state.
    """
    with torch.no_grad():
        num_env_steps = buffer.step_meanvarmarker_map.shape[0]

        state_values = value_fn(
            buffer.step_meanvarmarker_map,
            buffer.step_current_position,
            buffer.step_heading_velocity,
        ).squeeze(-1)  # [T]

        step_reward = buffer.reward[buffer.k == 0]  # [T], one reward per environment step

        # TD residuals delta_t = R_t + gamma_ENV * V(s_{t+1}) - V(s_t); V(s_T) = 0 since every
        # episode here is a fixed-length T-step rollout with no bootstrapping past the horizon.
        next_state_values = torch.cat([state_values[1:], torch.zeros(1)])
        deltas = step_reward + env_gamma * next_state_values - state_values

        # GAE backward recursion: A_t = delta_t + (gamma_ENV * lambda) * A_{t+1}, A_T = 0.
        step_advantages = torch.zeros(num_env_steps)
        running_advantage = 0.0
        for t in reversed(range(num_env_steps)):
            running_advantage = deltas[t].item() + env_gamma * gae_lambda * running_advantage
            step_advantages[t] = running_advantage

        # Discounted return-to-go per step: R_t + gamma_ENV * R_{t+1} + ... - update_value's
        # regression target, not the raw one-step reward.
        step_returns = torch.zeros(num_env_steps)
        running_return = 0.0
        for t in reversed(range(num_env_steps)):
            running_return = step_reward[t].item() + env_gamma * running_return
            step_returns[t] = running_return

        # Broadcast each environment step's advantage to every denoising row belonging to it,
        # then apply the existing denoising discount gamma_DENOISE^k on top (still per-row).
        row_advantage = step_advantages[buffer.env_step]
        denoising_discount = denoising_gamma ** (buffer.k.float())
        row_advantage = row_advantage * denoising_discount

        return row_advantage, step_returns


def collect_pooled_episodes(env, value_fn, num_episodes, comm, mpi_rank, mpi_size, base_seed,
                             denoising_gamma=0.99, env_gamma=ENV_GAMMA, gae_lambda=GAE_LAMBDA,
                             record_trajectory=False):
    """Collects `num_episodes` independent episodes against the same fixed `env`, distributed
    round-robin across MPI ranks (assigned_tasks_for_rank - same convention as
    DataCollector_beamsearch.py's task assignment), computes GAE/return-to-go PER EPISODE
    locally on each rank, then gathers every rank's results to rank 0 and pools them into one
    combined RolloutBuffer plus concatenated (row_advantage, step_returns) ready for
    update_policy/update_value.

    GAE must be computed per-episode, not once on the pooled buffer - each episode is its own
    independent T-step sequence, and bootstrapping V(s_{t+1}) across two *different* episodes'
    steps would be meaningless (they don't follow from one another).

    value_fn must already be identical across all ranks (see _broadcast_module_state) before
    calling this, so every rank's local GAE computation agrees with the others'.

    record_trajectory: if True, episode 0 (always assigned to rank 0 by assigned_tasks_for_rank's
    round robin) also records its flown position history, returned as a 4th value for
    plot_iteration_trajectory.

    The pooled row_advantage is standardized (zero mean, unit std) across the WHOLE pooled batch
    before being returned - deliberately NOT per-episode, which would erase genuine differences
    between a good episode and a bad one. Raw GAE advantage magnitude can vary a lot from
    iteration to iteration depending on which episodes happened to get sampled; without this,
    an iteration whose pooled batch happens to have unusually large raw advantages produces a
    proportionally larger PPO gradient step regardless of clip_epsilon, since clipping only
    bounds how much of that push counts toward the loss, not how large the push actually is.

    Only rank 0's return values are meaningful; other ranks get (None, None, None, None).
    """
    episode_indices = list(range(num_episodes))
    local_indices = assigned_tasks_for_rank(episode_indices, mpi_rank, mpi_size)

    local_results = []
    for episode_idx in local_indices:
        buf, position_history = collect_rollouts(
            env, base_seed=base_seed + episode_idx,
            record_trajectory=record_trajectory and episode_idx == 0,
        )
        row_advantage, step_returns = compute_advantages(
            buf, value_fn, env,
            denoising_gamma=denoising_gamma, env_gamma=env_gamma, gae_lambda=gae_lambda,
        )
        local_results.append((buf, row_advantage, step_returns, position_history))

    gathered = [local_results] if comm is None else comm.gather(local_results, root=0)

    if mpi_rank != 0:
        return None, None, None, None

    all_results = [item for rank_results in gathered for item in rank_results]
    buffers = [item[0] for item in all_results]
    row_advantages = [item[1] for item in all_results]
    step_returns_list = [item[2] for item in all_results]
    iter_trajectory = next((item[3] for item in all_results if item[3] is not None), None)

    pooled_buffer = _concat_rollout_buffers(buffers)
    pooled_row_advantage = torch.cat(row_advantages)
    pooled_row_advantage = (
        (pooled_row_advantage - pooled_row_advantage.mean()) / (pooled_row_advantage.std() + 1e-8)
    )
    pooled_step_returns = torch.cat(step_returns_list)
    return pooled_buffer, pooled_row_advantage, pooled_step_returns, iter_trajectory


def _ppo_step(idx, buffer, discounted_advantage, env, policy_optimizer, clip_epsilon):

    """
    K is the number of denoising steps, T is the number of environment steps in the episode.
    idx: [T*K] long, indices of the rows in the buffer to use for this PPO update step
    buffer: RolloutBuffer, containing the sampled trajectories and their denoising-step transitions
    discounted_advantage: [T*K] float, the advantage values for each row in the buffer, discounted by denoising_gamma
    env: SingleEnvironment, the fixed environment used for sampling
    policy_optimizer: torch.optim.Optimizer, the optimizer for the policy network
    clip_epsilon: float, the PPO clipping parameter
    """
    device = env.meanvarmarker_map.device
    x_k = buffer.x_k[idx].to(device)
    x_k_minus_1 = buffer.x_k_minus_1[idx].to(device)
    k = buffer.k[idx].to(device)
    k_prev = buffer.k_prev[idx].to(device)
    old_log_prob = buffer.old_log_prob[idx].to(device)
    advantage = discounted_advantage[idx].to(device)

    # Each row's OWN environment step's state - NOT env's fixed initial state, which was only
    # ever correct for t=0. Mirrors what actually conditioned that row's sample at rollout time.
    env_step_idx = buffer.env_step[idx]
    meanvarmarker_map = buffer.step_meanvarmarker_map[env_step_idx].to(device)
    current_position = buffer.step_current_position[env_step_idx].to(device)
    initial_heading_velocity = buffer.step_heading_velocity[env_step_idx].to(device)

    _, mean_theta_new, var_new = diffusion.ddpo_ddim_sample_timestep(x_k, k, k_prev, meanvarmarker_map, current_position, initial_heading_velocity, clip_x0=True, eta=1.0, sigma_prob_min=SIGMA_PROB_MIN)

    var_new = var_new.clamp_min(1e-8).expand_as(mean_theta_new)

    new_log_prob = -diffusion.gaussian_nll(input=mean_theta_new, target=x_k_minus_1, var=var_new)
    new_log_prob = new_log_prob.sum(dim=(1, 2))
    logratio = new_log_prob - old_log_prob
    ratio = torch.exp(logratio)

    surrogate1 = ratio * advantage
    surrogate2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantage
    policy_loss = -torch.min(surrogate1, surrogate2).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean().item()
    # Low-variance KL estimator (Schulman, http://joschu.net/blog/kl-approx.html, "k3"),
    # aggregated by update_policy to stop this iteration's epoch loop early once pi_theta has
    # drifted too far from pi_theta_old (paper Appendix B's KL=1 termination criterion).
    approx_kl = ((ratio - 1.0) - logratio).mean().item()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm_(diffusion.model.parameters(), max_norm=MAX_GRAD_NORM)
    policy_optimizer.step()
    return policy_loss.item(), clip_fraction, approx_kl


def update_policy(buffer, discounted_advantage, env, policy_optimizer,
                   num_epochs=NUM_PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE, clip_epsilon=CLIP_EPSILON,
                   target_kl=TARGET_KL):
    terminal_idx = (buffer.k == 0).nonzero(as_tuple=True)[0]
    nonterminal_idx = (buffer.k != 0).nonzero(as_tuple=True)[0]

    for epoch in range(num_epochs):
        clip_fractions = []
        approx_kls = []
        for group in (terminal_idx, nonterminal_idx):
            perm = group[torch.randperm(group.shape[0])]
            for start in range(0, perm.shape[0], minibatch_size):
                idx = perm[start:start + minibatch_size]
                loss, clip_fraction, approx_kl = _ppo_step(
                    idx, buffer, discounted_advantage, env, policy_optimizer, clip_epsilon,
                )
                clip_fractions.append(clip_fraction)
                approx_kls.append(approx_kl)
        mean_clip_fraction = sum(clip_fractions) / len(clip_fractions)
        mean_approx_kl = sum(approx_kls) / len(approx_kls)
        print(
            f"epoch {epoch}: policy_loss={loss:.4f}, clip_fraction={mean_clip_fraction:.3f}, "
            f"approx_kl={mean_approx_kl:.4f}"
        )
        if mean_approx_kl > target_kl:
            print(
                f"epoch {epoch}: approx_kl {mean_approx_kl:.4f} exceeded target_kl {target_kl} "
                "- stopping this iteration's policy update early"
            )
            break


def _value_step(idx, buffer, step_returns, env, value_fn, value_optimizer):
    """
    One gradient step of the value function's MSE loss (Eq. A4) against the discounted
    return-to-go for each environment step (from compute_advantages), evaluated at THAT step's
    own belief state - not the episode's fixed initial state. `idx` indexes into the buffer's
    terminal (k=0) rows - there's exactly one of those per environment step.
    """
    device = env.meanvarmarker_map.device

    env_step_idx = buffer.env_step[idx]
    target_return = step_returns[env_step_idx].to(device)  # [B]

    meanvarmarker_map = buffer.step_meanvarmarker_map[env_step_idx].to(device)
    current_position = buffer.step_current_position[env_step_idx].to(device)
    initial_heading_velocity = buffer.step_heading_velocity[env_step_idx].to(device)

    predicted_value = value_fn(
        meanvarmarker_map, current_position, initial_heading_velocity
    ).squeeze(-1)  # [B, 1] -> [B]

    value_loss = torch.nn.functional.mse_loss(predicted_value, target_return)

    value_optimizer.zero_grad()
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(value_fn.parameters(), max_norm=MAX_GRAD_NORM)
    value_optimizer.step()
    return value_loss.item()


def update_value(buffer, step_returns, env, value_fn, value_optimizer,
                  num_epochs=NUM_VALUE_EPOCHS, minibatch_size=MINIBATCH_SIZE):
    """
    Train the value function towards each environment step's discounted return-to-go
    (step_returns, from compute_advantages). Only ever sees the buffer's terminal (k=0) rows -
    one (state, return) pair per environment step - since every non-terminal row shares that
    step's state and has reward 0 by construction.
    """
    terminal_idx = (buffer.k == 0).nonzero(as_tuple=True)[0]

    for epoch in range(num_epochs):
        perm = terminal_idx[torch.randperm(terminal_idx.shape[0])]
        for start in range(0, perm.shape[0], minibatch_size):
            idx = perm[start:start + minibatch_size]
            loss = _value_step(idx, buffer, step_returns, env, value_fn, value_optimizer)
        print(f"epoch {epoch}: value_loss={loss:.6f}")


def plot_iteration_trajectory(position_history, env, iteration):
    """Saves a top-down (2D, over the ground-truth map) + 3D view of one representative pooled
    episode's flown path to IMAGE_DIR, so training instability can be diagnosed visually - is the
    drone flying somewhere sensible, orbiting in place, leaving the map, etc. - instead of only
    from aggregate RMSE/reward numbers."""
    positions = np.array(position_history)
    path_x, path_y, path_z = positions[:, 0], positions[:, 1], positions[:, 2]
    grid_shape = (len(env.ys), len(env.xs))

    fig = plt.figure(figsize=(12, 6))

    ax_2d = fig.add_subplot(1, 2, 1)
    ax_2d.imshow(
        env.true_map_flat.reshape(grid_shape),
        origin="lower",
        extent=[env.xmin, env.xmax, env.ymin, env.ymax],
        cmap="Greys",
        alpha=0.6,
    )
    ax_2d.plot(path_x, path_y, "-o", color="tab:blue", markersize=2, linewidth=1)
    ax_2d.plot(path_x[0], path_y[0], "go", markersize=8, label="start")
    ax_2d.plot(path_x[-1], path_y[-1], "ro", markersize=8, label="end")
    ax_2d.set_xlabel("x")
    ax_2d.set_ylabel("y")
    ax_2d.set_title(f"Top-down view - iteration {iteration}")
    ax_2d.legend()

    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax_3d.plot(path_x, path_y, path_z, "-", color="tab:blue", linewidth=1)
    ax_3d.scatter([path_x[0]], [path_y[0]], [path_z[0]], color="green", s=40, label="start")
    ax_3d.scatter([path_x[-1]], [path_y[-1]], [path_z[-1]], color="red", s=40, label="end")
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    ax_3d.set_title(f"3D view - iteration {iteration}")
    ax_3d.legend()

    fig.tight_layout()
    fig.savefig(IMAGE_DIR / f"iteration_{iteration:04d}.png", dpi=120)
    plt.close(fig)


def _save_checkpoint(model, value_fn, iteration):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "value_state_dict": value_fn.state_dict(),
            "iteration": iteration,
        },
        CHECKPOINT_DIR / f"dppo_iteration_{iteration:04d}.pth",
    )


def train(env, num_iterations=NUM_DPPO_ITERATIONS, num_episodes=NUM_EPISODES_PER_ITERATION):
    """
    Outer DPPO loop, MPI-parallel across episodes (same conventions as
    DataCollector_beamsearch.py - falls back to a single serial process if mpi4py isn't
    installed or the script isn't launched under mpirun/srun/mpiexec). Each iteration:
      1. collects `num_episodes` FRESH, independent episodes under the CURRENT policy,
         distributed round-robin across MPI ranks, and pools them into one buffer
         (collect_pooled_episodes),
      2. (rank 0 only) updates the policy off the pooled advantage, then updates the value
         function off the pooled return-to-go,
      3. broadcasts the just-updated policy/value weights back to every rank, so next
         iteration's rollout collection on non-zero ranks uses the current policy too.

    Pooling multiple independent episodes - rather than reusing one noisy episode for many PPO
    epochs - is what actually reduces the advantage-estimate variance; a single episode's reward
    is dominated by sampling noise (stochastic eta=1 denoising + sensor measurement noise), which
    previously showed up as policy_loss alternating sign almost every iteration with no net
    reward improvement.

    diffusion.model, value_fn, and both optimizers (with their Adam momentum) persist across
    iterations - only the pooled rollout buffer itself is discarded and replaced every
    iteration, so later iterations build on earlier ones instead of restarting from scratch.
    """
    comm, mpi_rank, mpi_size = get_mpi_context()
    if detect_mpi_launch_issue(os.environ, mpi_size):
        raise RuntimeError(
            "SLURM_NTASKS is greater than 1, but mpi4py sees MPI size 1. This means the script "
            "is being launched as repeated serial jobs, not one MPI job. Use an MPI-aware launch "
            "such as `srun --mpi=pmix python -m Diffusion.DPPOTRAINING` or "
            "`mpiexec -n $SLURM_NTASKS python -m Diffusion.DPPOTRAINING`."
        )
    if mpi_rank == 0:
        print(f"Running DPPO training with MPI size={mpi_size}, episodes/iteration={num_episodes}")
        if VISUALIZE:
            IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model = _load_checkpoint(DEFAULT_CHECKPOINT, diffusion.device)  # same file, every rank

    value_fn = ValueFunction().to(env.meanvarmarker_map.device)
    # value_fn's random init would otherwise differ per rank (each is a separate process with
    # its own RNG state) - broadcast rank 0's init so every rank starts identical.
    _broadcast_module_state(value_fn, comm, mpi_rank)

    policy_optimizer = torch.optim.AdamW(diffusion.model.parameters(), lr=4e-5)
    value_optimizer = torch.optim.AdamW(value_fn.parameters(), lr=VALUE_LR)

    for iteration in range(num_iterations):
        buffer, discounted_advantages, step_returns, iter_trajectory = collect_pooled_episodes(
            env, value_fn, num_episodes, comm, mpi_rank, mpi_size,
            base_seed=iteration * num_episodes,
            record_trajectory=VISUALIZE,
        )

        if mpi_rank == 0:
            terminal = buffer.k == 0
            mean_reward = buffer.reward[terminal].mean().item()
            mean_global_rmse = buffer.global_rmse[terminal].mean().item()
            mean_occupied_rmse = buffer.occupied_rmse[terminal].mean().item()
            print(
                f"=== iteration {iteration}: mean_reward={mean_reward:.4f}, "
                f"post-flight global_rmse={mean_global_rmse:.4f} "
                f"(baseline {env.baseline_global_rmse:.4f}), "
                f"occupied_rmse={mean_occupied_rmse:.4f} "
                f"(baseline {env.baseline_occupied_rmse:.4f}), "
                f"episodes_pooled={num_episodes} ==="
            )
            if VISUALIZE:
                plot_iteration_trajectory(iter_trajectory, env, iteration)

            update_policy(
                buffer, discounted_advantages, env, policy_optimizer,
                num_epochs=NUM_PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE, clip_epsilon=CLIP_EPSILON,
                target_kl=TARGET_KL,
            )
            update_value(
                buffer, step_returns, env, value_fn, value_optimizer,
                num_epochs=NUM_VALUE_EPOCHS, minibatch_size=MINIBATCH_SIZE,
            )
            _save_checkpoint(diffusion.model, value_fn, iteration)

        # Every rank's collect_rollouts call next iteration needs the just-updated weights.
        _broadcast_module_state(diffusion.model, comm, mpi_rank)
        _broadcast_module_state(value_fn, comm, mpi_rank)

    return model, value_fn


if __name__ == "__main__":
    env = build_environment()
    _comm, _mpi_rank, _ = get_mpi_context()
    if _mpi_rank == 0:
        print(f"Fixed DPPO environment: map_id={env.map_id}, start=({env.cx0}, {env.cy0}, {env.cz0})")
        print(f"Baseline global RMSE (no flight): {env.baseline_global_rmse:.4f}")
        print(f"Baseline occupied RMSE (no flight): {env.baseline_occupied_rmse:.4f}")

    train(env, num_iterations=NUM_DPPO_ITERATIONS, num_episodes=NUM_EPISODES_PER_ITERATION)
