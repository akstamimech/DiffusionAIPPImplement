"""
DPPOTRAINING_batched.py

GPU-batched variant of DPPOTRAINING_overnight.py - same DPPO algorithm, same validated
hyperparameters (ENV_HORIZON=30, NUM_EPISODES_PER_ITERATION=16, MINIBATCH_SIZE=512,
SIGMA_PROB_MIN=0.3, lower actor LR), same overnight-run hardening (checkpoint pruning, a
dedicated best-checkpoint, a high NUM_DPPO_ITERATIONS relying on SLURM's own wall-clock limit) -
but replaces MPI-based parallelism (N separate processes, each running one episode with
batch-size-1 model calls) with GPU batching (ONE process, running all N episodes' neural-network
calls as a single batch-N call per denoising step).

WHY: a GPU does its best work on one large batched operation, not many small serial ones. With
MPI, if N processes share one physical GPU (the typical case - one GPU per node, not one per
rank), the GPU driver just time-slices their N separate batch-1 calls one after another - you pay
N times the fixed per-call overhead (kernel launch, memory transfer) for the same total work.
Batching all N episodes' current denoising step into one batch-N tensor and calling the model
ONCE gets the same total computation done with that overhead paid only once. (Analogy: N taxis
each carrying 1 passenger down the same single-lane road vs. 1 bus carrying all N at once.)

WHAT CAN'T BE GPU-BATCHED, BUT CAN BE CPU-PARALLELIZED: the environment simulation itself
(Kalman-filter belief updates, drone dynamics) is plain NumPy/CPU code with per-episode Python
state (mu, P, position) - there is no GPU-vectorized version of this available (unlike the DPPO
paper's IsaacGym-based tasks, which get this for free from their simulator). Profiling found this
part - not the GPU-batched model calls - is actually the dominant cost (measured: ~500 sec/iteration
of sequential Kalman-filter calls vs. an order of magnitude less GPU compute), so
collect_rollouts_batched dispatches each replan's N independent per-episode chunks to a
persistent multiprocessing pool (see dppo_rollout_worker.py, NUM_ROLLOUT_WORKERS) instead of
running them in a serial Python for-loop in this process. Measured ~3x wall-clock reduction for
that part (not a naive N-fold speedup - this workload's dominant cost updates a full dense
2601x2601 covariance matrix per measurement, which is memory-bandwidth-bound, not purely
core-count-bound). Only the neural network's forward pass is GPU-batched; the environment
stepping is CPU-batched across a process pool instead.

REQUIRES BOTH AN ACTUAL GPU (for the batched model calls) AND SEVERAL CPU CORES (for
NUM_ROLLOUT_WORKERS, the environment-stepping pool) TO BE WORTHWHILE. If you request a GPU but
only a handful of CPU cores, the environment-stepping pool oversubscribes those cores and you
lose most of the benefit - see NUM_ROLLOUT_WORKERS' comment; size --cpus-per-task on whatever
launches this file to match it.

Same Steps 1-6 and 8-10 as DPPOTRAINING_overnight.py (see that file's docstring for the full
list) except Step 7, which is replaced entirely:

    Step 7 (collect_rollouts_batched / collect_pooled_episodes_batched, replaces MPI): runs all
        NUM_EPISODES_PER_ITERATION episodes SIMULTANEOUSLY in one process, in lockstep - at each
        of the ENV_HORIZON replans, all episodes' belief-conditioning tensors are stacked into
        one batch-N tensor, and at each denoising step within that replan, one batch-N call to
        ddpo_ddim_sample_timestep replaces what used to be N separate batch-1 calls spread across
        N MPI processes/ranks. After the batched denoising finishes, each episode's environment
        step (Kalman update + dynamics, per-episode NumPy state) is dispatched to a persistent
        multiprocessing pool (dppo_rollout_worker.py) instead of running in a serial per-episode
        Python for-loop in this process, since that part can't be batched onto the GPU but is
        independent across episodes and was measured to be the actual dominant per-iteration
        cost. Returns a list of N per-episode RolloutBuffers, fed into the SAME
        compute_advantages / _concat_rollout_buffers / advantage-normalization pipeline as
        before, completely unchanged - only the collection mechanism changed, nothing downstream
        of it (update_policy, update_value, checkpointing, visualization are all untouched).

    No MPI at all: get_mpi_context / assigned_tasks_for_rank / _broadcast_module_state / mpi4py
        are gone entirely - there is only ever one process, so there is nothing to distribute or
        broadcast. train() no longer has `if mpi_rank == 0:` guards - everything just runs.

    Reproducibility note: the old per-episode `torch.manual_seed(base_seed + episode_idx)` scheme
    (one independent seed per episode) is replaced by a single `torch.manual_seed(base_seed)`
    call per iteration, since all N episodes' noise is now drawn from ONE shared batched
    `torch.randn((N, ...))` call rather than N separate calls. Each episode still gets genuinely
    different noise (the N rows of that batched draw are independent), but a specific episode
    index is no longer reproducible in isolation the old way. The per-episode sensor-noise RNGs
    (numpy, not torch) are still seeded independently per episode, unchanged.

HOW TO RUN:
    Single process, no MPI, needs a GPU to be worth using (see above):
        python -m Diffusion.DPPOTRAINING_batched
    or under SLURM with a single task - see the carryover doc / conversation for the accompanying
    sbatch changes (--ntasks=1, one GPU, no more --mpi=pmix/mpiexec, no more OpenMPI module
    loads).
"""

import multiprocessing as mp
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the 3D projection

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from gaussianprocesstraining import initialize_gp, importance_filter
from evalmetrics import compute_reconstruction_rmse
from Diffusionplanner_singlemap import build_true_map_flat

from DPPOvaluefunction import ValueFunction
from threeDSparseTransDiffusion import NoisePredictor
import threeDSparseTransDiffusion as diffusion
import dppo_rollout_worker


# ---------------------------------------------------------------------------
# Step 1: the environment pool (one map per training/validation environment)
# ---------------------------------------------------------------------------

SENSORNOISE_SEED = 123
MAPTYPE = os.environ.get("MAPTYPE", "grf")
# NOTE: REPO_DIR is Diffusion/ (this file's own directory), but the map CSVs live one level up,
# at scripts/csv - REPO_DIR/"csv" would silently point at a directory that doesn't exist.
CSV_PATH = SCRIPT_DIR / "csv"
UTILITY_THRESHOLD = 0.5  # must match Diffusionplanner_singlemap.py's utility_threshold
SAMPLESTEP = 2.0

# T_a in the paper's terms: measurement updates executed per environment step before replanning.
# Must match Diffusionplanner_singlemap.py's execution_chunk (now 20) - see the §5.2-class bug
# this class of drift caused before: training replanning at a different cadence than deployment.
EXECUTION_CHUNK = 40
# T in the paper's terms: number of execute-then-replan environment steps per episode.
# NOTE: this is now numerically equal to diffusion.T (=30, the denoising-step count) - a
# coincidence of matching the batch-size scaling validated in the user's 10-iteration trial, not
# a merge of the two axes. They remain functionally independent: this is the OUTER (environment)
# loop count in collect_rollouts_batched, diffusion.T is the INNER per-replan denoising-step count
# in ddpo_ddim_sample_timestep. Just don't assume "step 30" in a log line means the same thing in
# both contexts.
ENV_HORIZON = 8

# K' in the paper's terms (Section 4.3, "Fine-tune only the last few denoising steps"): the
# full K=diffusion.T=20-step denoising chain still runs every replan (needed to get a coherent
# sample at all), but only the transitions from the LAST NUM_FINE_TUNE_STEPS steps (closest to
# k=0, the terminal/least-noisy step) get stored in the buffer and trained on - the early,
# noisiest steps are treated as frozen/untouched, matching the paper's theta vs theta_FT split
# (though we don't maintain a literal separate frozen copy of the weights, just skip recording
# and gradient-updating on those early steps).
NUM_FINE_TUNE_STEPS = 7

# Reward: DENSE, BELIEF-MASKED OCCUPIED VARIANCE SEEKING, map-agnostic. Every environment step is
# scored by the posterior variance REMAINING after its chunk, summed only over cells the CURRENT
# belief (not the ground truth) still flags as important via importance_filter, as a concave
# fraction of the GP prior's total variance:
#     mask_t = importance_filter(mu_after, P_after, beta=UCB_BETA, threshold=UTILITY_THRESHOLD) > 0
#     reward_t = -VARIANCE_WEIGHT * (sum(diag(P_after)[mask_t]) / baseline_total_variance)^CONCAVITY_FACTOR
# Goal, stated plainly: reduce posterior variance, restricted to wherever the belief itself still
# thinks matters, as much (and, via the concave per-step level, as urgently once nearly converged)
# as possible.
#   - Belief-masked, not ground-truth-masked: importance_filter is exactly what the BC-cloned
#     expert's own trajectory objective already optimizes against (masked_expected_variance_
#     reduction_from_sensor), so DPPO now fine-tunes the SAME objective the policy was cloned to
#     pursue, rather than a different (whole-grid) one. It also only uses information the policy's
#     own belief actually has - never the true map - unlike the ground-truth occupied_mask used for
#     logging/eval below.
#   - Denominator stays baseline_total_variance (trace(P0), fixed at mission start, identical
#     across every map): under the current GRF/UCB prior mu0 = UTILITY_THRESHOLD + 0.1 already
#     exceeds the threshold everywhere before beta*sigma is even added, so the INITIAL belief mask
#     is provably the whole grid on every map - trace(P0[initial_mask]) == trace(P0) - so this
#     fixed, map-agnostic denominator is already exactly right; only the numerator needs to be
#     masked. Summed (via GAE) across a mission's replans, this is a concave reshaping of the same
#     occupied-variance-AUC metric already used for evaluation throughout the thesis.
#   - Dense, not terminal: rolled back from the terminal-only reward deliberately - a per-step
#     signal is easier for the critic/PPO to assign credit against than one propagated terminal
#     scalar, which is the likelier cause of slow learning.
#   - CENTER-BIAS NOTE: the previous pure-whole-grid version of this reward had a measured ~2x
#     per-measurement bias toward central measurements (an interior FOV footprint / kernel
#     spillover is never clipped by the map boundary). Masking by current importance should reduce
#     this directly: once a central cell's posterior mean crosses the threshold it drops out of the
#     mask entirely, so re-measuring an already-resolved center stops contributing to the reward at
#     all (not just diminishing marginally) - still worth watching via plot_iteration_trajectory.
# RMSE stays out of the reward: it depends on true-map content and the noise draw, not just where
# you flew - noisier than variance for the same flight path (the Kalman covariance update depends
# only on measurement geometry).
VARIANCE_WEIGHT = 1.0  # single-term reward + advantage normalization => this is scale-only
CONCAVITY_FACTOR = 0.5
# beta for the reward's own importance_filter call - same conventional value (one std-dev
# confidence bound) used by the planner scripts' own BETA default.
UCB_BETA = 1.0

NUM_PPO_EPOCHS = 7
# Scales with total pooled rows per iteration (episodes x horizon), keeping the same NUMBER of
# minibatches/gradient-steps per epoch as smaller configurations tested earlier in this file's
# history (64 at 6 episodes x 10 steps -> 96 at 9x10 -> 512 at 16 episodes x 30 steps: terminal
# rows still fit one full-batch minibatch, nonterminal still splits into ~9 minibatches/epoch).
# DPPO's gradient averages over environment steps x denoising steps jointly, so minibatch size
# should track how much pooled data is actually available per iteration (paper Appendix B,
# "Large batch size"), not stay fixed while episode count/horizon change.
MINIBATCH_SIZE = 1024
# Paper (Table A7/A9) tunes epsilon per task in {0.1, 0.01, 0.001}, picking the highest value
# that keeps the batch's clipping fraction (see _ppo_step's returned clip_fraction) in the
# 10-20% range. 0.1 measured 60-100% clip_fraction in practice (slurm-10340607.out) - moved to
# the paper's actual DPPO value of 0.01. Watch the printed clip_fraction below; drop to 0.001
# next if it's still above ~20%.
CLIP_EPSILON = 0.03

# Paper (Appendix B): "the batch updates in an iteration terminate when the KL divergence
# between pi_theta and pi_theta_old reaches 1, although in practice we find this never
# happens" (under their tuned hyperparameters). Ours isn't necessarily tuned enough yet for that
# to hold, so this is a live circuit breaker on update_policy's epoch loop, not decoration - see
# _ppo_step's returned approx_kl.
TARGET_KL = 1.0

# Second circuit breaker on the same epoch loop, keyed to clipping instead of KL: stop this
# iteration's remaining PPO epochs once an epoch's mean clip_fraction exceeds this. Rationale:
# clip_fraction has chronically climbed within every iteration (measured ~0.1-0.2 at epoch 0 ->
# ~0.45-0.55 by epoch 6, across entire 300-iteration runs, surviving both a CLIP_EPSILON 3x raise
# and an actor-LR 4x cut AND a full reward restructure - so it's a data-staleness property of
# reusing one rollout for NUM_PPO_EPOCHS, not a reward or threshold miscalibration). A clipped
# row contributes zero gradient, so an epoch at 0.45 clip_fraction updates on barely half its
# batch - and a biased half at that (whichever rows the policy hasn't already moved past the
# clip window on). Threshold chosen just above the paper's own 10-20% healthy range; with the
# measured climb shape this typically stops around epoch 3-4 instead of running all 7. Adaptive
# by construction - an iteration whose policy genuinely barely moves still gets all 7 epochs.
CLIP_FRACTION_EARLY_STOP = 0.30

# Standard PPO safety net (e.g. OpenAI baselines' PPO2 default) independent of clip_epsilon -
# clip_epsilon only stops the LOSS from rewarding ratio moving past the threshold, it doesn't
# bound how large a single minibatch's gradient can be. A handful of outlier-advantage rows
# could still yank the shared weights hard even after advantage normalization (see
# collect_pooled_episodes_batched) reduces how often that happens.
MAX_GRAD_NORM = 0.5

NUM_VALUE_EPOCHS = 7
# Reverted 1e-4 -> 1e-3. The 1e-4 (AID-matching) value LR was too slow HERE: the value batch is
# only NUM_EPISODES_PER_ITERATION*ENV_HORIZON = 32*15 = 480 terminal rows < MINIBATCH_SIZE, so
# each iteration does just ONE full-batch step x NUM_VALUE_EPOCHS(=7) = 7 tiny gradient steps at
# the value LR. At 1e-4 that critic cannot track the drifting returns - directly observed as
# value_loss climbing ~1.0 -> ~3.0 over iterations while the policy slowly degraded off the BC
# prior (slurm_10436057). 1e-3 is the value LR the earlier runs that improved on val used.
VALUE_LR = 1e-3

# GAE across environment steps (paper Eq. A2): gamma_ENV discounts reward across steps t (NOT
# denoising steps k - that's still denoising_gamma, passed separately to compute_advantages),
# lambda trades off bias/variance in the advantage estimate the usual GAE way.
# 0.99 -> 0.999 (AID's gamma) alongside the terminal-only reward: the ONLY reward now sits at
# the episode's last step, and early steps only ever see it discounted by gamma^(T-1-t) -
# 0.99^14 =~ 0.87 would shave 13% off the signal reaching step 0 for no reason, 0.999^14 =~ 0.99
# keeps it essentially undecayed across ENV_HORIZON=15.
ENV_GAMMA = 0.999
GAE_LAMBDA = 0.95

# Number of DPPO training iterations. Set far higher than one night can plausibly finish -
# per-iteration checkpointing (Step 10) means there's no downside to this being "too high": the
# run just gets stopped by SLURM's own wall-clock limit whenever that happens, with the latest
# and best checkpoints already safely on disk. Guessing a precise iteration count that fits in
# one night isn't worth the risk of guessing low and leaving compute unused.
NUM_DPPO_ITERATIONS = 300

# N in the paper's "N environments in parallel" (Algorithm 1): independent episodes collected
# and pooled together before each policy/value update, now all run SIMULTANEOUSLY in one
# process's batched model calls (collect_rollouts_batched) instead of being spread across N MPI
# ranks. A single episode's reward is dominated by sampling noise (stochastic eta=1 denoising +
# sensor measurement noise) - pooling several independent episodes and averaging their advantage
# estimates is what actually reduces that noise, rather than reusing one noisy episode for many
# PPO epochs (NUM_PPO_EPOCHS) and overfitting to whatever that one episode happened to look like.
NUM_EPISODES_PER_ITERATION = 32

# Number of worker processes in the persistent pool collect_rollouts_batched dispatches each
# replan's per-episode Kalman-filter/dynamics chunk to (see dppo_rollout_worker.py) - the part of
# the rollout that can't be GPU-batched (module docstring) but IS embarrassingly parallel across
# independent episodes. Measured on a 12-core dev machine: ~37s single-threaded-sequential drops
# to ~12s with 16 workers (a real ~3x, not a naive 16x - this workload's dominant cost updates a
# full dense 2601x2601 covariance matrix per measurement, which is memory-bandwidth-bound, not
# purely core-count-bound). Defaults to NUM_EPISODES_PER_ITERATION (one worker per episode slot);
# lower this to match your actual allocated CPU count if requesting fewer cores than episodes.
NUM_ROLLOUT_WORKERS = NUM_EPISODES_PER_ITERATION

# Map pool DPPO trains against, for generalization rather than overfitting to one fixed map.
# Every training rollout draws its map from TRAIN_MAP_IDS (see _sample_episode_envs); VAL_MAP_IDS
# is held out entirely from training rollouts and used only by evaluate_generalization for an
# honest unseen-map check. Matches the thesis's own GRF convention: 50 training maps (0-49) with
# maps 51-60 held out for evaluation across all benchmarks (map 50 is left out of both pools to
# match compare_multimap_all_planners.py's DEFAULT_MAPS boundary).
TRAIN_MAP_IDS = list(range(0, 50))
VAL_MAP_IDS = list(range(51, 61))
# Starting heading/velocity for every episode - a raw (unnormalized) world-frame direction
# vector, not literally unit-length; normalize_xyz_displacement handles scaling it into the
# model's conditioning space. NOTE: with the warmup below this is now only a placeholder for the
# pre-warmup state - the heading the FIRST replan actually sees is the last warmup step, computed
# by warmup_chunk (see collect_rollouts_batched), matching deployment.
INITIAL_HEADING_VELOCITY = (1.0, 1.0, 1.0)

# Initial conditions matched to Diffusionplanner_singlemap.py exactly. Deployment starts every
# flight at the map corner (cx,cy,cz = 4,4,INIT_ALTITUDE) and, for its first WARMUP_STEPS
# timesteps (its `ts <= 1` branch), flies toward world (WARMUP_GOAL_XY, WARMUP_GOAL_XY,
# INIT_ALTITUDE) taking one measurement per step BEFORE the first diffusion replan. Replicating
# this means the first replan sees the same belief (prior + WARMUP_STEPS measurements), position,
# and heading (last warmup step) as deployment, rather than the first replan starting from the raw
# prior at the corner with the arbitrary INITIAL_HEADING_VELOCITY above.
START_XY = 4.0
INIT_ALTITUDE = diffusion.Z_MIN  # 10.0, == Diffusionplanner_singlemap.INIT_ALTITUDE
WARMUP_GOAL_XY = 80.0
WARMUP_STEPS = 2

# How often (in training iterations) to run evaluate_generalization against VAL_MAP_IDS. Every
# iteration would double rollout-collection cost for no real benefit - the whole point is
# checking whether the policy is generalizing over the medium term, not a per-iteration signal.
VAL_EVAL_EVERY = 10

# Variance floor used when evaluating the Gaussian likelihood for the PPO ratio (NOT the
# variance used for actual sampling noise - that has its own, separate floor below). Paper's own
# minimum is 0.1; raised to 0.3 here because approx_kl was blowing well past TARGET_KL within
# 1-2 epochs at 0.1, even after advantage normalization and gradient clipping - since the
# likelihood divides by this value, raising it from 0.1 to 0.3 cuts that sensitivity by
# (0.3/0.1)^2 = 9x. This was the main lever that brought approx_kl comfortably under TARGET_KL
# in the user's 10-iteration trial.
SIGMA_PROB_MIN = 0.2
# Floor on the std of the noise ACTUALLY ADDED during rollout sampling (DPPO paper section 4.3's
# min_sampling_denoising_std; AID ships 0.05). Distinct from SIGMA_PROB_MIN, which only affects
# the likelihood used for the PPO ratio: without this floor, the schedule-derived posterior
# variance decays toward zero at the late (least-noisy) denoising steps - which are exactly the
# NUM_FINE_TUNE_STEPS being trained - so on-policy exploration collapses precisely where
# fine-tuning happens. The terminal (k=0) output stays deterministic (the executed action is
# still the predicted mean); this floors only the intermediate transitions' noise.
SIGMA_SAMPLE_MIN = 0.05
# Per-dimension clamp applied to log-probs BEFORE summing over the 3x8 action dims, on BOTH the
# stored (old) side at collection and the recomputed (new) side in _ppo_step - AID clamps both
# sides to exactly this range. Bounds how much any single >3-sigma outlier dimension can
# contribute to a row's summed log-prob, and therefore to its PPO ratio; also zeroes those
# outlier dims' gradients. (At sigma=0.3 the per-dim log-prob maxes at ~+0.29, so the upper
# clamp is essentially inactive - the guard that matters is the lower one.)
LOGPROB_CLAMP_MIN = -5.0
LOGPROB_CLAMP_MAX = 2.0
# Iterations at the start of training during which ONLY the value function updates (actor
# frozen) - AID's n_critic_warmup_itr=2. The critic starts randomly initialized, so iteration
# 0-1 advantages are noise; without a warmup the very first policy updates consume that noise
# as if it were signal. (The reward is now dense per-step - see VARIANCE_WEIGHT - so the critic
# is less load-bearing for credit assignment than under a terminal reward, but the warmup is
# still cheap insurance against garbage first-iteration advantages.)
N_CRITIC_WARMUP_ITERATIONS = 2

DEFAULT_CHECKPOINT = (
    SCRIPT_DIR / "checkpoints" / "current_best.pth"
)

# Toggle: dump a top-down + 3D trajectory plot of one representative pooled episode after every
# iteration (see plot_iteration_trajectory) - lets training instability be diagnosed visually
# (orbiting in place, leaving the map, etc.) instead of only from aggregate RMSE/reward numbers.
VISUALIZE = True
# Plotting (matplotlib figure render + savefig) and the routine per-iteration checkpoint (torch.save
# of model+value_fn state dicts, often to a networked/shared HPC filesystem) both add real fixed
# overhead if paid every single iteration - PLOT_EVERY/CHECKPOINT_EVERY make both periodic instead.
# _save_best_checkpoint is NOT gated by CHECKPOINT_EVERY - it should always fire the moment a new
# best is found, independent of this cadence.
PLOT_EVERY = 5
CHECKPOINT_EVERY = 5
# Distinct directory names from DPPOTRAINING.py's/DPPOTRAINING_overnight.py's so this file's
# outputs never collide with either of theirs if run from the same working directory.
IMAGE_DIR = SCRIPT_DIR / "rl_correction_images_batched"

# Every iteration's fine-tuned policy/value weights, so a specific iteration can be reloaded
# later (e.g. to inspect why training regressed past it) - distinct from the pretrained BC
# checkpoint DEFAULT_CHECKPOINT loads from.
CHECKPOINT_DIR = SCRIPT_DIR / "dppo_checkpoints_batched"
# Only the last CHECKPOINT_KEEP_LAST numbered per-iteration checkpoints are kept on disk (see
# _prune_old_checkpoints) - at NUM_DPPO_ITERATIONS=300, keeping every single one would be ~300 x
# the per-checkpoint size (model + value_fn state dicts) on scratch storage for an unattended
# run nobody's watching. The single best iteration is preserved separately and indefinitely in
# dppo_best.pth (_save_best_checkpoint), independent of this pruning window.
CHECKPOINT_KEEP_LAST = 10


@dataclass
class SingleEnvironment:
    """Everything needed to sample from, and score trajectories against, one
    (map, current_position, heading) scenario. One of these is built per map_id in the
    training/validation pool (see build_environment_pool), not a single global instance."""

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
    # Total posterior variance (trace of P0) before any measurements are taken - the baseline
    # compute_step_reward's global variance term is normalized against.
    baseline_total_variance: float
    # Same, but restricted to occupied_mask (true_map < utility_threshold) - the baseline
    # compute_step_reward's occupied variance term is normalized against. Floored at a small
    # epsilon (see build_environment) so a map with zero occupied cells can't produce a literal
    # division by zero (float division by zero raises, it doesn't just become NaN) and crash an
    # unattended run mid-iteration.
    baseline_occupied_variance: float


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


def build_environment(map_id, gp_prior=None, cx0=None, cy0=None, cz0=None, heading_velocity=INITIAL_HEADING_VELOCITY):
    """Step 1: resolve one (map, starting position, heading) scenario - called once per map_id in
    build_environment_pool to build the whole training/validation pool, rather than a single fixed
    environment. The initial belief state comes directly from initialize_gp()'s raw GP prior (zero
    mean, kernel-induced covariance) - the same starting point Diffusionplanner_singlemap.py's
    receding-horizon loop uses - rather than a diffusion-dataset row, since map_id/starting pose
    are chosen directly instead of looked up from a precomputed condition_index.

    gp_prior: optional pre-computed initialize_gp() tuple, shared across every map in a pool (see
    build_environment_pool) - initialize_gp() takes no map_id, so its output is identical for every
    map, and recomputing it per map wastes both the (cheap) GP setup call and, more importantly,
    the (~54MB per map) covariance-matrix allocation. Defaults to None so build_environment still
    works standalone (computes its own prior) if ever called outside a pool."""

    if gp_prior is None:
        gp_prior = initialize_gp()
    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, grid_step = gp_prior
    # initialize_gp()'s own mean is raw GP-prior zero; Diffusionplanner_singlemap.py overrides it
    # to utility_threshold + 0.1 (UCB/grf: cells start optimistically-unimportant) before its first
    # measurement - match that override here so training and deployment start from the same belief mean.
    mean = np.full_like(mean, UTILITY_THRESHOLD + 0.1)

    # Corner start matching Diffusionplanner_singlemap.py (cx,cy,cz = 4,4,INIT_ALTITUDE), not the
    # old map-center start - see START_XY/INIT_ALTITUDE. The warmup in collect_rollouts_batched
    # then advances from here toward WARMUP_GOAL_XY before the first replan.
    cx0 = START_XY if cx0 is None else cx0
    cy0 = START_XY if cy0 is None else cy0
    cz0 = INIT_ALTITUDE if cz0 is None else cz0

    true_map_flat, pts = _load_true_map(map_id, X_test, grid_step)

    # mean/cov come straight from the (possibly pool-shared) GP prior - reference them directly
    # rather than copying per environment. Safe: nothing ever mutates env.mu0/env.P0 in place:
    # every consumer copies first (collect_rollouts_batched's
    # mus[i]=envs[i].mu0.copy() / Ps[i]=envs[i].P0.copy()) - so every environment built from the
    # same gp_prior can safely share the identical underlying arrays instead of each holding its
    # own redundant ~54MB (2601x2601) copy of an otherwise map-independent covariance matrix.
    mu0 = mean
    P0 = cov

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
    baseline_total_variance = float(np.sum(np.diag(P0)))
    occupied_mask = baseline_metrics["occupied_mask"].ravel()
    baseline_occupied_variance = max(float(np.sum(np.diag(P0)[occupied_mask])), 1e-6)

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
        baseline_total_variance=baseline_total_variance,
        baseline_occupied_variance=baseline_occupied_variance,
    )


def build_environment_pool(map_ids):
    """Builds one SingleEnvironment per map_id upfront, for multi-map training/evaluation.
    initialize_gp() takes no map_id (the GP prior/grid is identical for every map - see
    build_environment) so every environment in the pool only differs in its own true_map_flat and
    the baseline RMSE/variance derived from it; precomputing the whole pool once here (rather than
    reloading CSVs every iteration) is cheap and keeps collect_rollouts_batched simple.

    Computes the GP prior ONCE and shares it across every environment in the pool, rather than
    letting each build_environment call it independently - avoids both len(map_ids)-1 redundant
    GP setup calls and, more importantly, len(map_ids)-1 redundant ~54MB (2601x2601) covariance
    matrix allocations for data that's identical across every map (e.g. 65 maps: ~3.5GB of
    duplicate arrays down to one ~54MB copy)."""
    gp_prior = initialize_gp()
    return {map_id: build_environment(map_id=map_id, gp_prior=gp_prior) for map_id in map_ids}


# ---------------------------------------------------------------------------
# Step 2: execute one environment step's chunk and score it
# ---------------------------------------------------------------------------
# The per-episode chunk rollout (build_spline_trajectory_3d + waypoint_3d/dynamics_3d +
# apply_measurement_update_3d) used to live here as _rollout_trajectory, called in a serial
# per-episode Python loop inside collect_rollouts_batched. It's now dppo_rollout_worker's
# rollout_chunk (verified byte-identical - see that module's docstring for why it's a separate,
# CUDA-free reimplementation rather than importing this file's own dependencies), dispatched to a
# persistent multiprocessing pool so the 16 independent episodes' chunks - the one part of the
# rollout that can't be GPU-batched - run across CPU cores instead of one at a time.


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


def compute_step_reward(mu_before, mu_after, P_before, P_after, env, env_step):
    """Step 2: DENSE belief-masked occupied-variance reward (see VARIANCE_WEIGHT's comment block
    above for the full rationale). EVERY environment step is scored by the posterior variance
    remaining after its chunk, summed only over cells the CURRENT belief still flags as important:

        mask = importance_filter(mu_after, P_after, beta=UCB_BETA, threshold=UTILITY_THRESHOLD) > 0
        -VARIANCE_WEIGHT * (sum(diag(P_after)[mask]) / baseline_total_variance)^CONCAVITY_FACTOR

    This is belief-masked, not ground-truth-masked: it's exactly the same importance criterion the
    BC-cloned expert's own trajectory objective already optimizes against, using only information
    the policy's own belief has (never the true map). baseline_total_variance (trace(P0), fixed at
    mission start) stays the denominator unchanged from the whole-grid version - under the current
    GRF/UCB prior the initial belief mask is provably the whole grid on every map, so this
    map-agnostic fixed denominator is already exactly right; only the numerator is masked.

    mu_before/P_before are unused (the reward scores the post-chunk level, not the drop) but kept
    in the signature so the call site doesn't have to change shape.

    Returns (reward, global_rmse, occupied_rmse, variance_global, variance_occupied) - global_rmse,
    occupied_rmse (ground-truth-masked), and variance_global/variance_occupied are all computed for
    logging and checkpoint selection only, not the reward - see plot_iteration_trajectory,
    evaluate_generalization.
    """

    after_metrics = _rmse_metrics(mu_after, env)
    global_rmse = after_metrics["global_rmse"]
    occupied_rmse = after_metrics["occupied_rmse"]
    occupied_mask = after_metrics["occupied_mask"].ravel()

    variance_after_global = float(np.sum(np.diag(P_after)))
    variance_after_occupied = float(np.sum(np.diag(P_after)[occupied_mask]))

    belief_utility = importance_filter(mu_after, P_after, beta=UCB_BETA, threshold=UTILITY_THRESHOLD)
    belief_mask = belief_utility > 0
    variance_after_belief = float(np.sum(np.diag(P_after)[belief_mask]))

    reward = -VARIANCE_WEIGHT * (
        variance_after_belief / env.baseline_total_variance
    ) ** CONCAVITY_FACTOR

    return (
        float(reward), float(global_rmse), float(occupied_rmse),
        float(variance_after_global), float(variance_after_occupied),
    )


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
    variance_global: torch.Tensor    # [T*K]   - same, but total posterior variance (trace of P)
                                      #           after that environment step's chunk, whole grid
    variance_occupied: torch.Tensor  # [T*K]   - same, but restricted to the ground-truth
                                      #           occupied_mask cells

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
        variance_global=buffer.variance_global,
        variance_occupied=buffer.variance_occupied,
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
        variance_global=torch.cat([b.variance_global for b in shifted]),
        variance_occupied=torch.cat([b.variance_occupied for b in shifted]),
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


def _sample_episode_envs(train_envs, num_episodes, iteration):
    """Picks which map each of this iteration's `num_episodes` pooled episodes trains against.
    Concatenates whole shuffled passes over `train_envs` (reshuffled each pass) rather than
    sampling every slot i.i.d., so within any one pass every map gets seen at most once before any
    map repeats. With the current pool (TRAIN_MAP_IDS, 50 maps) bigger than num_episodes (32), the
    loop below only ever runs one pass: each iteration trains on 32 distinct maps drawn from a
    fresh shuffle of all 50, and the other 18 sit out that round entirely - there's no repeat
    within a round, but also no persistent cross-iteration cursor, so coverage across iterations is
    even on average (each map ~32/50 = 64% likely per round) rather than a strict round-robin.
    Seeded off `iteration` so a given iteration's map assignment is reproducible across reruns."""
    rng = random.Random(iteration)
    envs = []
    while len(envs) < num_episodes:
        shuffled = list(train_envs)
        rng.shuffle(shuffled)
        envs.extend(shuffled)
    return envs[:num_episodes]


@torch.no_grad()
def collect_rollouts_batched(envs, pool, num_steps=None, clip_x0=True, eta=1.0, base_seed=None,
                              horizon=ENV_HORIZON, chunk_size=EXECUTION_CHUNK,
                              num_fine_tune_steps=NUM_FINE_TUNE_STEPS, record_trajectory_episode=None):
    """Batched replacement for collect_rollouts + collect_pooled_episodes' per-episode loop: runs
    `len(envs)` independent episodes SIMULTANEOUSLY in lockstep, under the current (frozen) policy
    weights - i.e. pi_theta_old - batching every neural-network call across all of them (batch dim
    = len(envs)) instead of running each episode as its own serial batch-1 call spread across MPI
    ranks (see module docstring for why this matters).

    `envs`: list of `SingleEnvironment`, one per episode slot - NOT necessarily all the same map.
    _sample_episode_envs picks these per training iteration from the training map pool, so a
    single batched call can mix episodes against different maps; each episode only ever reads its
    own envs[i] (own true_map_flat, own baseline_*), never another episode's.

    `pool`: a persistent multiprocessing.Pool (see train()'s `with mp.get_context("spawn").Pool(...)`),
    reused across the whole training run - NOT created/torn down per call, which would repay
    worker-startup cost every replan. Each replan's per-episode Kalman-filter/dynamics chunk is
    dispatched to it via dppo_rollout_worker.pool_task instead of running in a serial Python loop
    in THIS process - measured ~3x wall-clock reduction for that part (see NUM_ROLLOUT_WORKERS'
    comment). That part is still the one piece of the rollout that can't be GPU-batched (see
    module docstring) - this only parallelizes it across CPU cores instead, nothing about what
    gets computed changes (dppo_rollout_worker.rollout_chunk is a verified byte-identical
    reimplementation of _rollout_trajectory/apply_measurement_update_3d).

    At each of `horizon` replans: build each episode's OWN belief-conditioning tensors (from its
    own, independently-evolving mu/P/position - these do NOT get batched, only the resulting
    tensors do), stack them into one batch, then run the FULL K-step denoising loop with ONE
    batched model call per denoising step. After the batched denoising finishes, each episode's
    environment step (Kalman-filter measurement update + drone dynamics - plain NumPy, per-episode
    state, not GPU-batchable) runs via the worker pool across CPU cores instead of a serial
    Python for-loop in this process.

    Returns a list of `len(envs)` RolloutBuffers - one per episode, IDENTICAL shape/fields to
    collect_rollouts' single-episode buffer - ready to feed straight into compute_advantages and
    _concat_rollout_buffers exactly as before; only how the underlying data got computed changed,
    not its shape or meaning. Also returns the position history of episode index
    `record_trajectory_episode` (or None if that argument is None), for plot_iteration_trajectory.

    Runs under torch.no_grad() deliberately: pi_theta_old must stay fixed for the whole PPO
    update phase, so rollout collection never needs gradients here - the update step later
    re-runs diffusion.ddpo_ddim_sample_timestep WITH grad at the stored (x_k, x_k_minus_1) pairs
    under the *current* weights to get the new log-probs for the PPO ratio.
    """

    num_episodes = len(envs)
    schedule = _denoising_schedule(num_steps)
    device = envs[0].meanvarmarker_map.device

    if base_seed is not None:
        torch.manual_seed(base_seed)

    # Per-episode environment state, advanced independently (NumPy/CPU) but in lockstep across
    # replans with the batched model calls below. Each episode seeds from its OWN env (own
    # map's mu0/P0/starting pose) - envs[i] may differ from envs[j] since episodes can be drawn
    # from different maps in the training pool.
    mus = [envs[i].mu0.copy() for i in range(num_episodes)]
    Ps = [envs[i].P0.copy() for i in range(num_episodes)]
    cxs = [envs[i].cx0 for i in range(num_episodes)]
    cys = [envs[i].cy0 for i in range(num_episodes)]
    czs = [envs[i].cz0 for i in range(num_episodes)]
    heading_velocity_worlds = [
        np.array(envs[i].initial_heading_velocity_world, dtype=np.float32) for i in range(num_episodes)
    ]
    rngs = []
    for episode_idx in range(num_episodes):
        # SeedSequence combines distinct entropy inputs into a well-decorrelated seed - unlike
        # plain integer addition (the previous scheme), which collided constantly: whenever
        # map_id + episode_idx summed to the same value across two different episode slots
        # (verified happening on ~every iteration against the real training pool), those two
        # episodes drew bit-for-bit identical sensor noise for their whole rollout, silently
        # undermining the "N independent episodes" assumption NUM_EPISODES_PER_ITERATION pooling
        # relies on to reduce advantage-estimate variance.
        base_component = 0 if base_seed is None else base_seed
        seed_seq = np.random.SeedSequence(
            [SENSORNOISE_SEED, envs[episode_idx].map_id, base_component, episode_idx]
        )
        rngs.append(np.random.default_rng(seed_seq))

    position_histories = [
        [(cxs[i], cys[i], czs[i])] if record_trajectory_episode == i else None
        for i in range(num_episodes)
    ]

    # Warmup, matching Diffusionplanner_singlemap.py's ts<=1 branch: before the first replan, fly
    # toward WARMUP_GOAL_XY for WARMUP_STEPS measurement steps, so the first replan sees the same
    # belief (prior + WARMUP_STEPS measurements), position, and heading as deployment. Run here in
    # the main process (cheap: WARMUP_STEPS * num_episodes measurements) rather than via the pool -
    # rngs[i] is a live Generator, mutated in place by the measurement noise draws, so its advanced
    # state carries straight into the replan chunks below without needing to be returned.
    for i in range(num_episodes):
        mus[i], Ps[i], cxs[i], cys[i], czs[i], warm_positions, warm_heading = dppo_rollout_worker.warmup_chunk(
            mus[i], Ps[i], cxs[i], cys[i], czs[i],
            WARMUP_GOAL_XY, WARMUP_GOAL_XY, INIT_ALTITUDE,
            envs[i].grid_step, envs[i].xmin, envs[i].xmax, envs[i].ymin, envs[i].ymax,
            envs[i].true_map_flat, envs[i].xs, envs[i].ys,
            diffusion.Z_MIN, diffusion.Z_MAX, SAMPLESTEP, rngs[i], WARMUP_STEPS,
            record_positions=(record_trajectory_episode == i),
        )
        heading_velocity_worlds[i] = np.array(warm_heading, dtype=np.float32)
        if warm_positions is not None:
            position_histories[i].extend(warm_positions)

    # Per-episode row accumulators - mirrors RolloutBuffer's fields, kept as separate per-episode
    # lists (not one big batched buffer) so downstream code (compute_advantages,
    # _concat_rollout_buffers) needs no changes at all.
    env_steps = [[] for _ in range(num_episodes)]
    ks = [[] for _ in range(num_episodes)]
    k_prevs = [[] for _ in range(num_episodes)]
    x_ks = [[] for _ in range(num_episodes)]
    x_k_minus_1s = [[] for _ in range(num_episodes)]
    old_log_probs = [[] for _ in range(num_episodes)]
    rewards = [[] for _ in range(num_episodes)]
    global_rmses = [[] for _ in range(num_episodes)]
    occupied_rmses = [[] for _ in range(num_episodes)]
    variances_global = [[] for _ in range(num_episodes)]
    variances_occupied = [[] for _ in range(num_episodes)]
    step_meanvarmarker_maps = [[] for _ in range(num_episodes)]
    step_current_positions = [[] for _ in range(num_episodes)]
    step_heading_velocities = [[] for _ in range(num_episodes)]

    for t in range(horizon):
        # Each episode builds its OWN belief conditioning from its OWN (mu, P, position) -
        # these differ per episode - then get stacked into one batch-N tensor for the model.
        meanvarmarker_map_list, current_position_list, heading_velocity_list = [], [], []
        for i in range(num_episodes):
            mvm, cp = _build_belief_conditioning(
                mus[i], Ps[i], cxs[i], cys[i], czs[i], envs[i].xs, envs[i].ys, device,
            )
            hv = diffusion.normalize_xyz_displacement(
                torch.tensor(heading_velocity_worlds[i][None, :], dtype=torch.float32, device=device)
            )
            meanvarmarker_map_list.append(mvm[0])
            current_position_list.append(cp[0])
            heading_velocity_list.append(hv[0])
            step_meanvarmarker_maps[i].append(mvm[0])
            step_current_positions[i].append(cp[0])
            step_heading_velocities[i].append(hv[0])

        meanvarmarker_map = torch.stack(meanvarmarker_map_list, dim=0)  # [N, 3, 51, 51]
        current_position = torch.stack(current_position_list, dim=0)   # [N, 3]
        heading_velocity = torch.stack(heading_velocity_list, dim=0)   # [N, 3]

        x = torch.randn((num_episodes, *diffusion.TARGET_SHAPE), device=device)
        for step_idx, step_k in enumerate(schedule):
            prev_k = schedule[step_idx + 1] if step_idx + 1 < len(schedule) else -1
            k = torch.full((num_episodes,), step_k, dtype=torch.long, device=device)
            k_prev = torch.full((num_episodes,), prev_k, dtype=torch.long, device=device)

            # ONE batched call for all N episodes' current denoising step - this is the actual
            # GPU-batching payoff; everything else in this loop is per-episode bookkeeping.
            x_prev, mean_theta, var = diffusion.ddpo_ddim_sample_timestep(
                x, k, k_prev,
                meanvarmarker_map, current_position, heading_velocity,
                clip_x0=clip_x0, eta=eta, sigma_prob_min=SIGMA_PROB_MIN,
                min_sampling_std=SIGMA_SAMPLE_MIN,
            )

            if step_k < num_fine_tune_steps:
                var_floored = var.clamp_min(1e-8).expand_as(mean_theta)
                log_prob = -diffusion.gaussian_nll(input=mean_theta, target=x_prev, var=var_floored)
                # Per-dim clamp BEFORE summing (AID: newlogprobs/oldlogprobs.clamp(min=-5, max=2))
                # - one >3-sigma outlier dimension can otherwise dominate the whole row's summed
                # log-prob and blow up its PPO ratio. Must match _ppo_step's clamp exactly, or the
                # ratio's old/new sides are measured on different scales.
                log_prob = log_prob.clamp(min=LOGPROB_CLAMP_MIN, max=LOGPROB_CLAMP_MAX)
                log_prob = log_prob.sum(dim=(1, 2))  # [N]

                for i in range(num_episodes):
                    env_steps[i].append(t)
                    ks[i].append(step_k)
                    k_prevs[i].append(prev_k)
                    x_ks[i].append(x[i])
                    x_k_minus_1s[i].append(x_prev[i])
                    old_log_probs[i].append(log_prob[i])

            x = x_prev

        # x now holds ALL episodes' fully-denoised (k=0) actions. Execute each episode's chunk -
        # Kalman-filter/dynamics is per-episode NumPy state, not GPU-batchable (module docstring),
        # but IS independent across episodes, so dispatch all num_episodes chunks to the
        # persistent worker pool in one batched call instead of a serial per-episode Python loop.
        traj_worlds = [
            diffusion.extract_control_waypoints(x[i]).detach().cpu().numpy().T
            for i in range(num_episodes)
        ]
        mu_befores = [mus[i] for i in range(num_episodes)]
        P_befores = [Ps[i] for i in range(num_episodes)]

        tasks = [
            (
                traj_worlds[i], mus[i], Ps[i], cxs[i], cys[i], czs[i],
                envs[i].grid_step, envs[i].xmin, envs[i].xmax, envs[i].ymin, envs[i].ymax,
                envs[i].true_map_flat, envs[i].xs, envs[i].ys,
                diffusion.Z_MIN, diffusion.Z_MAX, SAMPLESTEP,
                rngs[i], chunk_size, record_trajectory_episode == i,
            )
            for i in range(num_episodes)
        ]
        results = pool.map(dppo_rollout_worker.pool_task, tasks)

        for i, (mu, P, cx, cy, cz, chunk_positions, last_step_heading, updated_rng) in enumerate(results):
            mus[i], Ps[i], cxs[i], cys[i], czs[i] = mu, P, cx, cy, cz
            # rng's internal state only advanced inside the worker process's own copy - keep
            # using the RETURNED generator for this episode's next replan, not the one sent in.
            rngs[i] = updated_rng
            if chunk_positions is not None:
                position_histories[i].extend(chunk_positions)

            # Heading for the NEXT replan = the LAST single (~SAMPLESTEP) step of this chunk, as
            # rollout_chunk returns it - matching Diffusionplanner_singlemap.py's per-step
            # current_heading_velocity at replan time. Previously this used the net displacement
            # over the whole chunk (cxs[i]-previous_poses[i]), which is ~18x larger in normalized
            # magnitude and put the model's heading conditioning far out of the distribution it
            # sees at deployment - a train/deploy mismatch val-eval could not catch.
            heading_velocity_worlds[i] = np.array(last_step_heading, dtype=np.float32)
            reward, global_rmse, occupied_rmse, variance_global, variance_occupied = compute_step_reward(
                mu_befores[i], mus[i], P_befores[i], Ps[i], envs[i], t,
            )
            rows_this_step = min(num_fine_tune_steps, len(schedule))
            rewards[i].extend([0.0] * (rows_this_step - 1) + [reward])
            global_rmses[i].extend([0.0] * (rows_this_step - 1) + [global_rmse])
            occupied_rmses[i].extend([0.0] * (rows_this_step - 1) + [occupied_rmse])
            variances_global[i].extend([0.0] * (rows_this_step - 1) + [variance_global])
            variances_occupied[i].extend([0.0] * (rows_this_step - 1) + [variance_occupied])

    buffers = []
    for i in range(num_episodes):
        buffers.append(RolloutBuffer(
            # device= matches x_k/step_meanvarmarker_map/etc. below (already on `device` via
            # torch.stack of device-resident tensors) - these six were previously left on CPU
            # by default, so update_policy/update_value's index tensors (derived from buffer.k)
            # would mismatch buffer.x_k/step_meanvarmarker_map's device the moment a real GPU
            # is in use.
            env_step=torch.tensor(env_steps[i], dtype=torch.long, device=device),
            k=torch.tensor(ks[i], dtype=torch.long, device=device),
            k_prev=torch.tensor(k_prevs[i], dtype=torch.long, device=device),
            x_k=torch.stack(x_ks[i]),
            x_k_minus_1=torch.stack(x_k_minus_1s[i]),
            old_log_prob=torch.stack(old_log_probs[i]),
            reward=torch.tensor(rewards[i], dtype=torch.float32, device=device),
            global_rmse=torch.tensor(global_rmses[i], dtype=torch.float32, device=device),
            occupied_rmse=torch.tensor(occupied_rmses[i], dtype=torch.float32, device=device),
            variance_global=torch.tensor(variances_global[i], dtype=torch.float32, device=device),
            variance_occupied=torch.tensor(variances_occupied[i], dtype=torch.float32, device=device),
            step_meanvarmarker_map=torch.stack(step_meanvarmarker_maps[i]),
            step_current_position=torch.stack(step_current_positions[i]),
            step_heading_velocity=torch.stack(step_heading_velocities[i]),
        ))

    trajectory = (
        position_histories[record_trajectory_episode] if record_trajectory_episode is not None else None
    )
    return buffers, trajectory


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
    """
    with torch.no_grad():
        num_env_steps = buffer.step_meanvarmarker_map.shape[0]

        state_values = value_fn(
            buffer.step_meanvarmarker_map,
            buffer.step_current_position,
            buffer.step_heading_velocity,
        ).squeeze(-1)  # [T]
        # value_fn's inputs/output live on whatever device the model was moved to (GPU, if one
        # is in use). buffer.reward/buffer.k are now built with device=device in
        # collect_rollouts_batched, but torch.zeros(1) below still defaults to CPU unless told
        # otherwise - keep it explicit rather than relying on buffer's device matching by luck.
        device = state_values.device
        step_reward = buffer.reward[buffer.k == 0].to(device)  # [T], one reward per environment step

        # TD residuals delta_t = R_t + gamma_ENV * V(s_{t+1}) - V(s_t); V(s_T) = 0 since every
        # episode here is a fixed-length T-step rollout with no bootstrapping past the horizon.
        next_state_values = torch.cat([state_values[1:], torch.zeros(1, device=device)])
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
        # step_advantages is CPU (built via a sequential Python backward-recursion loop above,
        # not worth moving to GPU for), but buffer.env_step now lives on `device` (see
        # collect_rollouts_batched's RolloutBuffer construction) - move the index to CPU for the
        # lookup, then bring the result back to `device` to match denoising_discount below.
        row_advantage = step_advantages[buffer.env_step.cpu()].to(device)
        denoising_discount = denoising_gamma ** (buffer.k.float())
        row_advantage = row_advantage * denoising_discount

        return row_advantage, step_returns


def collect_pooled_episodes_batched(envs, value_fn, pool, base_seed, record_trajectory=False,
                                      denoising_gamma=0.99, env_gamma=ENV_GAMMA, gae_lambda=GAE_LAMBDA):
    """Single-process replacement for collect_pooled_episodes: collects all len(envs) episodes
    in ONE call to collect_rollouts_batched (batched neural-net calls across episodes - see module
    docstring), computes GAE per-episode, then pools exactly as before. No MPI - this entire
    function runs in one process, since batching the model calls across episodes IS the
    parallelism now, instead of spreading episodes across MPI ranks.

    `envs`: list of SingleEnvironment, one per episode slot (see _sample_episode_envs) - may mix
    several different maps within the same pooled batch for multi-map training.

    The pooled row_advantage is standardized (zero mean, unit std) across the WHOLE pooled batch
    before being returned - deliberately NOT per-episode, which would erase genuine differences
    between a good episode and a bad one. Raw GAE advantage magnitude can vary a lot from
    iteration to iteration depending on which episodes happened to get sampled; without this, an
    iteration whose pooled batch happens to have unusually large raw advantages produces a
    proportionally larger PPO gradient step regardless of clip_epsilon, since clipping only
    bounds how much of that push counts toward the loss, not how large the push actually is.
    """
    buffers, iter_trajectory = collect_rollouts_batched(
        envs, pool, base_seed=base_seed,
        record_trajectory_episode=0 if record_trajectory else None,
    )

    row_advantages, step_returns_list = [], []
    for buf, ep_env in zip(buffers, envs):
        row_advantage, step_returns = compute_advantages(
            buf, value_fn, ep_env,
            denoising_gamma=denoising_gamma, env_gamma=env_gamma, gae_lambda=gae_lambda,
        )
        row_advantages.append(row_advantage)
        step_returns_list.append(step_returns)

    pooled_buffer = _concat_rollout_buffers(buffers)
    pooled_row_advantage = torch.cat(row_advantages)
    pooled_row_advantage = (
        (pooled_row_advantage - pooled_row_advantage.mean()) / (pooled_row_advantage.std() + 1e-8)
    )
    # Quantile clip after standardization (AID's clip_advantage_lower/upper_quantile, 5%/95%):
    # standardization fixes the batch's overall scale but a handful of extreme-advantage rows can
    # still dominate the PPO gradient - clip_epsilon bounds how much of their push counts toward
    # the loss, not how hard they yank the shared weights within the window. Applied once to the
    # pooled batch (AID applies it per-minibatch inside its loss; same intent, simpler here).
    pooled_row_advantage = pooled_row_advantage.clamp(
        min=torch.quantile(pooled_row_advantage, 0.05),
        max=torch.quantile(pooled_row_advantage, 0.95),
    )
    pooled_step_returns = torch.cat(step_returns_list)
    return pooled_buffer, pooled_row_advantage, pooled_step_returns, iter_trajectory


def evaluate_generalization(val_envs, iteration, pool):
    """Periodic honest generalization check: runs ONE no-grad episode per held-out validation map
    (VAL_MAP_IDS, never used for a training rollout) under the CURRENT policy weights - via the
    same collect_rollouts_batched used for training, just with no policy/value update afterward -
    and prints mean terminal reward/global_rmse/occupied_rmse/variance across them. Called every
    VAL_EVAL_EVERY iterations from train(). If the policy is genuinely generalizing (not just
    memorizing TRAIN_MAP_IDS), these numbers should improve over training the same way the
    training-map metrics do; if training metrics improve while these stay flat or worsen, that's
    overfitting to the training map pool.

    Returns the mean terminal variance fraction across validation maps (each episode's terminal
    whole-grid variance divided by its OWN map's no-flight baseline, then averaged - not a
    ratio of means, so no single map's larger baseline dominates). train() uses this as the
    best-checkpoint selection criterion: selection is FOR generalization, so it must be measured
    ON held-out maps, not on the training batch's own rollout (which rewards overfitting -
    exactly the failure mode that made earlier dppo_best.pth checkpoints lose to plain BC on
    genuinely unseen maps).

    Like every other single-rollout eval in this codebase, this is n=1 per map per check - noisy
    rollout-to-rollout (see CARRYOVER.md's periodic-eval jumpiness note), so read it as a trend
    over several VAL_EVAL_EVERY checks, not a single number to react to.
    """
    buffers, _ = collect_rollouts_batched(
        val_envs, pool, base_seed=iteration, record_trajectory_episode=None,
    )

    terminal_rewards, terminal_global_rmses, terminal_occupied_rmses = [], [], []
    terminal_variance_fractions = []
    for buf, env in zip(buffers, val_envs):
        terminal = buf.k == 0
        # Mean per-step reward (dense reward - one nonzero reward per env_step; see train()).
        terminal_rewards.append(buf.reward[terminal].mean().item())
        terminal_global_rmses.append(buf.global_rmse[terminal].mean().item())
        terminal_occupied_rmses.append(buf.occupied_rmse[terminal].mean().item())
        terminal_variance_fractions.append(
            buf.variance_global[terminal].mean().item() / env.baseline_total_variance
        )

    mean_baseline_global_rmse = sum(e.baseline_global_rmse for e in val_envs) / len(val_envs)
    mean_variance_fraction = sum(terminal_variance_fractions) / len(terminal_variance_fractions)
    print(
        f"    [val @ iteration {iteration}] "
        f"mean_reward={sum(terminal_rewards) / len(terminal_rewards):.4f}, "
        f"global_rmse={sum(terminal_global_rmses) / len(terminal_global_rmses):.4f} "
        f"(baseline {mean_baseline_global_rmse:.4f}), "
        f"occupied_rmse={sum(terminal_occupied_rmses) / len(terminal_occupied_rmses):.4f}, "
        f"variance_fraction={mean_variance_fraction:.4f}, "
        f"maps={[e.map_id for e in val_envs]}"
    )
    return mean_variance_fraction


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
    # Same per-dim clamp as collection applied to old_log_prob (see LOGPROB_CLAMP_MIN's comment)
    # - both sides of the ratio must be measured on the same clamped scale.
    new_log_prob = new_log_prob.clamp(min=LOGPROB_CLAMP_MIN, max=LOGPROB_CLAMP_MAX)
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
        losses = []
        for group in (terminal_idx, nonterminal_idx):
            # group (derived from buffer.k, now device-resident) must be permuted with a
            # same-device index - torch.randperm defaults to CPU regardless of group's device.
            perm = group[torch.randperm(group.shape[0], device=group.device)]
            for start in range(0, perm.shape[0], minibatch_size):
                idx = perm[start:start + minibatch_size]
                loss, clip_fraction, approx_kl = _ppo_step(
                    idx, buffer, discounted_advantage, env, policy_optimizer, clip_epsilon,
                )
                clip_fractions.append(clip_fraction)
                approx_kls.append(approx_kl)
                losses.append(loss)
        mean_clip_fraction = sum(clip_fractions) / len(clip_fractions)
        mean_approx_kl = sum(approx_kls) / len(approx_kls)
        mean_loss = sum(losses) / len(losses)
        print(
            f"epoch {epoch}: policy_loss={mean_loss:.4f}, clip_fraction={mean_clip_fraction:.3f}, "
            f"approx_kl={mean_approx_kl:.4f}"
        )
        if mean_approx_kl > target_kl:
            print(
                f"epoch {epoch}: approx_kl {mean_approx_kl:.4f} exceeded target_kl {target_kl} "
                "- stopping this iteration's policy update early"
            )
            break
        if mean_clip_fraction > CLIP_FRACTION_EARLY_STOP:
            print(
                f"epoch {epoch}: clip_fraction {mean_clip_fraction:.3f} exceeded "
                f"{CLIP_FRACTION_EARLY_STOP} - rollout data is stale for the current policy, "
                "stopping this iteration's policy update early"
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
    # step_returns is CPU (built via compute_advantages' sequential backward-recursion loop),
    # while env_step_idx now lives on `device` (buffer.env_step is device-resident) - move the
    # index to CPU for this lookup, then bring the result back to `device`.
    target_return = step_returns[env_step_idx.cpu()].to(device)  # [B]

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
        perm = terminal_idx[torch.randperm(terminal_idx.shape[0], device=terminal_idx.device)]
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


def _prune_old_checkpoints(iteration, keep_last=CHECKPOINT_KEEP_LAST):
    """Deletes the numbered per-iteration checkpoint from `keep_last` iterations ago, if it
    exists - keeps only a rolling window on disk for an unattended multi-hour run, independent
    of dppo_best.pth (_save_best_checkpoint), which is preserved indefinitely regardless of this
    window."""
    old_iteration = iteration - keep_last
    if old_iteration < 0:
        return
    old_path = CHECKPOINT_DIR / f"dppo_iteration_{old_iteration:04d}.pth"
    if old_path.exists():
        old_path.unlink()


def _save_best_checkpoint(model, value_fn, iteration, variance_fraction):
    """Overwrites a single dedicated dppo_best.pth whenever mean terminal variance (as a fraction
    of the no-flight baseline, lower = more variance reduced) improves on the best seen so far
    this run - so the single best iteration of an unattended overnight run is trivially
    reloadable afterward without scanning the log, and without depending on whether
    _prune_old_checkpoints has already deleted that iteration's numbered checkpoint.

    Selects on the whole-grid variance fraction specifically - not global_rmse/occupied_rmse
    (never scored by the loss; a global_rmse-based criterion was picking checkpoints by a
    quantity the policy was never gradient-updated toward, which can diverge freely from which
    iteration was actually best - a map can have low mean-error RMSE from a lucky prior while
    still carrying high posterior uncertainty), and not the reward's own belief-masked terminal
    variance (see compute_step_reward) - whole-grid keeps the selection metric fixed and
    comparable across iterations/runs, where the belief mask's extent varies with each policy's
    own final belief.

    When val_envs are available, the fraction passed in comes from evaluate_generalization's
    HELD-OUT validation maps (see train()) - selection is for generalization, so it's measured on
    maps the policy never trains against; the training batch's own metric is only used as a
    fallback when no validation pool exists."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "value_state_dict": value_fn.state_dict(),
            "iteration": iteration,
            "variance_fraction": variance_fraction,
        },
        CHECKPOINT_DIR / "dppo_best.pth",
    )


def train(train_envs, val_envs=None, num_iterations=NUM_DPPO_ITERATIONS,
          num_episodes=NUM_EPISODES_PER_ITERATION):
    """
    Outer DPPO loop - single process, GPU-batched across episodes (see module docstring). Each
    iteration:
      1. samples which map each of `num_episodes` pooled episodes trains against this iteration
         (_sample_episode_envs, drawn from `train_envs` - may mix several maps in one batch),
      2. collects those episodes FRESH under the CURRENT policy, all SIMULTANEOUSLY via
         collect_pooled_episodes_batched's batched model calls,
      3. updates the policy off the pooled advantage, then updates the value function off the
         pooled return-to-go,
      4. saves a checkpoint (pruned to a rolling window) and updates the best-checkpoint if this
         iteration improved on it,
      5. every VAL_EVAL_EVERY iterations, checks generalization against `val_envs` (maps never
         used in step 1) via evaluate_generalization - skipped if val_envs is None.

    No MPI, no broadcast - there's only ever one process holding one copy of the model/value_fn,
    so nothing needs to be kept in sync across ranks.

    diffusion.model, value_fn, and both optimizers (with their Adam momentum) persist across
    iterations - only the pooled rollout buffer itself is discarded and replaced every
    iteration, so later iterations build on earlier ones instead of restarting from scratch.
    """
    print(
        f"Running batched DPPO training (single process, GPU-batched across {num_episodes} "
        f"episodes, map pool={[e.map_id for e in train_envs]})"
    )
    if VISUALIZE:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    model = _load_checkpoint(DEFAULT_CHECKPOINT, diffusion.device)

    value_fn = ValueFunction().to(train_envs[0].meanvarmarker_map.device)
    policy_optimizer = torch.optim.AdamW(diffusion.model.parameters(), lr=1e-5)
    value_optimizer = torch.optim.AdamW(value_fn.parameters(), lr=VALUE_LR)

    best_variance_fraction = float("inf")

    # Persistent worker pool for collect_rollouts_batched's per-episode Kalman-filter/dynamics
    # chunks (see NUM_ROLLOUT_WORKERS' comment and dppo_rollout_worker.py) - created ONCE here,
    # not per iteration, so worker-startup cost is paid once for the whole run, not every replan.
    # "spawn" (not "fork") is required: threeDSparseTransDiffusion.py calls
    # torch.cuda.is_available() at import time, which has already happened by the time train()
    # runs, and forking a process with an already-initialized CUDA context is unsafe.
    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=NUM_ROLLOUT_WORKERS, initializer=dppo_rollout_worker.pool_worker_init)
    try:
        for iteration in range(num_iterations):
            episode_envs = _sample_episode_envs(train_envs, num_episodes, iteration)

            should_plot = VISUALIZE and iteration % PLOT_EVERY == 0
            buffer, discounted_advantages, step_returns, iter_trajectory = collect_pooled_episodes_batched(
                episode_envs, value_fn, pool,
                base_seed=iteration * num_episodes,
                record_trajectory=should_plot,
            )

            # Baselines now vary per episode (each map has its own no-flight baseline) - the
            # printed comparison uses this iteration's sampled batch's own mean baseline, not one
            # fixed env's.
            mean_baseline_global_rmse = sum(e.baseline_global_rmse for e in episode_envs) / num_episodes
            mean_baseline_occupied_rmse = sum(e.baseline_occupied_rmse for e in episode_envs) / num_episodes
            mean_baseline_total_variance = sum(e.baseline_total_variance for e in episode_envs) / num_episodes
            mean_baseline_occupied_variance = sum(e.baseline_occupied_variance for e in episode_envs) / num_episodes

            terminal = buffer.k == 0
            # Mean per-step reward across all env-step terminal (k=0) rows. With the dense reward
            # every env_step has a nonzero reward, so this is the average per-step variance-level
            # reward - directly comparable across iterations regardless of ENV_HORIZON.
            mean_reward = buffer.reward[terminal].mean().item()
            mean_global_rmse = buffer.global_rmse[terminal].mean().item()
            mean_occupied_rmse = buffer.occupied_rmse[terminal].mean().item()
            mean_variance_global = buffer.variance_global[terminal].mean().item()
            mean_variance_occupied = buffer.variance_occupied[terminal].mean().item()
            mean_variance_fraction = mean_variance_global / mean_baseline_total_variance
            mean_variance_occupied_fraction = mean_variance_occupied / mean_baseline_occupied_variance
            print(
                f"=== iteration {iteration}: mean_reward={mean_reward:.4f}, "
                f"post-flight global_rmse={mean_global_rmse:.4f} "
                f"(baseline {mean_baseline_global_rmse:.4f}), "
                f"occupied_rmse={mean_occupied_rmse:.4f} "
                f"(baseline {mean_baseline_occupied_rmse:.4f}), "
                f"variance_global={mean_variance_global:.4f} (baseline {mean_baseline_total_variance:.4f}, "
                f"fraction {mean_variance_fraction:.4f}), "
                f"variance_occupied={mean_variance_occupied:.4f} "
                f"(baseline {mean_baseline_occupied_variance:.4f}, "
                f"fraction {mean_variance_occupied_fraction:.4f}), "
                f"episodes_pooled={num_episodes}, "
                f"maps={sorted(set(e.map_id for e in episode_envs))} ==="
            )
            if should_plot:
                plot_iteration_trajectory(iter_trajectory, episode_envs[0], iteration)

            # Critic warmup (AID's n_critic_warmup_itr): value function trains from iteration 0,
            # the actor waits N_CRITIC_WARMUP_ITERATIONS - see the constant's comment.
            if iteration >= N_CRITIC_WARMUP_ITERATIONS:
                update_policy(
                    buffer, discounted_advantages, episode_envs[0], policy_optimizer,
                    num_epochs=NUM_PPO_EPOCHS, minibatch_size=MINIBATCH_SIZE, clip_epsilon=CLIP_EPSILON,
                    target_kl=TARGET_KL,
                )
            else:
                print(
                    f"    critic warmup ({iteration + 1}/{N_CRITIC_WARMUP_ITERATIONS}): "
                    "skipping policy update, value function only"
                )
            update_value(
                buffer, step_returns, episode_envs[0], value_fn, value_optimizer,
                num_epochs=NUM_VALUE_EPOCHS, minibatch_size=MINIBATCH_SIZE,
            )
            if iteration % CHECKPOINT_EVERY == 0:
                _save_checkpoint(diffusion.model, value_fn, iteration)
                _prune_old_checkpoints(iteration)

            # Best-checkpoint selection runs on the HELD-OUT validation maps, not the training
            # batch's own rollout. Selecting on the training batch metric rewards exactly the
            # overfitting-to-the-map-pool failure this pipeline has already demonstrated (earlier
            # dppo_best.pth checkpoints beat BC on training maps yet lost to it on unseen maps).
            # The price: selection only happens every VAL_EVAL_EVERY iterations, so a between-eval
            # best can be missed - lower VAL_EVAL_EVERY if that matters more than the eval cost
            # (one extra episode per val map per check). Falls back to the training-batch metric
            # only when no val_envs were provided at all.
            if val_envs and iteration % VAL_EVAL_EVERY == 0:
                val_variance_fraction = evaluate_generalization(val_envs, iteration, pool)
                if val_variance_fraction < best_variance_fraction:
                    best_variance_fraction = val_variance_fraction
                    _save_best_checkpoint(diffusion.model, value_fn, iteration, val_variance_fraction)
                    print(
                        f"    new best VAL variance_fraction={val_variance_fraction:.4f} "
                        f"at iteration {iteration} - saved dppo_best.pth"
                    )
            elif not val_envs and mean_variance_fraction < best_variance_fraction:
                best_variance_fraction = mean_variance_fraction
                _save_best_checkpoint(diffusion.model, value_fn, iteration, mean_variance_fraction)
                print(f"    new best variance_fraction={mean_variance_fraction:.4f} at iteration {iteration} - saved dppo_best.pth")
    finally:
        pool.close()
        pool.join()

    return model, value_fn


if __name__ == "__main__":
    train_envs = list(build_environment_pool(TRAIN_MAP_IDS).values())
    val_envs = list(build_environment_pool(VAL_MAP_IDS).values())

    print(f"DPPO training map pool: {[e.map_id for e in train_envs]}")
    print(f"DPPO held-out validation maps: {[e.map_id for e in val_envs]}")
    mean_train_baseline = sum(e.baseline_global_rmse for e in train_envs) / len(train_envs)
    mean_val_baseline = sum(e.baseline_global_rmse for e in val_envs) / len(val_envs)
    print(f"Mean baseline global RMSE (no flight) - train pool: {mean_train_baseline:.4f}")
    print(f"Mean baseline global RMSE (no flight) - held-out: {mean_val_baseline:.4f}")

    train(
        train_envs, val_envs,
        num_iterations=NUM_DPPO_ITERATIONS, num_episodes=NUM_EPISODES_PER_ITERATION,
    )
