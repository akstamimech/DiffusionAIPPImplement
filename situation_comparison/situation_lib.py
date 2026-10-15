"""
Shared machinery for the CMA-ES-vs-Diffusion-vs-ImitateTrans "same frozen
situation" comparison experiment.

Methodology: pick a real (mu, P, pose) belief state from an actual CMA-ES
flight (not synthetic), reconstructed by replaying the Kalman/GP mean and
covariance updates using the recorded trajectory poses, the real ground-truth
map, and the same seeded sensor noise the original flight used - so the
reconstructed belief is bit-identical to what CMA-ES actually had at that
instant. Then feed that identical belief to all three planners' one-shot
"situation in, plan out" functions (real_receding_horizon_planner,
sample_diffusion_trajectory, sample_imitation_trajectory - all already exist,
nothing new was added to the planners themselves) and compare what each
proposes.

Note on reproducibility: Diffusion's own sampling noise (sparse_noise in
sample_diffusion_trajectory) is drawn from the global torch RNG, which is NOT
seeded here (no RUN_SEED set) - so re-running the identical frozen situation
gives a genuinely different Diffusion sample each time (this is expected
diffusion-model behavior, not a bug - see this project's multimodality
investigation elsewhere). u_diff/diff_ratio will vary run to run by a
noticeable amount; treat individual saved numbers as one sample, not an exact
reproducible constant. CMA-ES and ImitateTrans are deterministic given the
frozen state (ImitateTrans is a single feedforward pass; CMA-ES's grid search
is deterministic and its refinement step is seeded via CMA_SEED).

Two-tier storage, by design (see results_index.csv / README):
  - belief_cache/  regenerable cheap cache (covariance diagonal only) used
    for the clustering search. Not curated, safe to delete and rebuild.
  - results/ + plots/  curated, saved ONLY for situations a human decided
    were worth keeping (e.g. clearly demonstrate a planner failure mode).
    Every candidate situation that gets scored is NOT saved here - only
    save_situation.py writes to this tier, deliberately, per-row.
"""
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # Thesis/scripts
DIFFUSION_DIR = SCRIPT_DIR / "Diffusion"
for _p in (SCRIPT_DIR, DIFFUSION_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from gaussianprocesstraining import initialize_gp, importance_filter, build_spline_trajectory_3d
from Diffusionplanner_singlemap import (
    apply_measurement_update_3d,
    build_true_map_flat,
    load_diffusion_model,
    sample_diffusion_trajectory,
)
from CMAES_classic_singlemap import real_receding_horizon_planner
from ImitateTrans_singlemap import (
    load_imitate_model,
    sample_imitation_trajectory,
    ZMIN as IMITATE_ZMIN,
    ZMAX as IMITATE_ZMAX,
)
from sample_3d_sparse_trans_diffusion import diffusion as diffusion_module

THIS_DIR = Path(__file__).resolve().parent
BELIEF_CACHE_DIR = THIS_DIR / "belief_cache"
RESULTS_DIR = THIS_DIR / "results"
PLOTS_DIR = THIS_DIR / "plots"
INDEX_CSV = THIS_DIR / "results_index.csv"

import os

# Defaults match this project's CURRENT live settings (checked directly
# against gaussianprocesstraining.py / Diffusionplanner_singlemap.py /
# ImitateTrans_singlemap.py on 2026-08-17: MAPTYPE=grf, UTILITY_THRESHOLD=0.5,
# LCB=False/UCB, checkpoints current_best_updated.pth / Imitate_best.pth).
# This project's map-type settings have flipped between NAIP and GRF multiple
# times across sessions - RE-VERIFY these against the live files before
# trusting them if it's been a while; don't assume this comment is still
# accurate. LCB itself is a hardcoded module-level constant in
# gaussianprocesstraining.py, NOT env-var controlled, so switching MAPTYPE
# here alone does not flip importance-mask polarity - check LCB directly too.
MAPTYPE = os.environ.get("SITUATION_MAPTYPE", "grf")
UTILITY_THRESHOLD = float(os.environ.get("SITUATION_UTILITY_THRESHOLD", "0.5"))
BETA = 1.0
SENSORNOISE_SEED = 123
PLANNING_HORIZON = 8
ALPHA = 0.02
CMA_SEED = 55
SAMPLES_PER_SEGMENT = 5

DIFFUSION_CHECKPOINT = Path(os.environ.get(
    "SITUATION_DIFFUSION_CHECKPOINT",
    str(DIFFUSION_DIR / "checkpoints" / "current_best_updated.pth"),
))
IMITATE_CHECKPOINT = Path(os.environ.get(
    "SITUATION_IMITATE_CHECKPOINT",
    str(SCRIPT_DIR / "checkpoints" / "Imitate_best.pth"),
))
CSV_DIR = SCRIPT_DIR / "csv"


def cmaes_run_dir(selected_map):
    return SCRIPT_DIR / "Vizualization" / f"classic_map_{MAPTYPE}_{selected_map}_viz_situation_search"


def cmaes_traj_csv(selected_map):
    return cmaes_run_dir(selected_map) / f"map_{selected_map}_executed_trajectory.csv"


def load_ground_truth(selected_map, X_test, step):
    data = np.loadtxt(CSV_DIR / f"map_{selected_map}_{MAPTYPE}_grid_counts.csv", delimiter=",", skiprows=1)
    pts = data[:, 0:3]
    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]
    return build_true_map_flat(pts, X_test)


def replay_diag_belief_history(selected_map, cache=True):
    """Cheap-ish replay: full mean history + covariance DIAGONAL only (not
    full P). apply_measurement_update_3d returns both mu and P from the same
    call regardless, so caching mu_hist alongside diagP_hist costs nothing
    extra - only the O(n^2) full-P storage is skipped. Used for the
    clustering search, which needs the real mu (not an approximation) to
    match evaluate_situations.py's importance mask exactly. Not a source of
    truth for running planners, which need the full P
    (see reconstruct_full_state_at_row)."""
    cache_path = BELIEF_CACHE_DIR / f"map{selected_map}_belief_diag_history.npz"
    if cache and cache_path.exists():
        d = np.load(cache_path)
        return d["mu_hist"], d["diagP_hist"], d["traj"], d["xs"], d["ys"], float(d["step"])

    gp, X_test, mean0, cov0, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()
    true_map_flat = load_ground_truth(selected_map, X_test, step)
    traj = np.loadtxt(cmaes_traj_csv(selected_map), delimiter=",", skiprows=1)

    rng = np.random.default_rng(SENSORNOISE_SEED + selected_map)
    mean = np.full(X_test.shape[0], UTILITY_THRESHOLD + 0.1)
    mu = mean.copy()
    P = cov0.copy()

    n_rows = traj.shape[0]
    mu_hist = np.zeros((n_rows, mu.shape[0]), dtype=np.float32)
    diagP_hist = np.zeros((n_rows, mu.shape[0]), dtype=np.float32)
    mu_hist[0] = mu
    diagP_hist[0] = np.diag(P)
    for i in range(1, n_rows):
        ts, wall_t, x, y, z = traj[i]
        mu, P = apply_measurement_update_3d(x, y, z, mu, P, true_map_flat, xs, ys, rng)
        mu_hist[i] = mu
        diagP_hist[i] = np.diag(P)

    if cache:
        BELIEF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, mu_hist=mu_hist, diagP_hist=diagP_hist, traj=traj, xs=xs, ys=ys, step=step)
    return mu_hist, diagP_hist, traj, xs, ys, step


def reconstruct_full_state_at_row(selected_map, target_row):
    """Expensive replay: full mu AND P. Needed to actually run the planners -
    CMA-ES's grid search uses the full covariance structure, not just its
    diagonal, so the cheap diag-only cache above isn't enough here."""
    gp, X_test, mean0, cov0, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()
    true_map_flat = load_ground_truth(selected_map, X_test, step)
    traj = np.loadtxt(cmaes_traj_csv(selected_map), delimiter=",", skiprows=1)

    rng = np.random.default_rng(SENSORNOISE_SEED + selected_map)
    mean = np.full(X_test.shape[0], UTILITY_THRESHOLD + 0.1)
    mu = mean.copy()
    P = cov0.copy()

    for i in range(1, target_row + 1):
        ts, wall_t, x, y, z = traj[i]
        mu, P = apply_measurement_update_3d(x, y, z, mu, P, true_map_flat, xs, ys, rng)

    pose_now = traj[target_row, 2:5]
    pose_prev = traj[target_row - 1, 2:5]
    heading_velocity = pose_now - pose_prev

    return dict(
        mu=mu, P=P, xs=xs, ys=ys, X=X, Y=Y, step=step,
        xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        pose=pose_now, heading_velocity=heading_velocity,
        row=target_row, wall_time=float(traj[target_row, 1]),
    )


_diffusion_model = None
_imitate_model = None


def get_diffusion_model():
    global _diffusion_model
    if _diffusion_model is None:
        _diffusion_model = load_diffusion_model(checkpoint_path=DIFFUSION_CHECKPOINT)
    return _diffusion_model


def get_imitate_model():
    global _imitate_model
    if _imitate_model is None:
        _imitate_model = load_imitate_model(checkpoint_path=IMITATE_CHECKPOINT)
    return _imitate_model


def utility_along_path(xy, xs, ys, utility_grid, step):
    ix = np.clip(np.round((xy[:, 0] - xs[0]) / step).astype(int), 0, len(xs) - 1)
    iy = np.clip(np.round((xy[:, 1] - ys[0]) / step).astype(int), 0, len(ys) - 1)
    return float(np.mean(utility_grid[iy, ix]))


def run_three_planners(state):
    mu, P = state["mu"], state["P"]
    xs, ys, X, step = state["xs"], state["ys"], state["X"], state["step"]
    xmin, xmax, ymin, ymax = state["xmin"], state["xmax"], state["ymin"], state["ymax"]
    cx, cy, cz = state["pose"]
    heading_velocity = state["heading_velocity"]

    diag_var = np.diag(P)
    utility_grid = importance_filter(mu, P, BETA, threshold=UTILITY_THRESHOLD).reshape(X.shape)

    cmaes_waypoints = real_receding_horizon_planner(
        cx, cy, cz, mu, P, xs, ys,
        utility_threshold=UTILITY_THRESHOLD, beta=BETA,
        planning_horizon=PLANNING_HORIZON, alpha=ALPHA, seed=CMA_SEED,
    )
    cmaes_dense = build_spline_trajectory_3d(cx, cy, cz, cmaes_waypoints, samples_per_segment=SAMPLES_PER_SEGMENT)
    cmaes_xyz = np.asarray(cmaes_dense, dtype=np.float32)

    diffusion_model = get_diffusion_model()
    diff_dense, diff_waypoints = sample_diffusion_trajectory(
        diffusion_model, current_position=(cx, cy, cz),
        current_mean=mu.reshape(X.shape), current_var=diag_var.reshape(X.shape),
        current_heading_velocity=heading_velocity, grid_step=step,
        bounds=(xmin, xmax, ymin, ymax, diffusion_module.Z_MIN, diffusion_module.Z_MAX),
    )
    diff_xyz = diff_dense.T.astype(np.float32)

    imitate_model = get_imitate_model()
    imit_dense, imit_waypoints = sample_imitation_trajectory(
        imitate_model, current_position=(cx, cy, cz),
        current_mean=mu.reshape(X.shape), current_var=diag_var.reshape(X.shape),
        current_heading_velocity=heading_velocity, grid_step=step,
        bounds=(xmin, xmax, ymin, ymax, IMITATE_ZMIN, IMITATE_ZMAX),
    )
    imit_xyz = imit_dense.T.astype(np.float32)

    u_cmaes = utility_along_path(cmaes_xyz[:, :2], xs, ys, utility_grid, step)
    u_diff = utility_along_path(diff_xyz[:, :2], xs, ys, utility_grid, step)
    u_imit = utility_along_path(imit_xyz[:, :2], xs, ys, utility_grid, step)

    return dict(
        utility_grid=utility_grid.astype(np.float32),
        cmaes_xyz=cmaes_xyz, diff_xyz=diff_xyz, imit_xyz=imit_xyz,
        cmaes_waypoints=np.asarray(cmaes_waypoints, dtype=np.float32),
        diff_waypoints=np.asarray(diff_waypoints, dtype=np.float32).T,
        imit_waypoints=np.asarray(imit_waypoints, dtype=np.float32).T,
        u_cmaes=u_cmaes, u_diff=u_diff, u_imit=u_imit,
    )


# Bright, high-contrast colors + black outline stroke so lines stay legible
# against every part of the viridis heatmap (dark purple through yellow).
COLOR_CMAES = "#00E5FF"
COLOR_DIFFUSION = "#FF00E6"
COLOR_IMITATE = "#FFA500"


def plot_situation(selected_map, state, result, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patheffects as pe

    stroke = [pe.withStroke(linewidth=4.5, foreground="black")]
    X, Y = state["X"], state["Y"]
    xmin, xmax, ymin, ymax = state["xmin"], state["xmax"], state["ymin"], state["ymax"]
    cx, cy, cz = state["pose"]

    fig, ax = plt.subplots(figsize=(9, 8.5))
    pcm = ax.pcolormesh(X, Y, result["utility_grid"], cmap="viridis", shading="auto")
    fig.colorbar(pcm, ax=ax, label="Utility (masked posterior variance)")

    ax.plot(result["cmaes_xyz"][:, 0], result["cmaes_xyz"][:, 1], color=COLOR_CMAES, linestyle="--",
            linewidth=3.0, label="CMA-ES", path_effects=stroke, zorder=4)
    ax.plot(result["diff_xyz"][:, 0], result["diff_xyz"][:, 1], color=COLOR_DIFFUSION,
            linewidth=3.0, label="Diffusion", path_effects=stroke, zorder=4)
    ax.plot(result["imit_xyz"][:, 0], result["imit_xyz"][:, 1], color=COLOR_IMITATE,
            linewidth=3.0, label="ImitateTrans", path_effects=stroke, zorder=4)
    ax.scatter([cx], [cy], color="white", edgecolor="black", linewidth=1.5,
               s=180, marker="*", zorder=5, label="Current pose")

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"Map {selected_map}, t={state['wall_time']:.1f}s: same situation, three planners")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    ax.set_aspect("equal")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path
