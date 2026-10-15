import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPTS_DIR = Path(r"C:\Users\Aksha\OneDrive\Year 6\Thesis\scripts")
sys.path.insert(0, str(SCRIPTS_DIR))

from Diffusion.DPPOTRAINING import (
    build_environment, _load_checkpoint, DEFAULT_CHECKPOINT,
    collect_rollouts, _rollout_trajectory, SENSORNOISE_SEED,
)
import Diffusion.threeDSparseTransDiffusion as diffusion
from gaussianprocesstraining import build_spline_trajectory_3d

env = build_environment()
_load_checkpoint(DEFAULT_CHECKPOINT, diffusion.device)

buffer = collect_rollouts(env, num_trajectories=4, base_seed=0)

TRAJ_ID = 2
mask = buffer.trajectory_id == TRAJ_ID
terminal_row = mask & (buffer.t == 0)
x_final = buffer.x_k_minus_1[terminal_row][0]  # the actual fully-denoised action for this rollout

traj_world = diffusion.extract_control_waypoints(x_final).detach().cpu().numpy().T  # [8, 3]
dense_path = np.array(build_spline_trajectory_3d(env.cx0, env.cy0, env.cz0, traj_world, samples_per_segment=5))

# same seed collect_rollouts used internally for this trajectory's reward, so the "end" map
# below matches exactly what buffer.reward[terminal_row] was actually scored against.
rng = np.random.default_rng(SENSORNOISE_SEED + env.map_id + TRAJ_ID)
mu_after, P_after = _rollout_trajectory(traj_world, env, rng)

grid_shape = (len(env.ys), len(env.xs))
mean_before = env.mu0.reshape(grid_shape)
mean_after = mu_after.reshape(grid_shape)
extent = (env.xmin, env.xmax, env.ymin, env.ymax)

vmin = min(mean_before.min(), mean_after.min())
vmax = max(mean_before.max(), mean_after.max())

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
for ax, mean_map, title in zip(
    axes, [mean_before, mean_after], ["Start (GP prior mean)", "End (post-flight mean)"]
):
    im = ax.imshow(mean_map, origin="lower", extent=extent, cmap="viridis", vmin=vmin, vmax=vmax)
    ax.plot(dense_path[:, 0], dense_path[:, 1], color="red", linewidth=2, label="trajectory")
    ax.scatter([env.cx0], [env.cy0], color="white", edgecolor="black", marker="*", s=160, label="start pos", zorder=5)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.legend(loc="upper right")

fig.colorbar(im, ax=axes, label="map mean", shrink=0.8)
reward = buffer.reward[terminal_row].item()
fig.suptitle(f"Rollout {TRAJ_ID} (map_id={env.map_id}) - reward={reward:.4f}")

output_path = Path(__file__).parent / "rollout_viz.png"
fig.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"Saved to {output_path}")
