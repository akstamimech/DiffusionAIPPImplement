"""
Captures the actual reverse-diffusion denoising sequence for ONE diffusion
call - the drone's very first plan, from the initial belief at the mission
start pose, on map 11 (NAIP) - every intermediate state between pure noise
and the resolved plan (T=20 total diffusion steps here, so "every
intermediate" is a small, manageable number, not the ~1000 a typical
image-diffusion model would have).

This is deliberately just the one trajectory that's actually being diffused:
the final checkpoint (step 20, fully resolved) IS the "final trajectory" for
the cover image - there's no separate, independently-sampled longer flight
appended after it. That's what keeps the echoes and the final path
consistent by construction: they're the same sample, at different points in
its own denoising process, not two different samples that happen to start at
the same place (which is what an earlier version of this did, and why the
final path and the last echo visibly didn't line up - Diffusion's sampling
noise isn't seeded, so two separate sampling calls never produce the same
plan even from identical conditioning).

diffusion.ddim_sample() (threeDSparseTransDiffusion.py) already builds this
whole sequence internally (samples_list) but only returns the last step; this
script doesn't touch that file, it just mirrors the same loop locally (same
ddim_sample_timestep primitive, same conditioning construction as
Diffusionplanner_singlemap.sample_diffusion_trajectory) and keeps every
intermediate x instead of discarding them.

Each captured x (still in the model's normalized space, including the very
noisy early ones) is pushed through the SAME extract_control_waypoints ->
pytorch_cubic_spline postprocessing the real planner uses - for early, noisy
x this produces a chaotic, scattered path; for the final x it's the actual
plan the drone would fly. That progression is the point.
"""
import os

os.environ.setdefault("SELECTED_MAP", "11")
os.environ.setdefault("MAPTYPE", "NAIP")
os.environ.setdefault("UTILITY_THRESHOLD", "0.3")
os.environ.setdefault("DIFFUSION_CHECKPOINT", "Diffusion/checkpoints/UNIMODAL_NAIP_FINAL_DIFF_1000.pth")
os.environ.setdefault("SKIP_VIZ", "1")

import sys
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
DIFFUSION_DIR = SCRIPT_DIR / "Diffusion"
sys.path.insert(0, str(DIFFUSION_DIR))

import Diffusionplanner_singlemap as sim
from gaussianprocesstraining import initialize_gp
from sample_3d_sparse_trans_diffusion import diffusion

OUT_DIR = SCRIPT_DIR / "hero_shot_data"
OUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_STEPS = list(range(21))  # out of T=20; every step, 0=pure noise, 20=final clean plan


@torch.no_grad()
def denoising_sequence(model, current_position, current_mean, current_var, current_heading_velocity=None):
    """Mirrors Diffusionplanner_singlemap.sample_diffusion_trajectory's
    conditioning setup, but calls diffusion.ddim_sample_timestep directly in
    a local loop so every intermediate x is kept."""
    model.eval()
    diffusion.model = model

    current_position_world = torch.as_tensor(current_position, dtype=torch.float32, device=diffusion.device).view(1, 3)
    current_position_model = diffusion.normalize_xyz(current_position_world)

    current_mean_t = torch.tensor(current_mean, dtype=torch.float32, device=diffusion.device)
    current_var_t = torch.tensor(current_var, dtype=torch.float32, device=diffusion.device)

    mean_map = (current_mean_t - diffusion.mean_center.to(diffusion.device)) / diffusion.mean_scale.to(diffusion.device)
    total_variance_condition = diffusion.normalize_total_variance(
        current_var_t.reshape(1, -1).sum(dim=1, keepdim=True)
    ).to(diffusion.device)
    var_map = (current_var_t - diffusion.var_center.to(diffusion.device)) / diffusion.var_scale.to(diffusion.device)
    marker_map = diffusion.make_position_marker_maps(
        current_position_world[:, :2].cpu(), grid_size=51, marker_radius=2,
    ).to(diffusion.device)[0, 0]
    meanvarmarker_map = torch.stack([mean_map, var_map, marker_map], dim=0).unsqueeze(0)

    if current_heading_velocity is None:
        initial_heading_velocity = torch.zeros((1, 3), dtype=torch.float32, device=diffusion.device)
    else:
        chv = torch.as_tensor(current_heading_velocity, dtype=torch.float32, device=diffusion.device).view(1, 3)
        initial_heading_velocity = diffusion.normalize_xyz_displacement(chv)

    x = torch.randn((1, *diffusion.TARGET_SHAPE), device=diffusion.device)
    T = diffusion.T
    schedule = list(range(T - 1, -1, -1))

    captured = {0: x.clone()}  # step 0 = pure noise, before any denoising
    for step_idx, i in enumerate(schedule):
        prev_i = schedule[step_idx + 1] if step_idx + 1 < len(schedule) else -1
        t = torch.full((x.shape[0],), i, dtype=torch.long, device=diffusion.device)
        t_prev = torch.full((x.shape[0],), prev_i, dtype=torch.long, device=diffusion.device)
        x, _, _ = diffusion.ddim_sample_timestep(
            x, t, t_prev, meanvarmarker_map, current_position_model,
            initial_heading_velocity, total_variance_condition,
            clip_x0=True, eta=0.0, sigma_prob_min=0.1,
        )
        captured[step_idx + 1] = x.clone()

    paths = {}
    sparse = {}
    for step in CHECKPOINT_STEPS:
        control_waypoints = diffusion.extract_control_waypoints(captured[step][0])
        dense_traj = diffusion.pytorch_cubic_spline(
            control_waypoints, current_position=current_position_world[0],
        )[0]
        paths[step] = dense_traj.detach().cpu().numpy()  # [3, N]
        sparse[step] = control_waypoints.detach().cpu().numpy()  # [3, 8]
    return paths, sparse


def main():
    diffusion_model = sim.load_diffusion_model(checkpoint_path=sim.diffusion_path)
    print(f"Loaded checkpoint: {sim.diffusion_path}")

    gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()
    mean_prior = np.full(X_test.shape[0], sim.utility_threshold - 0.1)
    mu = mean_prior.copy()
    P = cov.copy()

    cx, cy, cz = sim.START_X, sim.START_Y, sim.INIT_ALTITUDE
    paths, sparse = denoising_sequence(
        diffusion_model,
        current_position=(cx, cy, cz),
        current_mean=mu.reshape(X.shape),
        current_var=np.diag(P).reshape(X.shape),
        current_heading_velocity=None,
    )

    for step, arr in paths.items():
        out_path = OUT_DIR / f"map_11_denoising_step_{step:02d}_of_20.csv"
        np.savetxt(out_path, arr.T, delimiter=",", header="x,y,z", comments="")
        wp = sparse[step]
        wp_path = OUT_DIR / f"map_11_denoising_step_{step:02d}_of_20_sparse.csv"
        np.savetxt(wp_path, wp.T, delimiter=",", header="x,y,z", comments="")
        print(f"step {step:>2}/20: {arr.shape[1]} dense points, {wp.shape[1]} sparse -> {out_path.name}")

    print("\nThe final trajectory for the cover should be step 20's path "
          "(map_11_denoising_step_20_of_20.csv) directly - not a separately "
          "generated flight.")


if __name__ == "__main__":
    main()
