"""
Search a CMA-ES flight's belief history for timesteps where the utility mask
splits into >=2 spatially far-apart clusters ("situations" worth testing all
three planners on). Cheap: covariance-diagonal-only replay, no planner runs.

Usage: python search_clusters.py <map_id>
"""
import sys

import numpy as np
from scipy import ndimage

from situation_lib import replay_diag_belief_history, UTILITY_THRESHOLD, BETA
from gaussianprocesstraining import LCB

SIZE_MIN = 15


def main():
    selected_map = int(sys.argv[1])
    mu_hist, diagP_hist, traj, xs, ys, step = replay_diag_belief_history(selected_map)
    print(f"(live LCB={LCB}, threshold={UTILITY_THRESHOLD})")

    n_rows = traj.shape[0]
    ny, nx = len(ys), len(xs)
    candidates = []
    for i in range(1, n_rows):
        # Mirrors importance_filter's branching exactly (respects live
        # LCB/UCB) without constructing a fake dense covariance matrix just
        # to re-extract the diagonal we already have.
        sigma_i = np.sqrt(diagP_hist[i])
        if LCB:
            important_mask = (mu_hist[i] - BETA * sigma_i <= UTILITY_THRESHOLD).reshape(ny, nx)
        else:
            important_mask = (mu_hist[i] + BETA * sigma_i >= UTILITY_THRESHOLD).reshape(ny, nx)

        labeled, n_labels = ndimage.label(important_mask, structure=np.ones((3, 3)))
        if n_labels < 2:
            continue
        sizes = ndimage.sum(important_mask, labeled, index=range(1, n_labels + 1))
        centroids = ndimage.center_of_mass(important_mask, labeled, index=range(1, n_labels + 1))

        qualifying = [k for k, s in enumerate(sizes) if s >= SIZE_MIN]
        if len(qualifying) < 2:
            continue
        qualifying.sort(key=lambda k: sizes[k], reverse=True)
        top2 = qualifying[:2]
        world = []
        for k in top2:
            cy_idx, cx_idx = centroids[k]
            world.append((xs[0] + cx_idx * step, ys[0] + cy_idx * step))
        gap = np.hypot(world[0][0] - world[1][0], world[0][1] - world[1][1])

        candidates.append(dict(
            row=i, wall_time=float(traj[i, 1]), pose=tuple(traj[i, 2:5]),
            n_clusters=len(qualifying), gap=float(gap),
            top2_sizes=(float(sizes[top2[0]]), float(sizes[top2[1]])),
        ))

    print(f"Found {len(candidates)} timesteps with >=2 clusters (min size {SIZE_MIN} cells)\n")
    print("Best candidate per 30s time bin:")
    for lo, hi in [(0, 30), (30, 60), (60, 90), (90, 120), (120, 150), (150, 180)]:
        in_bin = [c for c in candidates if lo <= c["wall_time"] < hi]
        if not in_bin:
            print(f"  [{lo:>3}-{hi:>3}s): none")
            continue
        best = max(in_bin, key=lambda c: c["gap"])
        print(
            f"  [{lo:>3}-{hi:>3}s): row={best['row']:>3} wall={best['wall_time']:>6.1f}s "
            f"pose=({best['pose'][0]:.1f},{best['pose'][1]:.1f},{best['pose'][2]:.1f}) "
            f"n_clusters={best['n_clusters']} gap={best['gap']:.1f} sizes={best['top2_sizes']}"
        )

    candidates.sort(key=lambda c: c["gap"], reverse=True)
    print("\nTop 15 overall by gap:")
    for c in candidates[:15]:
        print(
            f"  row={c['row']:>3} wall={c['wall_time']:>6.1f}s "
            f"pose=({c['pose'][0]:.1f},{c['pose'][1]:.1f},{c['pose'][2]:.1f}) "
            f"n_clusters={c['n_clusters']} gap={c['gap']:.1f} sizes={c['top2_sizes']}"
        )


if __name__ == "__main__":
    main()
