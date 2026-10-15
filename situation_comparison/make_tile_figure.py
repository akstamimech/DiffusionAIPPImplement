"""
Combine several saved situations (results/map<M>_row<R>.npz) into one N-panel
tile figure with a single shared colorbar - for pasting into the thesis.
Reads only the curated, saved .npz bundles; never re-runs any planner.

Usage: python make_tile_figure.py <out.png> map<M>_row<R> [map<M>_row<R> ...]
Example: python make_tile_figure.py tile.png map55_row237 map56_row230 map128_row217
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

from situation_lib import RESULTS_DIR, COLOR_CMAES, COLOR_DIFFUSION, COLOR_IMITATE

STROKE = [pe.withStroke(linewidth=4.0, foreground="black")]


def main():
    out_path = Path(sys.argv[1])
    names = sys.argv[2:]
    n = len(names)

    bundles = []
    for name in names:
        d = np.load(RESULTS_DIR / f"{name}.npz")
        bundles.append(d)

    vmin = 0.0
    vmax = max(float(d["utility_grid"].max()) for d in bundles)

    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 5.0), squeeze=False)
    axes = axes[0]

    pcm = None
    for ax, d, name in zip(axes, bundles, names):
        xs, ys, step = d["xs"], d["ys"], float(d["step"])
        xmin, xmax, ymin, ymax = float(d["xmin"]), float(d["xmax"]), float(d["ymin"]), float(d["ymax"])
        X, Y = np.meshgrid(xs, ys)
        pcm = ax.pcolormesh(X, Y, d["utility_grid"], cmap="viridis", shading="auto", vmin=vmin, vmax=vmax)

        ax.plot(d["cmaes_xyz"][:, 0], d["cmaes_xyz"][:, 1], color=COLOR_CMAES, linestyle="--",
                linewidth=2.6, label="CMA-ES", path_effects=STROKE, zorder=4)
        ax.plot(d["diff_xyz"][:, 0], d["diff_xyz"][:, 1], color=COLOR_DIFFUSION,
                linewidth=2.6, label="Diffusion", path_effects=STROKE, zorder=4)
        ax.plot(d["imit_xyz"][:, 0], d["imit_xyz"][:, 1], color=COLOR_IMITATE,
                linewidth=2.6, label="ImitateTrans", path_effects=STROKE, zorder=4)
        cx, cy = float(d["pose"][0]), float(d["pose"][1])
        ax.scatter([cx], [cy], color="white", edgecolor="black", linewidth=1.3,
                   s=140, marker="*", zorder=5, label="Current pose")

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_xlabel("x (m)")
        ax.set_title(f"Map {int(d['map'])}, t={float(d['wall_time']):.1f}s")
        ax.set_aspect("equal")

    axes[0].set_ylabel("y (m)")
    for ax in axes[1:]:
        ax.set_yticklabels([])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=True, bbox_to_anchor=(0.5, -0.02))

    fig.subplots_adjust(right=0.9, bottom=0.18, wspace=0.08)
    cbar_ax = fig.add_axes([0.92, 0.18, 0.015, 0.7])
    fig.colorbar(pcm, cax=cbar_ax, label="Utility (masked posterior variance)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
