"""
Save one confirmed-worth-keeping situation to the curated store:
  results/map{M}_row{R}.npz  - utility grid + all three full (x,y,z) paths
                                + waypoints, so it can be re-plotted later
                                without re-running any model
  plots/map{M}_row{R}.png    - the rendered comparison figure
  results_index.csv          - one summary row appended

Deliberately manual/per-row - only situations someone decided are worth
keeping get in here. The search/evaluate scripts never write to this tier.

Usage: python save_situation.py <map_id> <row> [note text...]
"""
import csv
import sys
from pathlib import Path

import numpy as np

from situation_lib import (
    reconstruct_full_state_at_row, run_three_planners, plot_situation,
    RESULTS_DIR, PLOTS_DIR, INDEX_CSV,
)

INDEX_FIELDS = [
    "map", "row", "wall_time_s", "pose_x", "pose_y", "pose_z",
    "u_cmaes", "u_diff", "u_imit", "diff_ratio", "imit_ratio",
    "npz_path", "png_path", "note",
]


def main():
    selected_map = int(sys.argv[1])
    row = int(sys.argv[2])
    note = " ".join(sys.argv[3:])

    state = reconstruct_full_state_at_row(selected_map, row)
    result = run_three_planners(state)
    diff_ratio = result["u_diff"] / result["u_cmaes"]
    imit_ratio = result["u_imit"] / result["u_cmaes"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    npz_path = RESULTS_DIR / f"map{selected_map}_row{row}.npz"
    np.savez_compressed(
        npz_path,
        map=selected_map, row=row, wall_time=state["wall_time"],
        pose=state["pose"], xs=state["xs"], ys=state["ys"], step=state["step"],
        xmin=state["xmin"], xmax=state["xmax"], ymin=state["ymin"], ymax=state["ymax"],
        utility_grid=result["utility_grid"],
        cmaes_xyz=result["cmaes_xyz"], diff_xyz=result["diff_xyz"], imit_xyz=result["imit_xyz"],
        cmaes_waypoints=result["cmaes_waypoints"], diff_waypoints=result["diff_waypoints"],
        imit_waypoints=result["imit_waypoints"],
        u_cmaes=result["u_cmaes"], u_diff=result["u_diff"], u_imit=result["u_imit"],
    )

    png_path = PLOTS_DIR / f"map{selected_map}_row{row}.png"
    plot_situation(selected_map, state, result, png_path)

    row_exists = INDEX_CSV.exists()
    with INDEX_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if not row_exists:
            writer.writeheader()
        writer.writerow(dict(
            map=selected_map, row=row, wall_time_s=state["wall_time"],
            pose_x=state["pose"][0], pose_y=state["pose"][1], pose_z=state["pose"][2],
            u_cmaes=result["u_cmaes"], u_diff=result["u_diff"], u_imit=result["u_imit"],
            diff_ratio=diff_ratio, imit_ratio=imit_ratio,
            npz_path=str(npz_path.relative_to(RESULTS_DIR.parent)),
            png_path=str(png_path.relative_to(PLOTS_DIR.parent)),
            note=note,
        ))

    print(f"Saved: {npz_path}")
    print(f"Saved: {png_path}")
    print(f"Appended to: {INDEX_CSV}")
    print(f"diff_ratio={diff_ratio:.2f} imit_ratio={imit_ratio:.2f}")


if __name__ == "__main__":
    main()
