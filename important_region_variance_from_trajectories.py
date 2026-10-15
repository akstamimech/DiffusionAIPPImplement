"""
Reconstructs, from already-saved executed_trajectory.csv files under
results_hpc/planner_outputs, how much variance each planner removed
specifically within "important" regions - grid cells whose GROUND TRUTH
value is on the important side of GROUND_TRUTH_THRESHOLD (below it under
LCB/NAIP, above it under UCB/grf - see gaussianprocesstraining.LCB) - as
opposed to variance removed over the whole map. MAPTYPE/GROUND_TRUTH_THRESHOLD
are env-configurable (default grf/0.5) so this doesn't need hand-editing every
time the map type switches.

Why this is possible without re-running anything: the Kalman/GP covariance
update (kalman_update / covariance_after_sensor in gaussianprocesstraining.py)
is P_new = P - K @ (sensor @ P), which depends only on WHERE the sensor
looked (sensor matrix, from the recorded x,y,z) and the altitude-dependent
noise model - never on the actual measured/observed values. So the full
covariance trajectory can be replayed exactly from the recorded poses alone,
using the true map only to define which cells count as "important".

For tractability (2601-cell/51x51 full grid would take hours across ~200
saved runs), this replay runs on a coarser 4 m analysis grid (every other
point of the original 2 m grid, 26x26 = 676 cells) instead of the original
2 m/2601-cell grid used at collection time. All five planners are replayed
on the identical coarse grid/threshold/noise model, so the comparison
between them stays fair even though absolute variance values will not
exactly match a full-resolution recomputation.
"""
import os
import re
import sys
import time
import multiprocessing as mp
from pathlib import Path

import numpy as np
from scipy.linalg import solve as spd_solve

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from gaussianprocesstraining import (
    initialize_gp,
    noise_model,
    fov_grid_points,
    build_sensor_matrix,
    compress_shared_sensor_rows,
    measurement_noise_covariance,
    LCB,
)

PLANNER_OUTPUTS_ROOT = Path(r"C:\Users\Aksha\OneDrive\Year 6\results_hpc\planner_outputs")
CSV_DIR = SCRIPT_DIR / "csv"
OUT_DIR = PLANNER_OUTPUTS_ROOT / "aggregate_plots" / "important_region_variance"
MAPTYPE = os.environ.get("MAPTYPE", "NAIP")
GROUND_TRUTH_THRESHOLD = float(os.environ.get("UTILITY_THRESHOLD", "0.3"))
COARSEN_FACTOR = 2  # 2m -> 4m analysis grid
ANGLE_OF_VIEW = 60.0

DIR_PATTERN = re.compile(
    r"^(?P<outprefix>[a-zA-Z_]+?)_map_" + re.escape(MAPTYPE) + r"_(?P<mapid>\d+)_viz_hpc_(?P<planner>[a-zA-Z]+)_"
    r"map\d+_repeat(?P<repeat>\d+)_seed(?P<seed>\d+)$"
)

PLANNER_LABELS = {
    "cmaes": "CMA-ES",
    "diffusion": "Diffusion",
    "imitation": "ImitateTrans",
    "realgreedy": "RealGreedy",
    "lawnmower": "Lawnmower",
}


def discover_runs():
    runs = []
    for d in sorted(os.listdir(PLANNER_OUTPUTS_ROOT)):
        full = PLANNER_OUTPUTS_ROOT / d
        if not full.is_dir():
            continue
        m = DIR_PATTERN.match(d)
        if not m:
            continue
        map_id = int(m.group("mapid"))
        traj_path = full / f"map_{map_id}_executed_trajectory.csv"
        if not traj_path.exists():
            continue
        runs.append(
            {
                "planner": m.group("planner"),
                "map_id": map_id,
                "repeat": int(m.group("repeat")),
                "seed": int(m.group("seed")),
                "run_dir": str(full),
                "traj_path": str(traj_path),
            }
        )
    return runs


def build_coarse_grid():
    gp, X_test, mean0, cov0, xs, ys, X, Y, xmin, xmax, ymin, ymax, step = initialize_gp()
    n_full = len(xs)
    keep = list(range(0, n_full, COARSEN_FACTOR))
    xs_c = xs[keep]
    ys_c = ys[keep]
    idx_2d = np.array([yi * n_full + xi for yi in keep for xi in keep])
    cov_c = cov0[np.ix_(idx_2d, idx_2d)]
    # X_test rows in the same (x fastest, y slowest) order as idx_2d above
    coords_c = X_test[idx_2d]
    return xs_c, ys_c, cov_c, coords_c


_GRID_CACHE = None
_MASK_CACHE = {}


def _important_mask_for_map(map_id, coords_c):
    if map_id in _MASK_CACHE:
        return _MASK_CACHE[map_id]
    gt_path = CSV_DIR / f"map_{map_id}_{MAPTYPE}_grid_counts.csv"
    data = np.loadtxt(gt_path, delimiter=",", skiprows=1)
    gx, gy, gval = data[:, 0], data[:, 1], data[:, 2]
    n = coords_c.shape[0]
    values = np.full(n, np.nan)
    for i, (cx, cy) in enumerate(coords_c):
        idx = np.where(np.isclose(gx, cx) & np.isclose(gy, cy))[0]
        if idx.size > 0:
            values[i] = gval[idx[0]]
    if LCB:
        mask = values <= GROUND_TRUTH_THRESHOLD  # LCB/NAIP: "important" = at/below threshold
    else:
        mask = values >= GROUND_TRUTH_THRESHOLD  # UCB/grf: "important" = at/above threshold
    _MASK_CACHE[map_id] = (mask, values)
    return mask, values


def _covariance_step(P, sensor, R):
    if sensor.shape[0] == 0:
        return P
    noise_cov = measurement_noise_covariance(R, sensor.shape[0])
    projected_cov = np.asarray(sensor @ P)
    innovation_cov = np.asarray(sensor @ projected_cov.T) + noise_cov
    try:
        solved = spd_solve(innovation_cov, projected_cov, assume_a="pos")
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(innovation_cov) @ projected_cov
    return P - projected_cov.T @ solved


def _init_worker():
    global _GRID_CACHE
    _GRID_CACHE = build_coarse_grid()


def reconstruct_one(run):
    global _GRID_CACHE
    if _GRID_CACHE is None:
        _GRID_CACHE = build_coarse_grid()
    xs_c, ys_c, cov_c, coords_c = _GRID_CACHE
    mask, _ = _important_mask_for_map(run["map_id"], coords_c)

    traj = np.loadtxt(run["traj_path"], delimiter=",", skiprows=1)
    if traj.ndim == 1:
        traj = traj.reshape(1, -1)

    P = cov_c.copy()
    n_important = int(mask.sum())
    n_total = P.shape[0]

    rows_out = []
    initial_masked = float(np.diag(P)[mask].sum())
    initial_global = float(np.trace(P))
    rows_out.append((0.0, 0.0, initial_masked, initial_global))

    for row in traj[1:]:
        ts, wall_t, x, y, z = row[0], row[1], row[2], row[3], row[4]
        fov = fov_grid_points(x, y, z, xs_c, ys_c, angle_of_view=ANGLE_OF_VIEW)
        if not fov:
            diag = np.diag(P)
            rows_out.append((ts, wall_t, float(diag[mask].sum()), float(diag.sum())))
            continue
        sensor, block_ids = build_sensor_matrix(fov, z, xs_c, ys_c, return_block_ids=True)
        if sensor.shape[0] == 0:
            diag = np.diag(P)
            rows_out.append((ts, wall_t, float(diag[mask].sum()), float(diag.sum())))
            continue
        R = noise_model(z)
        sensor_c, _, R_c = compress_shared_sensor_rows(sensor, None, R, block_ids)
        P = _covariance_step(P, sensor_c, R_c)
        diag = np.diag(P)
        rows_out.append((ts, wall_t, float(diag[mask].sum()), float(diag.sum())))

    arr = np.array(rows_out, dtype=float)
    result = dict(run)
    result["n_important_cells"] = n_important
    result["n_total_cells"] = n_total
    result["timestep"] = arr[:, 0]
    result["wall_time_seconds"] = arr[:, 1]
    result["important_variance"] = arr[:, 2]
    result["global_variance"] = arr[:, 3]
    return result


def main():
    runs = discover_runs()
    print(f"Discovered {len(runs)} runs", flush=True)
    by_planner = {}
    for r in runs:
        by_planner.setdefault(r["planner"], []).append(r["map_id"])
    for p, maps in sorted(by_planner.items()):
        print(f"  {p}: {len(maps)} runs, maps {sorted(set(maps))}", flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    n_workers = min(10, mp.cpu_count())
    with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
        results = []
        for i, res in enumerate(pool.imap_unordered(reconstruct_one, runs)):
            results.append(res)
            if (i + 1) % 10 == 0 or (i + 1) == len(runs):
                print(f"  processed {i+1}/{len(runs)} runs, elapsed {time.time()-t0:.1f}s", flush=True)
    print(f"Total replay time: {time.time()-t0:.1f}s", flush=True)

    # ---- write long-format per-timestep CSV ----
    long_path = OUT_DIR / "important_region_variance_timeseries.csv"
    with long_path.open("w", newline="") as f:
        f.write("planner,map_id,repeat,seed,timestep,wall_time_seconds,important_variance,global_variance,n_important_cells,n_total_cells\n")
        for r in results:
            planner_label = PLANNER_LABELS.get(r["planner"], r["planner"])
            for ts, wt, iv, gv in zip(r["timestep"], r["wall_time_seconds"], r["important_variance"], r["global_variance"]):
                f.write(
                    f"{planner_label},{r['map_id']},{r['repeat']},{r['seed']},"
                    f"{ts:.1f},{wt:.6f},{iv:.6f},{gv:.6f},{r['n_important_cells']},{r['n_total_cells']}\n"
                )
    print(f"Wrote {long_path}", flush=True)

    # ---- per-run summary (initial/final/AUC) ----
    def trapz_mean(y, x):
        if len(y) <= 1 or x[-1] <= x[0]:
            return float(y[-1])
        return float(np.trapz(y, x) / (x[-1] - x[0]))

    summary_rows = []
    for r in results:
        planner_label = PLANNER_LABELS.get(r["planner"], r["planner"])
        ts = r["timestep"]
        iv = r["important_variance"]
        gv = r["global_variance"]
        summary_rows.append(
            {
                "planner": planner_label,
                "map_id": r["map_id"],
                "repeat": r["repeat"],
                "n_timesteps": len(ts) - 1,
                "n_important_cells": r["n_important_cells"],
                "n_total_cells": r["n_total_cells"],
                "important_variance_initial": iv[0],
                "important_variance_final": iv[-1],
                "important_variance_pct_reduced": 100.0 * (iv[0] - iv[-1]) / iv[0] if iv[0] > 0 else float("nan"),
                "important_variance_auc_timestep": trapz_mean(iv, ts),
                "global_variance_initial": gv[0],
                "global_variance_final": gv[-1],
                "global_variance_pct_reduced": 100.0 * (gv[0] - gv[-1]) / gv[0] if gv[0] > 0 else float("nan"),
                "global_variance_auc_timestep": trapz_mean(gv, ts),
            }
        )

    per_run_path = OUT_DIR / "important_region_variance_per_run.csv"
    fieldnames = list(summary_rows[0].keys())
    import csv as csv_mod
    with per_run_path.open("w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Wrote {per_run_path}", flush=True)

    # ---- per (planner, map) then per-planner aggregate ----
    import collections
    per_planner_map = collections.defaultdict(list)
    for row in summary_rows:
        per_planner_map[(row["planner"], row["map_id"])].append(row)

    agg_rows = []
    for (planner, map_id), rows in sorted(per_planner_map.items()):
        agg_rows.append(
            {
                "planner": planner,
                "map_id": map_id,
                "n_repeats": len(rows),
                "important_variance_final_mean": np.mean([r["important_variance_final"] for r in rows]),
                "important_variance_pct_reduced_mean": np.mean([r["important_variance_pct_reduced"] for r in rows]),
                "important_variance_auc_mean": np.mean([r["important_variance_auc_timestep"] for r in rows]),
                "global_variance_final_mean": np.mean([r["global_variance_final"] for r in rows]),
                "global_variance_pct_reduced_mean": np.mean([r["global_variance_pct_reduced"] for r in rows]),
                "global_variance_auc_mean": np.mean([r["global_variance_auc_timestep"] for r in rows]),
            }
        )
    per_planner_map_path = OUT_DIR / "important_region_variance_per_planner_map.csv"
    with per_planner_map_path.open("w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)
    print(f"Wrote {per_planner_map_path}", flush=True)

    per_planner = collections.defaultdict(list)
    for row in summary_rows:
        per_planner[row["planner"]].append(row)

    final_rows = []
    for planner, rows in sorted(per_planner.items()):
        maps_covered = sorted(set(r["map_id"] for r in rows))
        final_rows.append(
            {
                "planner": planner,
                "n_runs": len(rows),
                "maps_covered": " ".join(str(m) for m in maps_covered),
                "important_variance_final_mean": np.mean([r["important_variance_final"] for r in rows]),
                "important_variance_pct_reduced_mean": np.mean([r["important_variance_pct_reduced"] for r in rows]),
                "important_variance_auc_mean": np.mean([r["important_variance_auc_timestep"] for r in rows]),
                "global_variance_final_mean": np.mean([r["global_variance_final"] for r in rows]),
                "global_variance_pct_reduced_mean": np.mean([r["global_variance_pct_reduced"] for r in rows]),
                "global_variance_auc_mean": np.mean([r["global_variance_auc_timestep"] for r in rows]),
            }
        )
    final_path = OUT_DIR / "important_region_variance_comparison.csv"
    with final_path.open("w", newline="") as f:
        writer = csv_mod.DictWriter(f, fieldnames=list(final_rows[0].keys()))
        writer.writeheader()
        writer.writerows(final_rows)
    print(f"Wrote {final_path}", flush=True)

    for row in final_rows:
        print(
            f"{row['planner']:>14}: n_runs={row['n_runs']:>3} maps=[{row['maps_covered']}] "
            f"important_var final={row['important_variance_final_mean']:.4f} "
            f"(-{row['important_variance_pct_reduced_mean']:.1f}%) "
            f"AUC={row['important_variance_auc_mean']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
