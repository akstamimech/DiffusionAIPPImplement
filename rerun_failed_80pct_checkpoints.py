"""
For the 13 runs that never reached 80% occupied variance reduction within the original
150s cap (1 Diffusion repeat at epoch 10, plus 12 ImitateTrans epochs entirely), re-runs
the SAME (checkpoint, seed) pair with WALLCLOCK_SECONDS extended to EXTENDED_WALLCLOCK,
to find out how long it actually takes. Same seed as the original failing run, so this
reproduces the exact same trajectory for the first 150s and just lets it continue past
that, rather than drawing a fresh random sample.

Parallelized the same way as the other checkpoint sweeps: bounded worker pool,
single-threaded BLAS per subprocess (OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_NUM_THREADS=1) so the
concurrent processes don't oversubscribe cores during their brief compute bursts.

After each run finishes, replays the resulting executed_trajectory.csv (same Kalman-replay
approach as the rest of this analysis) to find the actual wall-clock time it crossed 80%
occupied variance reduction (fraction_left <= 0.20), instead of just reading the script's
own printed summary (which only reports the mission endpoint, not when the threshold was
first crossed).

Writes rerun_failed_80pct_checkpoints_summary.csv.
"""
import csv
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR_DIFFUSION = SCRIPT_DIR / "Diffusion" / "checkpoints"
CHECKPOINT_DIR_IMITATE = SCRIPT_DIR / "checkpoints"
VIZ_DIR = SCRIPT_DIR / "Vizualization"
LOG_DIR = SCRIPT_DIR / "rerun_failed_80pct_logs"
LOG_DIR.mkdir(exist_ok=True)
SUMMARY_CSV = SCRIPT_DIR / "rerun_failed_80pct_checkpoints_summary.csv"

SELECTED_MAP = 59
MAPTYPE = "NAIP"
EXTENDED_WALLCLOCK = 400.0
THRESHOLD = 0.20  # <=20% remaining = 80% reduction
MAX_WORKERS = 4
BLAS_THREAD_ENV = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
}
_write_lock = threading.Lock()

# (planner, epoch, seed) - the exact failing runs, same seeds as the original sweeps.
FAILED_RUNS = [
    ("diffusion", 10, 90003),
    ("imitation", 10, 90001),
    ("imitation", 50, 90001),
    ("imitation", 100, 90001),
    ("imitation", 200, 90001),
    ("imitation", 300, 90001),
    ("imitation", 700, 90001),
    ("imitation", 900, 90001),
    ("imitation", 1000, 90001),
    ("imitation", 1100, 90001),
    ("imitation", 1200, 90001),
    ("imitation", 1300, 90001),
    ("imitation", 1400, 90001),
]


def run_one(planner, epoch, seed):
    run_tag = f"extended_{planner}_epoch{epoch}"
    if planner == "diffusion":
        checkpoint_path = CHECKPOINT_DIR_DIFFUSION / f"sparse_trans_waypoints_epoch_{epoch}.pth"
        script = SCRIPT_DIR / "Diffusionplanner_singlemap.py"
        output_dir = VIZ_DIR / f"diffusion_map_{MAPTYPE}_{SELECTED_MAP}_viz_{run_tag}"
        checkpoint_env_var = "DIFFUSION_CHECKPOINT"
    else:
        checkpoint_path = CHECKPOINT_DIR_IMITATE / f"imitate_trans_waypoints_epoch_{epoch}.pth"
        script = SCRIPT_DIR / "ImitateTrans_singlemap.py"
        output_dir = VIZ_DIR / f"imitate_trans_map_{MAPTYPE}_{SELECTED_MAP}_viz_{run_tag}"
        checkpoint_env_var = "IMITATE_CHECKPOINT"

    log_path = LOG_DIR / f"{run_tag}.log"
    env = os.environ.copy()
    env.update(BLAS_THREAD_ENV)
    env.update({
        checkpoint_env_var: str(checkpoint_path),
        "RUN_SEED": str(seed),
        "RUN_OUTPUT_TAG": run_tag,
        "SKIP_VIZ": "1",
        "WALLCLOCK_SECONDS": str(EXTENDED_WALLCLOCK),
    })

    start = time.time()
    print(f"[{planner} epoch {epoch}] starting (extended to {EXTENDED_WALLCLOCK:.0f}s), log -> {log_path}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            [sys.executable, "-u", str(script)],
            cwd=str(SCRIPT_DIR),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        for line in process.stdout:
            log_file.write(line)
        return_code = process.wait()
    elapsed = time.time() - start

    row = {
        "planner": planner, "epoch": epoch, "seed": seed,
        "return_code": return_code, "wall_clock_elapsed_s": elapsed,
        "output_dir": str(output_dir), "log_path": str(log_path),
    }

    if return_code != 0:
        print(f"[{planner} epoch {epoch}] FAILED (exit {return_code}) after {elapsed:.1f}s", flush=True)
        row["status"] = "failed"
        return row

    traj_path = output_dir / f"map_{SELECTED_MAP}_executed_trajectory.csv"
    result = m.reconstruct_one({"map_id": SELECTED_MAP, "traj_path": str(traj_path)})
    wall_time = result["wall_time_seconds"]
    important_var = result["important_variance"]
    frac_left = important_var / important_var[0]
    hit = np.where(frac_left <= THRESHOLD)[0]

    row["mission_end_wall_time_s"] = float(wall_time[-1])
    if hit.size:
        row["time_to_80pct_reduction_s"] = float(wall_time[hit[0]])
        row["status"] = "reached"
    else:
        row["time_to_80pct_reduction_s"] = ""
        row["status"] = "still_not_reached"

    if row["status"] == "reached":
        status_msg = f"reached 80% at {row['time_to_80pct_reduction_s']:.1f}s"
    else:
        status_msg = f"STILL NOT REACHED (mission ran to {row['mission_end_wall_time_s']:.1f}s)"
    print(f"[{planner} epoch {epoch}] done in {elapsed:.1f}s - {status_msg}", flush=True)
    return row


def write_summary(rows_by_key):
    rows = [rows_by_key[k] for k in sorted(rows_by_key)]
    if not rows:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with SUMMARY_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print(f"Re-running {len(FAILED_RUNS)} failed runs with WALLCLOCK_SECONDS={EXTENDED_WALLCLOCK:.0f}, {MAX_WORKERS}-way parallel", flush=True)
    rows_by_key = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(run_one, planner, epoch, seed): (planner, epoch) for planner, epoch, seed in FAILED_RUNS}
        for future in as_completed(futures):
            row = future.result()
            key = (row["planner"], row["epoch"])
            with _write_lock:
                rows_by_key[key] = row
                write_summary(rows_by_key)

    print(f"\nDone. Wrote {SUMMARY_CSV}")
    for key in sorted(rows_by_key):
        r = rows_by_key[key]
        print(f"  {r['planner']} epoch {r['epoch']}: status={r['status']}, "
              f"time_to_80pct={r.get('time_to_80pct_reduction_s', 'N/A')}")


if __name__ == "__main__":
    main()
