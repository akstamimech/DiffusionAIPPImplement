"""
Occupied variance AUC (wall-clock integrated, capped at 150s, same convention as
plot_checkpoint_sweep_occupied_variance_auc.py) for the new checkpoint_sweep_imitatetrans_
150s_v2_summary.csv runs - one deterministic run per checkpoint, so no repeat averaging.
"""
import csv
from pathlib import Path

import numpy as np

import important_region_variance_from_trajectories as m

SCRIPT_DIR = Path(__file__).resolve().parent
SUMMARY_CSV = SCRIPT_DIR / "checkpoint_sweep_imitatetrans_150s_v2_summary.csv"
SELECTED_MAP = 59
WALLCLOCK_CAP = 150.0


def occupied_variance_auc(run):
    result = m.reconstruct_one(run)
    wall_time = result["wall_time_seconds"]
    important_var = result["important_variance"]
    within_cap = wall_time <= WALLCLOCK_CAP
    wt = wall_time[within_cap]
    iv = important_var[within_cap]
    return float(np.trapezoid(iv, wt)) if wt.size >= 2 else 0.0


def main():
    with SUMMARY_CSV.open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["status"] == "ok"]
    rows.sort(key=lambda r: int(r["epoch"]))

    out_rows = []
    for r in rows:
        epoch = int(r["epoch"])
        traj_path = Path(r["output_dir"]) / f"map_{SELECTED_MAP}_executed_trajectory.csv"
        run = {"map_id": SELECTED_MAP, "traj_path": str(traj_path)}
        auc = occupied_variance_auc(run)
        out_rows.append({"epoch": epoch, "occupied_variance_auc_150s": auc})
        print(f"epoch {epoch:>5}: occupied_variance_auc = {auc:.2f}", flush=True)

    out_csv = SCRIPT_DIR / "checkpoint_sweep_imitatetrans_150s_v2_occupied_variance_auc.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "occupied_variance_auc_150s"])
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
