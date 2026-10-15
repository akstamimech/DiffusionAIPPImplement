"""
Golden regression: complete closed-loop flights of every planner against recorded results.

Each planner's run script flies a short, fully seeded scenario in a subprocess (one thread, no pacing, no wall-clock
budget). The RMSE / variance trace must match the recorded one (relative tolerance 1e-6) and the executed trajectory
exactly. Run it after any change to aipp/, sim/ or a planner.

    python tests/test_golden_flights.py                          # check every planner
    python tests/test_golden_flights.py --planner cmaes          # check one
    python tests/test_golden_flights.py --update-golden          # re-record (only when a change in behaviour is intended)

The ONNX planner is checked with the fp32 network on purpose: the INT8 network quantizes activations at run time, so a
float-level change in the belief (even a different thread count) can flip a rounding and send a closed-loop flight
down another path (seen: final variance 14.93 vs 14.76 for the same scenario). fp32 depends smoothly on its inputs.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GOLDEN = HERE / "golden"

COMMON_ENV = {
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "ORT_THREADS": "1",
    "WALLCLOCK_SECONDS": "0", "ENFORCE_MIN_STEP_TIME": "0", "SKIP_VIZ": "1",
}

PLANNERS = {
    "diffusion_onnx": {
        "script": "planners/diffusion_onnx/run.py",
        "env": {"ONNX_MODEL": str(ROOT / "planners" / "diffusion_onnx" / "new_1200_fp32.onnx"), "ETA": "0.0"},
        "scenarios": {
            "map5_seed1_90": {"SELECTED_MAP": "5", "RUN_SEED": "1", "TIMEALLOTED": "90"},
            "map12_seed7_130": {"SELECTED_MAP": "12", "RUN_SEED": "7", "TIMEALLOTED": "130", "START_X": "30.0", "START_Y": "60.0"},
        },
    },
    # The goldens below were recorded from the legacy single-map scripts these planners replaced.
    "cmaes": {
        "script": "planners/cmaes/run.py",
        "scenarios": {"map5_45": {"SELECTED_MAP": "5", "TIMEALLOTED": "45"}},
    },
    "realgreedy": {
        "script": "planners/realgreedy/run.py",
        "scenarios": {"map5_45": {"SELECTED_MAP": "5", "TIMEALLOTED": "45"}},
    },
    "greedygradient": {
        "script": "planners/greedygradient/run.py",
        "scenarios": {"map5_45": {"SELECTED_MAP": "5", "TIMEALLOTED": "45"}},
    },
    "lawnmower": {
        "script": "planners/lawnmower/run.py",
        "scenarios": {"map5_60": {"SELECTED_MAP": "5", "TIMEALLOTED": "60"}},
    },
    "diffusion": {
        "script": "planners/diffusion/run.py",
        "env": {"ETA": "0.0"},
        "scenarios": {"map5_45": {"SELECTED_MAP": "5", "TIMEALLOTED": "45", "RUN_SEED": "1"}},
    },
    "imitatetrans": {
        "script": "planners/imitatetrans/run.py",
        "scenarios": {"map5_45": {"SELECTED_MAP": "5", "TIMEALLOTED": "45"}},
    },
}


def run_scenario(planner, name, out_root):
    spec = PLANNERS[planner]
    scenario = spec["scenarios"][name]
    tag = f"{planner}_{name}"
    env = {**os.environ, **COMMON_ENV, **spec.get("env", {}), **scenario, "RESULTS_ROOT": str(out_root), "RUN_OUTPUT_TAG": tag}
    proc = subprocess.run([sys.executable, str(ROOT / spec["script"])], env=env, capture_output=True, text=True, cwd=str(ROOT))
    if proc.returncode != 0:
        raise RuntimeError(f"{planner}/{name} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-2500:]}")
    m = scenario["SELECTED_MAP"]
    run_dir = next(Path(out_root).glob(f"*_map_*_{m}_viz_{tag}"))
    return run_dir / f"map_{m}_rmse_over_time.csv", run_dir / f"map_{m}_executed_trajectory.csv"


def load(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def golden_paths(planner, name):
    d = GOLDEN / planner
    return d / f"{name}_rmse_over_time.csv", d / f"{name}_executed_trajectory.csv"


def compare(planner, name, trace_path, traj_path):
    """Returns (problems, max abs deviation over the RMSE / variance columns)."""
    problems = []
    g_trace_path, g_traj_path = golden_paths(planner, name)
    g_trace, g_traj = load(g_trace_path), load(g_traj_path)
    trace, traj = load(trace_path), load(traj_path)
    label = f"{planner}/{name}"
    if trace.shape != g_trace.shape:
        return [f"{label}: trace shape {trace.shape} != golden {g_trace.shape}"], float("nan")
    if traj.shape != g_traj.shape:
        return [f"{label}: trajectory shape {traj.shape} != golden {g_traj.shape}"], float("nan")
    keep = [0, 2, 3, 4]                                   # timestep, global_rmse, occupied_rmse, global_variance (column 1 is wall time)
    dev = float(np.abs(trace[:, keep] - g_trace[:, keep]).max())
    if not np.allclose(trace[:, keep], g_trace[:, keep], rtol=1e-6, atol=1e-9):
        first = int(np.argmax(np.any(~np.isclose(trace[:, keep], g_trace[:, keep], rtol=1e-6, atol=1e-9), axis=1)))
        problems.append(f"{label}: variance/RMSE trace differs (first at step index {first}; max abs diff {dev:.3e})")
    cols = [0, 2, 3, 4]                                   # timestep, x, y, z (column 1 is wall time)
    if not np.allclose(traj[:, cols], g_traj[:, cols], rtol=0, atol=1e-9):
        first = int(np.argmax(np.any(np.abs(traj[:, cols] - g_traj[:, cols]) > 1e-9, axis=1)))
        problems.append(f"{label}: executed trajectory differs (first at row {first}: {traj[first, cols]} vs golden {g_traj[first, cols]})")
    return problems, dev


class GoldenFlights(unittest.TestCase):
    def test_flights(self):
        for planner, spec in PLANNERS.items():
            for name in spec["scenarios"]:
                with self.subTest(planner=planner, scenario=name), tempfile.TemporaryDirectory() as tmp:
                    trace, traj = run_scenario(planner, name, tmp)
                    problems, _ = compare(planner, name, trace, traj)
                    self.assertEqual(problems, [])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-golden", action="store_true")
    ap.add_argument("--planner", choices=list(PLANNERS))
    ap.add_argument("--scenario")
    a = ap.parse_args()
    failed = False
    for planner, spec in PLANNERS.items():
        if a.planner and a.planner != planner:
            continue
        for name in spec["scenarios"]:
            if a.scenario and a.scenario != name:
                continue
            with tempfile.TemporaryDirectory() as tmp:
                trace, traj = run_scenario(planner, name, tmp)
                if a.update_golden:
                    g_trace, g_traj = golden_paths(planner, name)
                    g_trace.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(trace, g_trace)
                    shutil.copy(traj, g_traj)
                    t = load(trace)
                    print(f"recorded golden for {planner}/{name}: {len(t)} steps, final global variance {t[-1, 4]:.4f}")
                else:
                    problems, dev = compare(planner, name, trace, traj)
                    print(f"{planner}/{name}: " + (f"matches golden (max |diff| over rmse/variance columns = {dev:.2e}; trajectory exact)" if not problems else "DIFFERENT"))
                    for p in problems:
                        print("   ", p)
                    failed |= bool(problems)
    sys.exit(1 if failed else 0)
