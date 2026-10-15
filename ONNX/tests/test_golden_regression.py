"""
End-to-end golden regression for ONNX/DIFFUSIONPLANNER_ONNX.py.

Runs the planner script on fixed scenarios (map, start, seeds, steps) and compares the per-step belief
variance / RMSE trace and the executed 3D trajectory with recorded golden files. The refactor into
ONNX/aipp/ must not change a single executed waypoint or the variance trace.

    python ONNX/tests/test_golden_regression.py                 # compare against the golden files
    python ONNX/tests/test_golden_regression.py --update-golden # (re)record them from the current code
    python -m unittest ONNX.tests.test_golden_regression        # same comparison via unittest

Determinism: one BLAS thread, one onnxruntime thread, fixed RUN_SEED / SENSORNOISE_SEED, ETA=0, no wall-clock
limits and no artificial step sleeps, so the run depends only on the code. Golden files are exact on the machine
they were recorded on; other CPUs may differ in the last float digits, hence the small tolerances.
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
MAIN = HERE.parent / "DIFFUSIONPLANNER_ONNX.py"
GOLDEN = HERE / "golden"

# The regression runs the fp32 ONNX model on purpose: the INT8 model quantizes activations at run time, so a float-level
# change in the belief (even a different thread count) can flip an int8 rounding and send the closed loop down another
# path (seen: final variance 14.93 vs 14.76 for the same scenario). fp32 depends smoothly on its inputs.
COMMON_ENV = {
    "ONNX_MODEL": str(HERE.parent / "new_1200_fp32.onnx"),
    "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "ORT_THREADS": "1",
    "WALLCLOCK_SECONDS": "0", "ENFORCE_MIN_STEP_TIME": "0", "SKIP_VIZ": "1", "ETA": "0.0",
}
SCENARIOS = {
    "map5_seed1_90": {"SELECTED_MAP": "5", "RUN_SEED": "1", "TIMEALLOTED": "90"},
    "map12_seed7_130": {"SELECTED_MAP": "12", "RUN_SEED": "7", "TIMEALLOTED": "130", "START_X": "30.0", "START_Y": "60.0"},
}


def run_scenario(name, out_root):
    env = {**os.environ, **COMMON_ENV, **SCENARIOS[name], "RESULTS_ROOT": str(out_root), "RUN_OUTPUT_TAG": name}
    proc = subprocess.run([sys.executable, str(MAIN)], env=env, capture_output=True, text=True, cwd=str(MAIN.parent.parent))
    if proc.returncode != 0:
        raise RuntimeError(f"scenario {name} failed:\n{proc.stdout[-1500:]}\n{proc.stderr[-1500:]}")
    run_dir = next(Path(out_root).glob(f"diffusion_onnx_map_*_{SCENARIOS[name]['SELECTED_MAP']}_viz_{name}"))
    m = SCENARIOS[name]["SELECTED_MAP"]
    return run_dir / f"map_{m}_rmse_over_time.csv", run_dir / f"map_{m}_executed_trajectory.csv"


def load(path):
    return np.loadtxt(path, delimiter=",", skiprows=1)


def compare(name, trace_path, traj_path):
    """Returns a list of problems (empty = identical within tolerance)."""
    problems = []
    g_trace, g_traj = load(GOLDEN / f"{name}_rmse_over_time.csv"), load(GOLDEN / f"{name}_executed_trajectory.csv")
    trace, traj = load(trace_path), load(traj_path)
    if trace.shape != g_trace.shape:
        return [f"{name}: trace shape {trace.shape} != golden {g_trace.shape}"]
    if traj.shape != g_traj.shape:
        return [f"{name}: trajectory shape {traj.shape} != golden {g_traj.shape}"]
    keep = [0, 2, 3, 4]                                   # timestep, global_rmse, occupied_rmse, global_variance (column 1 is wall time)
    if not np.allclose(trace[:, keep], g_trace[:, keep], rtol=1e-6, atol=1e-9):
        bad = np.abs(trace[:, keep] - g_trace[:, keep]).max(axis=0)
        first = int(np.argmax(np.any(~np.isclose(trace[:, keep], g_trace[:, keep], rtol=1e-6, atol=1e-9), axis=1)))
        problems.append(f"{name}: variance/RMSE trace differs (first at step index {first}; max abs diff per column {bad})")
    cols = [0, 2, 3, 4]                                   # timestep, x, y, z (column 1 is wall time)
    if not np.allclose(traj[:, cols], g_traj[:, cols], rtol=0, atol=1e-9):
        first = int(np.argmax(np.any(np.abs(traj[:, cols] - g_traj[:, cols]) > 1e-9, axis=1)))
        problems.append(f"{name}: executed trajectory differs (first at row {first}: {traj[first, cols]} vs golden {g_traj[first, cols]})")
    return problems


class GoldenRegression(unittest.TestCase):
    def test_scenarios(self):
        for name in SCENARIOS:
            with self.subTest(scenario=name), tempfile.TemporaryDirectory() as tmp:
                trace, traj = run_scenario(name, tmp)
                self.assertEqual(compare(name, trace, traj), [])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--update-golden", action="store_true")
    ap.add_argument("--only", choices=list(SCENARIOS))
    a = ap.parse_args()
    names = [a.only] if a.only else list(SCENARIOS)
    GOLDEN.mkdir(parents=True, exist_ok=True)
    failed = False
    for name in names:
        with tempfile.TemporaryDirectory() as tmp:
            trace, traj = run_scenario(name, tmp)
            if a.update_golden:
                shutil.copy(trace, GOLDEN / f"{name}_rmse_over_time.csv"); shutil.copy(traj, GOLDEN / f"{name}_executed_trajectory.csv")
                t = load(trace)
                print(f"recorded golden for {name}: {len(t)} steps, final global variance {t[-1, 4]:.4f}")
            else:
                problems = compare(name, trace, traj)
                g, t = load(GOLDEN / f"{name}_rmse_over_time.csv"), load(trace)
                dev = np.abs(t[:, [2, 3, 4]] - g[:, [2, 3, 4]]).max() if t.shape == g.shape else float("nan")
                print(f"{name}: " + (f"matches golden (max |diff| over rmse/variance columns = {dev:.2e}; trajectory exact)" if not problems else "DIFFERENT"))
                for p in problems:
                    print("   ", p)
                failed |= bool(problems)
    sys.exit(1 if failed else 0)
