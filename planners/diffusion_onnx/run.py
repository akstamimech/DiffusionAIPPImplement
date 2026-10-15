"""
Single-map flight with the diffusion planner running the exported ONNX network.

    SELECTED_MAP=5 TIMEALLOTED=200 RUN_SEED=1 python planners/diffusion_onnx/run.py

The simulator (map, belief, camera, drone, scoring, outputs) is the shared `sim` package; the loop is the shared
`aipp` mission. Only the planner is specific to this folder. Settings: see sim/flight.py (SELECTED_MAP, TIMEALLOTED,
WALLCLOCK_SECONDS, ...) plus the planner's own below.

    ONNX_MODEL, ONNX_DEPLOY_JSON   network and its constants (default: new_1200_int8.onnx, new_1200_deploy.json)
    ETA, DDIM_STEPS                sampler (default 0.0 = deterministic, all 20 steps)
    ORT_THREADS                    onnxruntime threads (default: onnxruntime's choice)
    RUN_SEED                       seed of the planner's initial noise (default 1121)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from planners.diffusion_onnx.planner import DiffusionPlanner, PlannerConfig
from sim import FlightSettings, run_flight
from sim.flight import GRID


def main():
    settings = FlightSettings.from_env(
        output_prefix="diffusion_onnx", plot_label="Diffusion",
        selected_map=26, timealloted=4000, execution_chunk=40,
    )
    model = Path(os.environ.get("ONNX_MODEL", str(HERE / "new_1200_int8.onnx")))
    deploy_json = Path(os.environ.get("ONNX_DEPLOY_JSON", str(HERE / "new_1200_deploy.json")))
    eta = float(os.environ.get("ETA", "0.0"))
    ddim_steps = int(os.environ["DDIM_STEPS"]) if os.environ.get("DDIM_STEPS") else None    # None = all T steps
    threads = int(os.environ["ORT_THREADS"]) if os.environ.get("ORT_THREADS") else None      # None = onnxruntime default
    run_seed = int(os.environ.get("RUN_SEED", "1121"))
    np.random.seed(run_seed)

    planner = DiffusionPlanner(model, deploy_json, GRID, PlannerConfig(eta=eta, ddim_steps=ddim_steps),
                               threads=threads, seed=run_seed)
    print(f"ONNX planner: model {model.name}, deploy JSON {deploy_json.name}, T={planner.T}, eta={eta}, "
          f"steps={ddim_steps or planner.T}", flush=True)
    run_flight(planner, settings)


if __name__ == "__main__":
    main()
