"""
Single-map flight with the imitation planner running the exported ONNX network.

    SELECTED_MAP=5 TIMEALLOTED=200 python planners/imitatetrans_onnx/run.py

The simulator (map, belief, camera, drone, scoring, outputs) is the shared `sim` package; the loop is the shared
`aipp` mission. Only the planner is specific to this folder. Settings: see sim/flight.py (SELECTED_MAP, TIMEALLOTED,
WALLCLOCK_SECONDS, ...) plus the planner's own below.

    ONNX_MODEL, ONNX_DEPLOY_JSON   network and its constants (default: new_imitate_1200_int8.onnx, new_imitate_1200_deploy.json)
    ORT_THREADS                    onnxruntime threads (default: onnxruntime's choice)
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                      # scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planners.imitatetrans_onnx.planner import ImitationOnnxPlanner
from sim import FlightSettings, run_flight
from sim.flight import GRID


def main():
    settings = FlightSettings.from_env(
        output_prefix="imitate_trans_onnx", plot_label="Imitation (ONNX)",
        selected_map=55, timealloted=3000, execution_chunk=40,
    )
    model = Path(os.environ.get("ONNX_MODEL", str(HERE / "new_imitate_1200_int8.onnx")))
    deploy_json = Path(os.environ.get("ONNX_DEPLOY_JSON", str(HERE / "new_imitate_1200_deploy.json")))
    threads = int(os.environ["ORT_THREADS"]) if os.environ.get("ORT_THREADS") else None      # None = onnxruntime default

    planner = ImitationOnnxPlanner(model, deploy_json, GRID, threads=threads)
    print(f"ONNX planner: model {model.name}, deploy JSON {deploy_json.name}, 1 network call per plan", flush=True)
    run_flight(planner, settings)


if __name__ == "__main__":
    main()
