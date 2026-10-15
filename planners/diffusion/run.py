"""
Single-map flight with the diffusion planner (PyTorch network).

    SELECTED_MAP=5 TIMEALLOTED=200 RUN_SEED=1 python planners/diffusion/run.py

Simulator and loop are the shared `sim` / `aipp`; settings: see sim/flight.py plus the planner's own below.

    DIFFUSION_CHECKPOINT   network weights (default Diffusion/checkpoints/new_1200.pth)
    DIFFUSION_DATASET_PATH dataset the belief normalization constants are derived from (default HIGHOPT_DATASET_NAIP_141026.pt)
    ETA                    DDIM stochasticity (default 0.0 = deterministic given the initial noise)
    RUN_SEED               seeds numpy and torch (unset: not seeded)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]       # scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from planners.diffusion.planner import DEFAULT_CHECKPOINT, DiffusionPlanner, import_diffusion_module
from sim import FlightSettings, run_flight
from sim.flight import GRID


def main():
    settings = FlightSettings.from_env(
        output_prefix="diffusion", plot_label="Diffusion",
        selected_map=26, timealloted=4000, execution_chunk=40,
    )
    module = import_diffusion_module()          # loads the dataset the normalization comes from; seed after, as always
    run_seed = os.environ.get("RUN_SEED")
    if run_seed is not None:
        np.random.seed(int(run_seed))
        torch.manual_seed(int(run_seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(run_seed))
    planner = DiffusionPlanner(
        GRID,
        checkpoint_path=Path(os.environ.get("DIFFUSION_CHECKPOINT", str(DEFAULT_CHECKPOINT))),
        eta=float(os.environ.get("ETA", "0.0")),
        module=module,
    )
    run_flight(planner, settings)


if __name__ == "__main__":
    main()
