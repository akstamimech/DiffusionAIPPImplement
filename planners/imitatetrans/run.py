"""
Single-map flight with the imitation (ImitateTrans) baseline.

    SELECTED_MAP=5 TIMEALLOTED=200 python planners/imitatetrans/run.py

Simulator and loop are the shared `sim` / `aipp`; settings: see sim/flight.py plus the planner's own below.

    IMITATE_CHECKPOINT     network weights (default checkpoints/imitate_trans_waypoints_epoch_1500_headingremoved.pth)
    IMITATE_DATASET_PATH   dataset the belief normalization constants are derived from (default dataset_grf_60.pt)
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]       # scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planners.imitatetrans.planner import DEFAULT_CHECKPOINT, ImitationPlanner
from sim import FlightSettings, run_flight
from sim.flight import GRID


def main():
    settings = FlightSettings.from_env(
        output_prefix="imitate_trans", plot_label="Imitation",
        selected_map=55, timealloted=3000, execution_chunk=40,
    )
    planner = ImitationPlanner(GRID, checkpoint_path=Path(os.environ.get("IMITATE_CHECKPOINT", str(DEFAULT_CHECKPOINT))))
    run_flight(planner, settings)


if __name__ == "__main__":
    main()
