"""
Single-map flight with the "real greedy" planner: the CMA-ES planner with a one-waypoint horizon (so it only ever
chases the single next-best point) and a short execution chunk (replan every 5 steps).

    SELECTED_MAP=5 TIMEALLOTED=200 python planners/realgreedy/run.py

Same environment variables as planners/cmaes/run.py; only the defaults differ (horizon 1, chunk 5).
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]       # scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planners.cmaes.planner import CMAESPlanner
from sim import FlightSettings, run_flight
from sim.flight import GRID


def main():
    settings = FlightSettings.from_env(
        output_prefix="greedy", plot_label="",
        selected_map=41, timealloted=3000, execution_chunk=5,
    )
    planner = CMAESPlanner(
        GRID,
        utility_threshold=settings.utility_threshold,
        beta=settings.beta,
        planning_horizon=int(os.environ.get("PLANNING_HORIZON", "1")),
        cma_seed=int(os.environ.get("CMA_SEED", "55")),
    )
    run_flight(planner, settings)


if __name__ == "__main__":
    main()
