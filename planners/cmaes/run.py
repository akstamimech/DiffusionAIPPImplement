"""
Single-map flight with the CMA-ES receding-horizon planner (the expert).

    SELECTED_MAP=5 TIMEALLOTED=200 python planners/cmaes/run.py

Simulator and loop are the shared `sim` / `aipp`; settings: see sim/flight.py plus the planner's own below.

    PLANNING_HORIZON   waypoints per plan (default 8)
    CMA_SEED           seed of the CMA-ES search (default 55)
    CMAES_REFINE       0 = skip the CMA-ES refinement and fly the raw grid-search waypoints (lattice-only baseline)
    BETA               weight of the standard deviation in the importance mask (default 1)
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
        output_prefix="classic", plot_label="",
        selected_map=11, timealloted=300, execution_chunk=40,
    )
    planner = CMAESPlanner(
        GRID,
        utility_threshold=settings.utility_threshold,
        beta=settings.beta,
        planning_horizon=int(os.environ.get("PLANNING_HORIZON", "8")),
        cma_seed=int(os.environ.get("CMA_SEED", "55")),
        refine=os.environ.get("CMAES_REFINE", "1") != "0",
    )
    run_flight(planner, settings)


if __name__ == "__main__":
    main()
