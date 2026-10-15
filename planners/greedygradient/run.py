"""
Single-map flight with the greedy-gradient planner.

    SELECTED_MAP=5 TIMEALLOTED=200 python planners/greedygradient/run.py

Simulator and loop are the shared `sim` / `aipp`; settings: see sim/flight.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]       # scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planners.greedygradient.follower import TargetFollower
from planners.greedygradient.planner import GreedyGradientPlanner
from sim import FlightSettings, run_flight
from sim.flight import GRID


def main():
    settings = FlightSettings.from_env(
        output_prefix="greedygradient", plot_label="Greedy Gradient",
        selected_map=53, timealloted=3000, wallclock_seconds=300.0,
    )
    planner = GreedyGradientPlanner(GRID, utility_threshold=settings.utility_threshold, beta=settings.beta)
    run_flight(planner, settings, follower=TargetFollower(GRID, planner.z_range))


if __name__ == "__main__":
    main()
