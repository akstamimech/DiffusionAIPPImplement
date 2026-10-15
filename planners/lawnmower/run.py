"""
Single-map flight with the lawnmower baseline.

    SELECTED_MAP=5 TIMEALLOTED=400 python planners/lawnmower/run.py

Simulator and loop are the shared `sim` / `aipp` (so the lawnmower now sees the same camera, noise and drone as every
other planner); settings: see sim/flight.py.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]       # scripts/
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planners.lawnmower.follower import SweepFollower
from planners.lawnmower.planner import LawnmowerPlanner
from sim import FlightSettings, run_flight
from sim.flight import GRID


def main():
    settings = FlightSettings.from_env(
        output_prefix="lawnmower", plot_label="Lawnmower",
        selected_map=55, timealloted=400,
    )
    planner = LawnmowerPlanner(GRID, altitude=settings.init_altitude)
    run_flight(planner, settings, follower=SweepFollower(GRID, planner.z_range))


if __name__ == "__main__":
    main()
