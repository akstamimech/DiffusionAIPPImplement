# planners/mdn: mixture density network planner

Planner for the mixture-density baseline (not written yet). Follows the layout in `planners/README.md`:

    planner.py    class with  z_range = (10.0, 40.0)  and  plan(belief, pose) -> FlightPlan
    run.py        thin driver: sim.FlightSettings.from_env(...) + sim.run_flight(planner, settings)
    checkpoints/  trained weights (git-ignored folder; add an LFS rule in .gitattributes to track a checkpoint)

`planners/_torch_policy.py` already turns a belief into network inputs and a network output into a clamped, splined
FlightPlan; the new planner only has to turn the mixture into one [3, 8] set of normalized waypoints. When it exists, add it
to `PLANNERS` in `tests/test_golden_flights.py` and record a golden flight.
