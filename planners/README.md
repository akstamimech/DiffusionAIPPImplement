# planners

One folder per planning algorithm. A planner knows nothing about the simulator or the robot: it receives the belief and
the pose and returns a flight plan. Everything else (camera, drone, mission loop, scoring, output files) is the shared
[`aipp/`](../aipp/README.md) and [`sim/`](../sim/README.md), so planners can only differ in how they plan.

## What a planner folder contains

```
planners/<name>/
├── __init__.py
├── planner.py     a class with  z_range = (zmin, zmax)  and  plan(belief, pose) -> FlightPlan
├── run.py         ~30 lines: read the planner's own settings, build it, call sim.run_flight(planner, settings)
└── follower.py    only if the planner flies its plan differently from "follow the spline" (greedygradient, lawnmower)
```

`belief` is an `aipp.GPBelief`: `belief.mu` and `belief.P` (mean and covariance), or `belief.mean_map()` /
`belief.var_map()` as `[51, 51]` arrays for the networks. A `FlightPlan` is `waypoints` (the control points, `[K, 3]`),
`path` (the dense path to fly, `[N, 3]`) and `seconds` (how long planning took). Settings that apply to every planner
(map, time budget, start, execution chunk, ...) come from `sim.FlightSettings.from_env(...)`; the planner's `run.py`
only adds its own.

## The planners

| folder | planner | notes |
|---|---|---|
| `diffusion/` | trained diffusion policy, PyTorch | `new_1200.pth`, normalization from `HIGHOPT_DATASET_NAIP_141026.pt` |
| `diffusion_onnx/` | the same policy through onnxruntime | [README](diffusion_onnx/README.md); the one that runs on the drone |
| `imitatetrans/` | imitation baseline, PyTorch | one forward pass, no sampling |
| `cmaes/` | the expert | lattice grid search, then CMA-ES; needs the covariance |
| `realgreedy/` | greedy | `cmaes/` with horizon 1 and a chunk of 5 |
| `greedygradient/` | greedy | straight at the best lattice point, replan on arrival |
| `lawnmower/` | non-adaptive sweep | planned once, ignores the belief |

`_torch_policy.py` is what the two PyTorch planners share (belief to network inputs, network output to a clamped
spline).

## Adding a planner

1. Make `planners/<name>/planner.py` with the class above and `planners/<name>/run.py` modeled on `cmaes/run.py`.
2. Add it to `PLANNERS` in `tests/test_golden_flights.py` with a short seeded scenario, record its golden with
   `python tests/test_golden_flights.py --planner <name> --update-golden`, and look at the result before keeping it.
3. `python tests/test_imports_resolve.py` should stay clean.
