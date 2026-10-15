# DiffusionAIPP

Diffusion models for adaptive informative path planning (AIPP). This repo trains a diffusion policy to
plan sensor trajectories that reduce uncertainty about an unknown spatial field as quickly as possible,
by imitating an expert planner (CMA-ES trajectory optimization over a Gaussian Process belief) rather
than hand-coding a policy or training with online RL alone.

Written as part of a thesis on learning-based informative path planning; the code is research-grade and
evolves quickly, not a packaged library.

## The problem

An aerial sensor needs to map an unknown 2D field (for example, vegetation health from NDVI, or any
scalar quantity read off a map) as accurately as possible within a limited flight time. The agent
maintains a Gaussian Process belief over the field, updated via Kalman-style measurement updates as it
flies, and has to decide where to look next to shrink uncertainty fastest, especially in the regions
that matter (cells above or below some threshold of interest).

## Approach

1. **Expert demonstrations via CMA-ES.** `datacollection/DataCollector_3D_randomstart_CMAESregularized.py`
   runs a receding-horizon planner: a coarse grid search proposes candidate waypoints, and CMA-ES refines
   them into continuous 3D trajectories that maximize expected variance reduction per unit distance
   travelled. Several CMA-ES runs with different seeds are recorded per decision point to capture genuinely
   multimodal choices, not just a single "correct" answer.
2. **Diffusion policy trained by imitation.** `training/diffusion/` trains a conditional denoising diffusion
   model to reproduce these expert trajectories, conditioned on the current belief (mean/variance maps),
   position, and total remaining uncertainty.
3. **Baselines for comparison.** The imitation regressor (`planners/imitatetrans/`, a direct deterministic
   waypoint regressor trained on the same data), the lawnmower (fixed systematic coverage), two greedy
   planners, and live CMA-ES itself. They all fly in the same simulator with the same robot, so they are
   directly comparable.
4. **Evaluation.** Planners are compared on how fast and how completely they reduce variance, in both the
   whole map and specifically the regions that cross the importance threshold, over wall-clock time or a
   fixed number of simulation steps.

## How the code is organized

**Every planner flies in the same simulator with the same robot. Only the planning algorithm differs.**

- [`aipp/`](aipp/README.md) is the robot: the GP belief and Kalman update, the camera model, the path
  follower and the mission loop. numpy only, and it is what runs on the real drone.
- [`sim/`](sim/) is the simulator: the ground-truth map, the noisy camera, a point-mass drone, the scoring,
  and `run_flight(planner, settings)`, which flies one map and writes the CSVs and plots.
- [`planners/<name>/`](planners/README.md) holds one planning algorithm each: a class with
  `plan(belief, pose) -> FlightPlan` and a short `run.py`. Nothing about the simulator or the robot is
  repeated there.

| planner | what it does |
|---|---|
| `planners/diffusion/` | the trained diffusion policy, PyTorch network |
| `planners/diffusion_onnx/` | the same policy exported to ONNX (fp32, INT8) for the drone ([README](planners/diffusion_onnx/README.md)) |
| `planners/imitatetrans/` | imitation baseline: a transformer regresses the expert waypoints in one pass |
| `planners/cmaes/` | the expert: lattice grid search, then CMA-ES refinement |
| `planners/realgreedy/` | CMA-ES with a one-waypoint horizon, replanned every 5 steps |
| `planners/greedygradient/` | fly straight at the best lattice point, replan on arrival |
| `planners/lawnmower/` | fixed back-and-forth sweep, ignores the belief |

## Repository layout

```
scripts/
├── aipp/                 robot core (belief, camera model, follower, mission loop); deployable, numpy only
├── sim/                  shared simulator: world, camera, drone, metrics, post-hoc replay, run_flight()
├── planners/             one folder per planner (see the table above), plus _torch_policy.py shared by the two torch planners
├── training/
│   ├── diffusion/        diffusion model, training loops, sampler modules, checkpoints/
│   ├── imitatetrans/     imitation model and its training loops
│   └── kernel/           GP kernel hyperparameter fitting against real map statistics
├── datacollection/       expert data collector (MPI), dataset tools, smoke tests
├── maps/                 ground-truth map tools (synthetic fields, NAIP, normalization)
├── analysis/             post-hoc analysis of finished runs (local, not in git)
├── plots/                figures for the thesis and papers (local, not in git)
├── animations/           recordings of planner behaviour (local, not in git)
├── sweeps/               parameter sweeps and the SLURM job scripts that launch them
├── situation_comparison/ side-by-side comparison of planners on identical belief states
├── tests/                regression and consistency tests (see below)
├── legacy/               superseded code, kept for reference (see below)
├── docs/                 design notes
├── functions.py          shared function library: GP/sensor/Kalman functions, the lattice grid search, CMA-ES
│                         refinement and spline builders, plotting. Used by the expert planner, the trainers
│                         and the data collector.
├── csv/                  per-map ground-truth grids (map_<id>_<MAPTYPE>_grid_counts.csv)
├── checkpoints/          saved model weights (git-ignored)
├── *.pt                  training datasets (git-ignored, except the ones committed on purpose)
├── Vizualization/        per-run output: trajectories, plots, videos (git-ignored)
└── CARRYOVER_*.md        session handoff notes: what changed, why, and what's still open
```

`legacy/` holds what the new structure replaced. `legacy/singlemap/` has the original single-map planner
scripts (each carried its own copy of the simulator loop; they still run and several scratch scripts import
them), `legacy/eval/` the multi-map evaluation loops, `legacy/ppo/` the PPO/DPPO experiments, and
`legacy/datacollection/`, `legacy/receding_gridsearch/`, `legacy/CMAES_2D/`, `legacy/old/`,
`legacy/diffusion_old/` older experiments. Nothing in `legacy/` is maintained.

## Map sources

Two kinds of scalar field are supported, selected via the `MAPTYPE` environment variable:

- **Synthetic Gaussian Random Fields** (`grf`), generated for controlled experiments with a known,
  tunable spatial correlation structure.
- **Real NAIP aerial imagery** (`NAIP`), NDVI tiles pulled from actual satellite/aerial data, used to test
  planners against genuine, spatially irregular ground truth rather than an idealized field.

The default is `NAIP`. Several downstream constants (the GP kernel's `sigma2`/lengthscale in
`aipp/config.py`, the importance-mask direction via the `LCB` flag in `functions.py`, the utility
threshold, and the belief prior mean) all have to move together when switching map types.

## Setup

```bash
pip install torch numpy scipy scikit-learn matplotlib cma imageio tqdm onnxruntime
```

`mpi4py` is additionally required for parallel data collection. The ONNX planner and `aipp/` itself need
only numpy and onnxruntime. There is no pinned `requirements.txt` yet; the above covers everything imported
by the maintained code. Developed against Python 3.12. Run everything from this folder (`scripts/`).

## Running things

Every planner is configured entirely through environment variables with sensible defaults, so nothing needs
to be edited to try a different setting. The simulator settings are the same for all planners (see
`sim/flight.py`): `SELECTED_MAP`, `MAPTYPE`, `TIMEALLOTED`, `UTILITY_THRESHOLD`, `BETA`, `SENSORNOISE_SEED`,
`START_X`/`START_Y`, `EXECUTION_CHUNK`, `WALLCLOCK_SECONDS` (`<=0` runs to a fixed step count instead of a
wall-clock budget), `ENFORCE_MIN_STEP_TIME` (pads each step to the 2/3 s a 2 m step takes at 3 m/s),
`RUN_OUTPUT_TAG` (keeps outputs from different configurations apart in `Vizualization/`), `RESULTS_ROOT`,
`CSV_DIR` and `SKIP_VIZ=1` (no videos). Each `run.py` documents its planner's own variables in its header.

```bash
SELECTED_MAP=51 WALLCLOCK_SECONDS=150 python planners/diffusion/run.py        # the trained diffusion policy
SELECTED_MAP=51 WALLCLOCK_SECONDS=150 python planners/cmaes/run.py            # the expert, on the same map
SELECTED_MAP=51 WALLCLOCK_SECONDS=150 python planners/lawnmower/run.py        # and the lawnmower
```

Outputs go to `Vizualization/<planner>_map_<MAPTYPE>_<map>_viz[_<tag>]/`.

To train the diffusion model, from `training/diffusion/` (for example
`python threeDSparseTransDiffusion_periodic_eval_headingremoved.py`); to collect expert data,
`datacollection/DataCollector_3D_randomstart_CMAESregularized.py` (MPI, see `datacollection/mapeval_multimodal.sh`).

## Tests

```bash
python tests/test_core_equivalence.py     # aipp / sim equal the original sklearn-scipy code
python tests/test_mission.py              # the mission loop (stub parts, fake clock) and the ONNX planner wrapper
python tests/test_golden_flights.py       # a complete short flight of every planner against recorded results
python tests/test_imports_resolve.py      # every project import resolves to a real module and name
```

The golden flights were recorded from the original single-map scripts before they were replaced, so a
passing run means each planner still behaves exactly as it did. Re-record (`--update-golden`) only when a
change in behaviour is intended.

## Where did it go?

| before | now |
|---|---|
| `gaussianprocesstraining.py` | `functions.py` |
| `evalmetrics.py` | `sim/metrics.py` |
| `important_region_variance_from_trajectories.py` | `sim/replay.py` |
| `CMAES_classic_singlemap.py` | `planners/cmaes/` (original: `legacy/singlemap/`) |
| `realgreedy_singlemap.py`, `greedygradient_singlemap.py`, `lawnmower_singlemap.py` | `planners/realgreedy/`, `planners/greedygradient/`, `planners/lawnmower/` |
| `Diffusionplanner_singlemap_headingremoved.py` | `planners/diffusion/` |
| `ImitateTrans_singlemap_headingremoved.py` | `planners/imitatetrans/` |
| `Diffusionplanner_singlemap.py`, `ImitateTrans_singlemap.py` (with heading input) | `legacy/singlemap/` |
| `random_singlemap.py` (an unfinished stub, never ran) | `legacy/singlemap/` |
| `ONNX/` | `planners/diffusion_onnx/` (robot core: `aipp/`) |
| `Diffusion/` | `training/diffusion/` (ONNX export: `planners/diffusion_onnx/export/`, DPPO: `legacy/ppo/`) |
| `ImitateTrans*.py` (model and training) | `training/imitatetrans/` |
| `Kerneltraining/` | `training/kernel/` |
| `DataCollector_3D_randomstart_CMAESregularized.py` | `datacollection/` |
| `DataCollector+datasets/` | `legacy/datacollection/` (SLURM job scripts: `sweeps/`) |
| `sweep_*`, `checkpoint_sweep_*`, `hpc_sweep_common.py` | `sweeps/` |
| `analyze_*`, `compare_*` | `analysis/` |
| `plot_*`, `generate_*`, `combine_*` | `plots/` |
| `record_*` | `animations/` |

The complete list of moves is in [`docs/reorganization_moves.csv`](docs/reorganization_moves.csv).

## Status

Active thesis research code. Currently In progress!
