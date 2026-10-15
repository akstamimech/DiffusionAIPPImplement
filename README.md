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

1. **Expert demonstrations via CMA-ES.** `DataCollector_3D_randomstart_CMAESregularized.py` runs a
   receding-horizon planner: a coarse grid search proposes candidate waypoints, and CMA-ES refines them
   into continuous 3D trajectories that maximize expected variance reduction per unit distance travelled.
   Multiple near-tied candidate branches are recorded per decision point to capture genuinely multimodal
   choices, not just a single "correct" answer.
2. **Diffusion policy trained by imitation.** `Diffusion/threeDSparseTransDiffusion.py` trains a
   conditional denoising diffusion model to reproduce these expert trajectories, conditioned on the
   current belief (mean/variance maps), position, heading, and total remaining uncertainty.
3. **Baselines for comparison.** `ImitateTrans.py` (a direct, deterministic waypoint regressor trained on
   the same data), `lawnmower_singlemap.py` (fixed systematic coverage), `realgreedy_singlemap.py` /
   `greedygradient_singlemap.py` (myopic planners), and live CMA-ES itself (`CMAES_classic_singlemap.py`)
   all share the same simulation, belief-update, and metric machinery, so they're directly comparable.
4. **Evaluation.** Planners are compared on how fast and how completely they reduce variance, in both the
   whole map and specifically the regions that cross the importance threshold, over wall-clock time or a
   fixed number of simulation steps.

## Map sources

Two kinds of scalar field are supported, selected per-script via the `MAPTYPE` environment variable:

- **Synthetic Gaussian Random Fields** (`grf`), generated for controlled experiments with a known,
  tunable spatial correlation structure.
- **Real NAIP aerial imagery** (`NAIP`), NDVI tiles pulled from actual satellite/aerial data (see
  `Datasets/NAIP_dataset/`), used to test planners against genuine, spatially irregular ground truth
  rather than an idealized field.

Check each script's own `MAPTYPE` default before running. Individual scripts sometimes differ, and
several downstream constants (the GP kernel's `sigma2`/lengthscale, the importance-mask direction via the
`LCB` flag, the utility threshold, and the belief prior mean) all have to move together when switching
map types. `gaussianprocesstraining.py`'s module-level `LCB`/`COST_EXPONENT` and `initialize_gp()`'s
kernel defaults are the shared source of truth for most of this.

## Repository layout

```
scripts/
├── gaussianprocesstraining.py       Shared GP belief tracking, Kalman updates, importance filtering,
│                                    grid-search + CMA-ES refinement planners
├── evalmetrics.py                   RMSE / variance / task-completion metrics
├── important_region_variance_from_trajectories.py
│                                    Post-hoc Kalman replay: reconstructs occupied-region variance from
│                                    a saved executed_trajectory.csv, no re-simulation needed
├── DataCollector_3D_randomstart_CMAESregularized.py
│                                    Canonical expert-trajectory data collector (MPI-parallel)
├── CMAES_classic_singlemap.py       Live CMA-ES receding-horizon planner (single map, evaluation)
├── ImitateTrans.py / ImitateTrans_singlemap.py
│                                    Deterministic imitation-learning baseline (train / evaluate)
├── lawnmower_singlemap.py, realgreedy_singlemap.py, greedygradient_singlemap.py
│                                    Non-adaptive and myopic baselines
├── Diffusion/
│   ├── threeDSparseTransDiffusion.py         Diffusion model architecture + training loop
│   ├── threeDSparseTransDiffusion_periodic_eval.py
│   │                                          Training with periodic single-map held-out evaluation
│   ├── sample_3d_sparse_trans_diffusion.py   Loads a trained model for sampling
│   └── checkpoints/                          Saved model weights
├── Diffusionplanner_singlemap.py    Diffusion policy, single-map evaluation
├── Kerneltraining/                  GP kernel hyperparameter fitting against real map statistics
├── csv/                             Per-map ground-truth grids (`map_<id>_<MAPTYPE>_grid_counts.csv`)
├── Vizualization/                   Per-run output: trajectories, plots, GIFs
└── CARRYOVER_*.md                   Session handoff notes: what changed, why, and what's still open
```

## Setup

```bash
pip install torch numpy scipy scikit-learn matplotlib cma imageio tqdm
```

`mpi4py` is additionally required for parallel data collection
(`DataCollector_3D_randomstart_CMAESregularized.py` via `mpiexec`/`srun`). There is no pinned
`requirements.txt` yet; the above covers everything imported by the core pipeline. Developed against
Python 3.12.

## Running things

Every single-map script is configured entirely through environment variables, with sensible defaults
baked in, so nothing needs to be edited to try a different setting:

```bash
# Evaluate the diffusion planner on one map
SELECTED_MAP=51 MAPTYPE=grf WALLCLOCK_SECONDS=150 python Diffusionplanner_singlemap.py

# Compare against a live CMA-ES run on the same map
SELECTED_MAP=51 MAPTYPE=grf WALLCLOCK_SECONDS=150 python CMAES_classic_singlemap.py

# Or the lawnmower baseline
SELECTED_MAP=51 MAPTYPE=grf WALLCLOCK_SECONDS=150 python lawnmower_singlemap.py
```

Common variables across the single-map scripts: `SELECTED_MAP`, `MAPTYPE`, `UTILITY_THRESHOLD`,
`WALLCLOCK_SECONDS` (`<=0` runs to a fixed step count instead of a wall-clock budget),
`ENFORCE_MIN_STEP_TIME`, `RUN_SEED`, `RUN_OUTPUT_TAG` (keeps outputs from different configurations from
overwriting each other in `Vizualization/`), and `SKIP_VIZ=1` to skip plot/GIF generation for faster
batch runs.

To train the diffusion model:

```bash
cd Diffusion
python threeDSparseTransDiffusion.py
```

## Status

Active thesis research code. Currently In progress!
