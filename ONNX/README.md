# new_1200 diffusion planner, ONNX export (this folder)

The network is the noise predictor of a diffusion policy for adaptive informative path planning (one UAV,
8 control waypoints per plan, trained on NAIP belief maps; epoch 1200, trained on HIGHOPT_DATASET_NAIP_141026.pt).
The ONNX files contain ONE denoising pass. A plan needs 20 calls (DDIM, deterministic) plus the small pre/post
processing in `onnx_sample_reference.py` (numpy + onnxruntime only).

| file | what | size |
|---|---|---|
| `new_1200_int8.onnx` | INT8 (dynamic, uint8 weights, per-channel). Recommended. | 2.6 MB |
| `new_1200_int8_per_tensor.onnx` | INT8, per-tensor weights (slightly less accurate, possibly simpler kernels) | 2.6 MB |
| `new_1200_fp32.onnx` | float32 baseline (matches PyTorch to 4e-6) | 6.3 MB |
| `new_1200_deploy.json` | normalization constants, noise schedule, world bounds, input/output description | |
| `onnx_sample_reference.py` | complete sampler: inputs -> normalization -> marker map -> 20-step DDIM -> meters -> spline | |
| `new_1200_reference_cases.npz` | 16 real belief states with the expected PyTorch result, for validating a re-implementation | |
| `new_1200_test_vectors.npz` | 64 single network calls (inputs and expected fp32 outputs) | |
| `aipp/` | the planner as a program: belief, camera model, planner, path follower, mission loop ([below](#the-aipp-package)) | |
| `DIFFUSIONPLANNER_ONNX.py` | simulator driver: flies one map with `aipp`, scores it against ground truth, writes CSVs and plots | |
| `tests/` | equivalence tests, mission-loop tests, golden regression flights ([below](#tests)) | |

Quick check on the target machine: `pip install numpy onnxruntime`, then
`python onnx_sample_reference.py --model new_1200_int8.onnx --selftest`
(prints the waypoint error in meters against the PyTorch fp32 sampler and the time per full plan).

## Network interface (batch dimension is dynamic)
inputs: `x` [B,3,8] float32 noisy waypoints | `t` [B] int64 diffusion step (0..19) | `meanvarmarker_map` [B,3,51,51]
float32 | `current_position` [B,3] float32 | `total_variance_condition` [B,1] float32. output: `noise_pred` [B,3,8].
All inputs are normalized exactly as described in `new_1200_deploy.json` (and implemented in `build_inputs`).
The planner's belief is the GP posterior mean and variance on a 51 x 51 grid (2 m spacing, 100 m x 100 m field).

## The `aipp` package

`DIFFUSIONPLANNER_ONNX.py` used to be one script that did everything. Its logic now lives in `aipp/`, which needs
only numpy (the planner module also needs onnxruntime). Nothing in `aipp` imports the simulator, scikit-learn, scipy,
matplotlib, PyTorch or the training code, so it can be read, ported or run on the drone as it is. What only exists in
simulation (a camera that knows the true field, a drone that is a point mass) is in `aipp/sim/`.

### One mission step (`aipp/mission.py`)

1. The first two steps (warm-up) fly toward (80, 80, 10) so the belief gets a few measurements. From then on, when
   there is no plan or the current one is used up (`EXECUTION_CHUNK` = 40 of its 41 points passed, or all of them),
   the planner is called with the belief's mean and variance maps and the current pose, and returns a new `FlightPlan`.
2. `PathFollower` aims at the current plan point, advances to the next one once the drone is within one grid step of
   it, and returns a `Command` (unit direction vector).
3. The `Vehicle` carries out the command and reports its new pose.
4. The `MeasurementSource` returns readings at the new pose, and `GPBelief` does the Kalman update.
5. Observers registered on the mission get a `StepRecord` (pose, plan, variance, ...). The simulator uses one to
   compute RMSE against ground truth; the mission itself knows nothing about ground truth, metrics or files.

### Modules

| module | what it is | needs |
|---|---|---|
| `config.py` | `GridSpec` (51 x 51 cells, 2 m, 100 m field), `KernelSpec` (Matern 3/2 prior, sigma2 0.0079, lengthscale 4.78 m), `SensorSpec` (60 degree FOV, noise variance 0.01 at 10 m growing toward 0.04, one reading per cell up to 20 m, per 2 x 2 cells up to 30 m, per 4 x 4 above) | numpy |
| `sensor_model.py` | `observation(x, y, z, grid, sensor)`: which cells the camera sees, how they group into resolution blocks, the averaging matrix `H` and the noise variance | numpy |
| `belief.py` | `GPBelief`: mean `mu` and covariance `P` over all cells, `update(H, z, noise_var)` (Kalman), `mean_map()`, `var_map()`, `total_variance()` | numpy |
| `interfaces.py` | `FlightPlan`, `Command`, and the three seams the mission talks to: `MeasurementSource`, `Vehicle`, `PathPlanner` | numpy |
| `planner.py` | `DiffusionPlanner.plan(mean_map, var_map, pose)` returns a `FlightPlan` (8 waypoints, 41-point spline, seconds taken). Wraps `onnx_sample_reference.Planner`, keeps the plan at least 4 m inside the field edge and z within 10..40 m | onnxruntime |
| `follower.py` | `PathFollower`: plan to one `Command` per step, and when a new plan is due | numpy |
| `mission.py` | `Mission`, `MissionConfig`, `StepRecord`: the loop above, plus the optional wall-clock budget and a minimum time per step | numpy |
| `sim/sim_sensor.py` | `SimSensor`: readings from a ground-truth field plus seeded noise | numpy |
| `sim/vehicle.py` | `PointMassVehicle`: moves one grid step per command, position and altitude snapped to the grid | numpy |

`import aipp` does not import `aipp.planner`, so the rest of the package works without onnxruntime.

### Conventions
- Belief maps are `[51, 51]` with rows = y and columns = x; the flat cell index is `row * 51 + col`, which is also the
  order of `mu` and the rows/columns of `P`.
- Poses are `(x, y, z)` in meters in the field frame (x, y in 0..100, z in 10..40).
- `P` is a dense 2601 x 2601 float64 matrix. One Kalman update takes about 50 to 70 ms on the x86 laptop (not
  measured on the Pi); that is the largest per-step cost besides planning.
- Readings are one number per resolution block (the mean of the block's visible cells), not per cell. The rows of `H`
  are the blocks, in the order they first appear when scanning the footprint x-major.

### Using it on a drone
Write two small classes and give them to `Mission` together with the unchanged belief, planner and follower:
- `Vehicle`: a `pose` property returning the current `(x, y, z)` in the field frame, and `fly(command)`. `command.direction`
  is a unit vector toward the point being tracked, to be flown for one step (2 m in the simulation); `command.hold`
  means do not move. `command.margin` only matters to the simulated vehicle (it clips to the field); on hardware the
  geofence is the autopilot's job.
- `MeasurementSource`: `measure(pose)` returning `(H, z, noise_var)`. Get `H` and `noise_var` from
  `observation(x, y, z, grid, sensor)`, and fill `z` with one reading per row of `H` (the mean NDVI of that block).

Not done yet: mapping between the 0..100 m field frame and the flight arena, replanning in the background while
flying (a plan is computed synchronously, between two steps), and poses that are not on the 2 m grid (the camera
footprint accepts any pose, but the simulation never tries one, and the network's position marker is rounded to the
nearest cell).

### Running the simulator driver
Needs the simulation stack too (`pip install numpy onnxruntime scipy scikit-learn matplotlib cma imageio tqdm`).
From `scripts/`:

```bash
SELECTED_MAP=5 TIMEALLOTED=200 RUN_SEED=1 python ONNX/DIFFUSIONPLANNER_ONNX.py
```

| variable | default | meaning |
|---|---|---|
| `SELECTED_MAP`, `MAPTYPE` | 26, NAIP | which ground-truth field (`csv/map_<id>_<MAPTYPE>_grid_counts.csv`) |
| `TIMEALLOTED` | 4000 | number of steps to fly |
| `WALLCLOCK_SECONDS` | 150 | stop starting steps after this many seconds; `<= 0` = no limit |
| `ENFORCE_MIN_STEP_TIME` | 1 | pad each step to at least 2/3 s (one 2 m grid step at 3 m/s); `0` = off |
| `RUN_SEED`, `SENSORNOISE_SEED` | 1121, 123 | planner noise seed; camera noise seed (plus the map id) |
| `START_X`, `START_Y` | 4.0, 4.0 | start position (altitude starts at 10 m) |
| `ONNX_MODEL`, `ONNX_DEPLOY_JSON` | `new_1200_int8.onnx`, `new_1200_deploy.json` | network and its constants |
| `ETA`, `DDIM_STEPS`, `ORT_THREADS` | 0.0, all 20, onnxruntime default | sampler and runtime settings |
| `EXECUTION_CHUNK` | 40 | plan points flown before replanning |
| `UTILITY_THRESHOLD` | 0.3 | importance threshold (the belief prior mean is this minus 0.1) |
| `SKIP_VIZ` | 0 | `1` skips the videos (the RMSE and variance plots are always written) |
| `RESULTS_ROOT`, `RUN_OUTPUT_TAG`, `CSV_DIR` | `Vizualization/`, none, `csv/` | where outputs go and where maps are read |

Outputs, in `<RESULTS_ROOT>/diffusion_onnx_map_<MAPTYPE>_<map>_viz[_<tag>]/`:
`map_<M>_rmse_over_time.csv` (timestep, wall time, global RMSE, occupied RMSE, total variance),
`map_<M>_executed_trajectory.csv` (timestep, wall time, x, y, z), the RMSE and variance plots, and videos unless `SKIP_VIZ=1`.

### Tests
From `scripts/`:

```bash
python ONNX/tests/test_core_equivalence.py   # aipp's belief, camera, vehicle and follower equal the original code
python ONNX/tests/test_mission.py            # mission loop with stub parts and a fake clock; planner wrapper
python ONNX/tests/test_golden_regression.py  # two complete flights against recorded results
```

- `test_core_equivalence.py` compares the numpy code with the original scikit-learn/scipy functions in `functions.py`
  and `CMAES_classic_singlemap.py` (prior covariance, footprint and blocks at every altitude band and at the field
  edges, the exact random draws, chains of Kalman updates, the dynamics, the follower against a copy of the old loop).
- `test_mission.py` covers what the golden flights cannot: warm-up, when replans happen, the wall-clock budget, step
  pacing, observers, and that `DiffusionPlanner` reproduces the old composition of network calls.
- `test_golden_regression.py` flies map 5 (seed 1, 90 steps) and map 12 (seed 7, 130 steps, other start) and compares
  RMSE and variance (relative tolerance 1e-6) and the executed trajectory (exact) with `tests/golden/`. Run it after
  any change to `aipp/` or the driver. Re-record with `--update-golden` only when a behaviour change is intended.
  It uses the fp32 model, one thread and no pacing on purpose: the INT8 model quantizes activations at run time, so
  a float-level change in the belief (even a different thread count) can flip a rounding and send a closed-loop
  flight down another path (seen: final variance 14.93 vs 14.76 for the same scenario).

## Accuracy of the INT8 model (against the fp32 model, held-out real data)
- denoising loss (training objective): fp32 0.1677, INT8 0.1680
- sampled waypoints, same initial noise, 128 real conditions: mean error 0.8 m, 95th percentile 2.3 m, worst 13 m
  (two fp32 samples for the same condition differ by 32 m on average, since the policy is multimodal)
- worst cases are conditions where the denoising path flips to another mode, not a uniform degradation

## Notes
- Speed: on an x86 laptop INT8 and fp32 take the same time per call (about 3.6 ms, 1 thread); the INT8 gain
  measured here is size. Speed on a Raspberry Pi Zero 2 W (Cortex-A53) has not been measured.
- Static INT8 quantization and int8 (signed) weights were tested and are clearly less accurate for this network.
- About two thirds of the compute per call is the belief-map CNN, which does not depend on the diffusion step;
  splitting it into its own graph run once per plan would cut a 20-step plan to roughly a third.
- Rebuild: `python ../Diffusion/export_onnx.py`, `python ../Diffusion/make_reference_cases.py`, `python ../Diffusion/quantize_onnx.py`.
- The reference clips waypoints to the field (x, y in 0..100 m, z in 10..40 m). `aipp/planner.py` additionally keeps
  the spline and the waypoints at least 4 m away from the field edge (two grid steps, `PlannerConfig.padding_cells`),
  as the simulation always did; a re-implementation needs that clamp to fly identical paths.
