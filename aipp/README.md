# aipp: the robot

The part of the system that runs on the drone: it keeps the Gaussian-process belief over the field, models what the
camera sees, follows a flight plan, and runs the mission loop. It needs only numpy. Nothing in here imports the
simulator, the planners, scikit-learn, scipy, matplotlib, PyTorch or the training code, so it can be read, ported or
run on the drone as it is.

The planning algorithm is deliberately **not** in here: a planner is anything that turns a belief and a pose into a
flight plan (see [`planners/`](../planners/README.md)). The simulator that stands in for the real world is in
[`sim/`](../sim/README.md).

## One mission step (`mission.py`)

1. The first two steps (warm-up) fly toward (80, 80, 10) so the belief gets a few measurements. From then on, when
   there is no plan or the current one is used up (`EXECUTION_CHUNK` = 40 of its 41 points passed, or all of them),
   the planner is called with the belief and the current pose, and returns a new `FlightPlan`.
2. The follower aims at the current plan point, advances to the next one once the drone is within one grid step of
   it, and returns a `Command` (unit direction vector).
3. The `Vehicle` carries out the command and reports its new pose.
4. The `MeasurementSource` returns readings at the new pose, and `GPBelief` does the Kalman update.
5. Observers registered on the mission get a `StepRecord` (pose, plan, variance, ...). The simulator uses one to
   compute RMSE against ground truth; the mission itself knows nothing about ground truth, metrics or files.

## Modules

| module | what it is |
|---|---|
| `config.py` | `GridSpec` (51 x 51 cells, 2 m, 100 m field), `KernelSpec` (Matern 3/2 prior, sigma2 0.0079, lengthscale 4.78 m), `SensorSpec` (60 degree FOV, noise variance 0.01 at 10 m growing toward 0.04, one reading per cell up to 20 m, per 2 x 2 cells up to 30 m, per 4 x 4 above) |
| `sensor_model.py` | `observation(x, y, z, grid, sensor)`: which cells the camera sees, how they group into resolution blocks, the averaging matrix `H` and the noise variance |
| `belief.py` | `GPBelief`: mean `mu` and covariance `P` over all cells, `update(H, z, noise_var)` (Kalman), `mean_map()`, `var_map()`, `total_variance()` |
| `interfaces.py` | `FlightPlan`, `Command`, and the seams the mission talks to: `MeasurementSource`, `Vehicle`, `Follower`, `PathPlanner` |
| `follower.py` | `PathFollower`: flight plan to one `Command` per step, and when a new plan is due. Planners with their own way of flying (`planners/greedygradient`, `planners/lawnmower`) provide their own follower. |
| `mission.py` | `Mission`, `MissionConfig`, `StepRecord`: the loop above, plus the optional wall-clock budget and a minimum time per step |

## Conventions
- Belief maps are `[51, 51]` with rows = y and columns = x; the flat cell index is `row * 51 + col`, which is also the
  order of `mu` and the rows/columns of `P`.
- Poses are `(x, y, z)` in meters in the field frame (x, y in 0..100, z in 10..40). One simulation step moves the
  drone one grid step (2 m).
- `P` is a dense 2601 x 2601 float64 matrix. One Kalman update takes about 50 to 70 ms on the x86 laptop (not
  measured on the Pi); that is the largest per-step cost besides planning.
- Readings are one number per resolution block (the mean of the block's visible cells), not per cell. The rows of `H`
  are the blocks, in the order they first appear when scanning the footprint x-major.

## Using it on a drone
Write two small classes and give them to `Mission` together with the unchanged belief, a planner and a follower:
- `Vehicle`: a `pose` property returning the current `(x, y, z)` in the field frame, and `fly(command)`. `command.direction`
  is a unit vector toward the point being tracked, to be flown for one step (2 m in the simulation); `command.hold`
  means do not move. `command.margin` only matters to the simulated vehicle (it clips to the field); on hardware the
  geofence is the autopilot's job.
- `MeasurementSource`: `measure(pose)` returning `(H, z, noise_var)`. Get `H` and `noise_var` from
  `observation(x, y, z, grid, sensor)`, and fill `z` with one reading per row of `H` (the mean NDVI of that block).

For the diffusion policy the planner is [`planners/diffusion_onnx`](../planners/diffusion_onnx/README.md).

Not done yet: mapping between the 0..100 m field frame and the flight arena, replanning in the background while
flying (a plan is computed synchronously, between two steps), and poses that are not on the 2 m grid (the camera
footprint accepts any pose, but the simulation never tries one, and the network's position marker is rounded to the
nearest cell).

## Tests
`tests/test_core_equivalence.py` holds the belief, camera model, vehicle and follower to the original
scikit-learn/scipy code (prior covariance, footprint and blocks at every altitude band and at the field edges, the
exact random draws, chains of Kalman updates, the dynamics, the follower against a copy of the old loop).
`tests/test_mission.py` covers the loop itself with stub parts and a fake clock: warm-up, when replans happen, the
wall-clock budget, step pacing, observers.
