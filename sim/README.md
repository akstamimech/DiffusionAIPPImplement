# sim: the simulator

Everything a flight needs besides the robot ([`aipp/`](../aipp/README.md)) and the planner
([`planners/`](../planners/README.md)): the ground-truth map, a noisy camera, a point-mass drone, the scoring, and the
function that ties them together. All planners run through the same code, so the simulator and robot settings cannot
drift apart between them.

```python
from sim import FlightSettings, run_flight

settings = FlightSettings.from_env(output_prefix="classic", plot_label="", selected_map=11, timealloted=300)
result = run_flight(my_planner, settings)        # my_planner.plan(belief, pose) -> FlightPlan, my_planner.z_range
```

| module | what it is |
|---|---|
| `flight.py` | `FlightSettings` (all simulator settings, read from the environment) and `run_flight(planner, settings)`: builds the world, belief, camera and drone, runs the mission, scores every step against ground truth, writes the CSVs and plots |
| `world.py` | `load_ground_truth`: reads `csv/map_<id>_<MAPTYPE>_grid_counts.csv` into the field vector |
| `sensor.py` | `SimSensor`: readings from the ground-truth field plus seeded noise (one noise draw per resolution block) |
| `vehicle.py` | `PointMassVehicle`: moves one grid step per command, position and altitude snapped to the grid |
| `metrics.py` | reconstruction RMSE (global and in the important regions), task completion, coverage efficiency, AUC |
| `replay.py` | post-hoc Kalman replay: reconstructs the variance of the important region from a saved `executed_trajectory.csv`, no re-simulation needed |

## Settings

Read from environment variables; each planner's `run.py` passes its own defaults for the ones that differ.

| variable | default | meaning |
|---|---|---|
| `SELECTED_MAP`, `MAPTYPE` | per planner, NAIP | which ground-truth field |
| `TIMEALLOTED` | per planner | number of steps to fly |
| `WALLCLOCK_SECONDS` | 150 | stop starting steps after this many seconds; `<= 0` = no limit |
| `ENFORCE_MIN_STEP_TIME` | 1 | pad each step to at least 2/3 s (one 2 m grid step at 3 m/s); `0` = off |
| `SENSORNOISE_SEED` | 123 | camera noise seed (plus the map id) |
| `START_X`, `START_Y` | 4.0, 4.0 | start position (altitude starts at 10 m) |
| `EXECUTION_CHUNK` | per planner | plan points flown before replanning |
| `UTILITY_THRESHOLD`, `BETA` | 0.3, 1 | importance threshold and weight of the standard deviation in the importance mask (the belief prior mean is the threshold minus 0.1) |
| `SKIP_VIZ` | 0 | `1` skips the videos (the RMSE and variance plots are always written) |
| `RESULTS_ROOT`, `RUN_OUTPUT_TAG`, `CSV_DIR` | `Vizualization/`, none, `csv/` | where outputs go and where maps are read |

## Outputs

In `<RESULTS_ROOT>/<planner>_map_<MAPTYPE>_<map>_viz[_<tag>]/`: `map_<M>_rmse_over_time.csv` (timestep, wall time,
global RMSE, occupied RMSE, total variance), `map_<M>_executed_trajectory.csv` (timestep, wall time, x, y, z), the
RMSE and variance plots, and videos unless `SKIP_VIZ=1`.
