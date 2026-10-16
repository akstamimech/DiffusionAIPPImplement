# imitation planner on ONNX (new_imitate_1200)

The imitation policy (a transformer that regresses the 8 control waypoints of the expert in one pass) exported for the
drone. Unlike the diffusion policy there is no sampling loop: **one network call is the whole policy**, so a plan costs
one call instead of 20, and the result is deterministic. Trained on NAIP belief maps (epoch 1200, normalized with
HIGHOPT_DATASET_NAIP_141026.pt, the same dataset and the same normalization constants as the diffusion model).
The pre/post processing around the call is in `onnx_reference.py` (numpy + onnxruntime only).

| file | what | size |
|---|---|---|
| `new_imitate_1200_int8.onnx` | INT8 (dynamic, uint8 weights, per-channel). Recommended. | 2.5 MB |
| `new_imitate_1200_int8_per_tensor.onnx` | INT8, per-tensor weights (slightly less accurate, possibly simpler kernels) | 2.5 MB |
| `new_imitate_1200_fp32.onnx` | float32 baseline (matches PyTorch to 4e-6) | 6.0 MB |
| `new_imitate_1200_deploy.json` | normalization constants, world bounds, input/output description | |
| `onnx_reference.py` | complete runtime: inputs -> normalization -> marker map -> network call -> meters -> spline | |
| `new_imitate_1200_reference_cases.npz` | 16 real belief states with the expected PyTorch result, for validating a re-implementation | |
| `new_imitate_1200_test_vectors.npz` | 64 single network calls (inputs and expected fp32 outputs) | |
| `new_imitate_1200_quant_report.npz` | per-waypoint INT8 errors behind the accuracy numbers below | |
| `planner.py` | `ImitationOnnxPlanner`: this network as a planner for the mission loop | |
| `run.py` | flies one map in the simulator with this planner | |
| `export/` | exports the trained PyTorch network to ONNX, makes the reference cases, quantizes | |

Quick check on the target machine: `pip install numpy onnxruntime`, then
`python onnx_reference.py --model new_imitate_1200_int8.onnx --selftest`
(prints the waypoint error in meters against the PyTorch fp32 policy and the time per plan).

## Network interface (batch dimension is dynamic)
inputs: `meanvarmarker_map` [B,3,51,51] float32 | `current_position` [B,3] float32 | `total_variance_condition` [B,1]
float32. output: `waypoints_normalized` [B,3,8], the 8 control waypoints with xyz in [-1, 1].
All inputs are normalized exactly as described in `new_imitate_1200_deploy.json` (and implemented in `build_inputs`);
the output becomes meters with `denormalize` (clamped to the field and the altitude band 10..40 m).
The planner's belief is the GP posterior mean and variance on a 51 x 51 grid (2 m spacing, 100 m x 100 m field).
These are the same inputs as the diffusion network's, minus the noisy waypoints and the diffusion step.

## Running it

As a planner in the simulator (needs the simulation stack, see the root README), from `scripts/`:

```bash
SELECTED_MAP=5 TIMEALLOTED=200 python planners/imitatetrans_onnx/run.py
```

`planner.py` is `ImitationOnnxPlanner`, the planner as the mission loop sees it: it wraps `onnx_reference.Planner`, keeps
the plan at least 4 m inside the field edge (`PlannerConfig.padding_cells`, two grid steps) and z within 10..40 m, and
returns a `FlightPlan`. The robot around it is [`aipp/`](../../aipp/README.md); the simulator is
[`sim/`](../../sim/README.md), whose README lists the shared settings. This planner's own variables:
`ONNX_MODEL`, `ONNX_DEPLOY_JSON` (default `new_imitate_1200_int8.onnx`, `new_imitate_1200_deploy.json`) and `ORT_THREADS`.

Checks: `tests/test_golden_flights.py` flies a complete scenario with the fp32 model; that flight is identical to the
PyTorch planner's flight with the same model (`planners/imitatetrans`), pose for pose. `tests/test_mission.py` checks the
wrapper against the plain composition of reference calls, that planning is deterministic, and that the 16 reference
cases reproduce the PyTorch policy to under a centimeter.

## Accuracy of the INT8 model (against the fp32 model, 1024 real conditions)
- regression loss (MSE to the expert's waypoints, normalized): fp32 0.0507, INT8 0.0511
- waypoint difference vs fp32: mean 1.1 m, 95th percentile 2.6 m, worst 12 m (per-tensor: 1.2 / 2.7 / 13 m)
- the policy is deterministic, so unlike the diffusion model these errors are not hidden by mode flips: they are
  plain quantization error

## Notes
- Speed: on an x86 laptop one call takes about 8.6 ms in fp32 and 5.5 ms in INT8 (1 thread). Unlike the diffusion
  model, INT8 is faster here as well as smaller. Speed on a Raspberry Pi Zero 2 W (Cortex-A53) has not been measured.
- The adaptive average pooling inside the belief-map CNN (25 x 25 -> 12 x 12) is replaced by two fixed matrix multiplies
  that equal it exactly, because the ONNX exporter cannot express it for this size.
- Rebuild, from `scripts/`: `python planners/imitatetrans_onnx/export/export_onnx.py`, `.../make_reference_cases.py`,
  `.../quantize_onnx.py`.
- The reference clips waypoints to the field (x, y in 0..100 m, z in 10..40 m). `planner.py` additionally keeps the
  spline and the waypoints at least 4 m away from the field edge, as the simulation always did; a re-implementation needs
  that clamp to fly identical paths.
- If you export a different checkpoint, pass the dataset it was trained on (`--dataset`): the normalization constants
  come from it, and wrongly scaled inputs give plausible-looking but wrong plans without any error.
