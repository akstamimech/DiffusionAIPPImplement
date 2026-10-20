# diffusion planner on ONNX (new_1200)

The network is the noise predictor of a diffusion policy for adaptive informative path planning (one UAV,
8 control waypoints per plan, trained on NAIP belief maps; epoch 1200, trained on HIGHOPT_DATASET_NAIP_141026.pt).
The ONNX files contain ONE denoising pass. A plan needs 20 calls (DDIM, deterministic) plus the small pre/post
processing in `onnx_sample_reference.py` (numpy + onnxruntime only).

| file | what | size |
|---|---|---|
| `new_1200_int8.onnx` | INT8 (dynamic, uint8 weights, per-channel). Recommended. | 2.6 MB |
| `new_1200_int16.onnx` | 16 bit (static QDQ: int16 weights per channel, uint16 activations, calibrated on 4096 real inputs). Near-lossless. | 4.1 MB |
| `new_1200_int8_per_tensor.onnx` | INT8, per-tensor weights (slightly less accurate, possibly simpler kernels) | 2.6 MB |
| `new_1200_fp32.onnx` | float32 baseline (matches PyTorch to 4e-6) | 6.3 MB |
| `new_1200_deploy.json` | normalization constants, noise schedule, world bounds, input/output description | |
| `onnx_sample_reference.py` | complete sampler: inputs -> normalization -> marker map -> 20-step DDIM -> meters -> spline | |
| `new_1200_reference_cases.npz` | 16 real belief states with the expected PyTorch result, for validating a re-implementation | |
| `new_1200_test_vectors.npz` | 64 single network calls (inputs and expected fp32 outputs) | |
| `planner.py` | `DiffusionPlanner`: this network as a planner for the mission loop ([below](#running-it)) | |
| `run.py` | flies one map in the simulator with this planner | |
| `export/` | exports the trained PyTorch network to ONNX, makes the reference cases, quantizes | |

Quick check on the target machine: `pip install numpy onnxruntime`, then
`python onnx_sample_reference.py --model new_1200_int8.onnx --selftest`
(prints the waypoint error in meters against the PyTorch fp32 sampler and the time per full plan).

## Network interface (batch dimension is dynamic)
inputs: `x` [B,3,8] float32 noisy waypoints | `t` [B] int64 diffusion step (0..19) | `meanvarmarker_map` [B,3,51,51]
float32 | `current_position` [B,3] float32 | `total_variance_condition` [B,1] float32. output: `noise_pred` [B,3,8].
All inputs are normalized exactly as described in `new_1200_deploy.json` (and implemented in `build_inputs`).
The planner's belief is the GP posterior mean and variance on a 51 x 51 grid (2 m spacing, 100 m x 100 m field).

## Running it

As a planner in the simulator (needs the simulation stack, see the root README), from `scripts/`:

```bash
SELECTED_MAP=5 TIMEALLOTED=200 RUN_SEED=1 python planners/diffusion_onnx/run.py
```

`planner.py` is `DiffusionPlanner`, the planner as the mission loop sees it: it wraps `onnx_sample_reference.Planner`,
keeps the plan at least 4 m inside the field edge (`PlannerConfig.padding_cells`, two grid steps) and z within 10..40 m,
and returns a `FlightPlan`. It needs only numpy and onnxruntime. The robot around it (belief, camera model, path
follower, mission loop) is [`aipp/`](../../aipp/README.md); the simulator is [`sim/`](../../sim/README.md), whose README
lists the shared settings. This planner's own variables:

| variable | default | meaning |
|---|---|---|
| `ONNX_MODEL`, `ONNX_DEPLOY_JSON` | `new_1200_int8.onnx`, `new_1200_deploy.json` | network and its constants |
| `ETA`, `DDIM_STEPS`, `ORT_THREADS` | 0.0, all 20, onnxruntime default | sampler and runtime settings |
| `RUN_SEED` | 1121 | seed of the planner's initial noise |

The golden-flight test (`tests/test_golden_flights.py`, planner `diffusion_onnx`) flies two complete scenarios with the
fp32 model, one thread and no pacing, on purpose: the INT8 model quantizes activations at run time, so a float-level
change in the belief (even a different thread count) can flip a rounding and send a closed-loop flight down another
path (seen: final variance 14.93 vs 14.76 for the same scenario). `tests/test_mission.py` checks that `DiffusionPlanner`
reproduces the plain composition of network calls.

## Accuracy of the INT8 model (against the fp32 model, held-out real data)
- denoising loss (training objective): fp32 0.1677, INT8 0.1680
- sampled waypoints, same initial noise, 128 real conditions: mean error 0.8 m, 95th percentile 2.3 m, worst 13 m
  (two fp32 samples for the same condition differ by 32 m on average, since the policy is multimodal)
- worst cases are conditions where the denoising path flips to another mode, not a uniform degradation

### 16 bit (`new_1200_int16.onnx`)
- sampled waypoints, same initial noise, 128 real conditions, vs fp32: mean 0.015 m, 95th percentile 0.04 m, worst 0.9 m
  (INT8: 0.8 / 2.3 / 13 m); denoising loss 0.1669, identical to fp32
- built by `export/quantize_int16.py`. It is static quantization, so the activation ranges come from real denoiser inputs
  (noised expert waypoints at random diffusion steps with real belief maps); the calibration set has to be large (4096 here)
- it uses the `com.microsoft` Quantize/DequantizeLinear operators because 16 bit is not in the standard opset used here;
  a runtime without them (for example a hand-written C++ port) cannot load it and should use fp32 or INT8
- on the x86 laptop it runs slower than fp32 (about 6.6 ms vs 3.4 ms per call): onnxruntime's CPU kernels dequantize
  16-bit tensors to float, so 16 bit saves memory (4.1 vs 6.3 MB), not time

## Notes
- Speed: on an x86 laptop INT8 and fp32 take the same time per call (about 3.6 ms, 1 thread); the INT8 gain
  measured here is size. Speed on a Raspberry Pi Zero 2 W (Cortex-A53) has not been measured.
- Static INT8 quantization and int8 (signed) weights were tested and are clearly less accurate for this network.
- About two thirds of the compute per call is the belief-map CNN, which does not depend on the diffusion step;
  splitting it into its own graph run once per plan would cut a 20-step plan to roughly a third.
- Rebuild, from `scripts/`: `python planners/diffusion_onnx/export/export_onnx.py`, `.../make_reference_cases.py`, `.../quantize_onnx.py`, `.../quantize_int16.py`.
- The reference clips waypoints to the field (x, y in 0..100 m, z in 10..40 m). `planner.py` additionally keeps
  the spline and the waypoints at least 4 m away from the field edge (two grid steps, `PlannerConfig.padding_cells`),
  as the simulation always did; a re-implementation needs that clamp to fly identical paths.
