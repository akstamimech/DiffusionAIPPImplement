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

Quick check on the target machine: `pip install numpy onnxruntime`, then
`python onnx_sample_reference.py --model new_1200_int8.onnx --selftest`
(prints the waypoint error in meters against the PyTorch fp32 sampler and the time per full plan).

## Network interface (batch dimension is dynamic)
inputs: `x` [B,3,8] float32 noisy waypoints | `t` [B] int64 diffusion step (0..19) | `meanvarmarker_map` [B,3,51,51]
float32 | `current_position` [B,3] float32 | `total_variance_condition` [B,1] float32. output: `noise_pred` [B,3,8].
All inputs are normalized exactly as described in `new_1200_deploy.json` (and implemented in `build_inputs`).
The planner's belief is the GP posterior mean and variance on a 51 x 51 grid (2 m spacing, 100 m x 100 m field).

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
- The reference clips waypoints to the field (x, y in 0..100 m, z in 10..40 m). The full simulation planner additionally
  keeps x, y at least 4 m away from the field edge before building the spline; add that clamp if you want identical flights.
