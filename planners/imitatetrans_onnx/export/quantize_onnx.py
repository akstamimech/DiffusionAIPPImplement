"""
Quantize the exported fp32 ONNX imitation policy to INT8 and measure what it costs in accuracy.

Same recipe as the diffusion policy's (dynamic quantization, uint8 weights; activations quantized at run time):
  <name>_int8.onnx             per-channel weights   (recommended)
  <name>_int8_per_tensor.onnx  per-tensor weights    (slightly less accurate, possibly simpler kernels)

Accuracy is measured on real conditions from the dataset, against the fp32 model. The policy is deterministic (one
forward pass), so the waypoint error in meters is directly comparable between variants; no sampling noise is involved.
"""
# --- repo root on sys.path (this file lives in a sub-package) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[3]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ---
import argparse
import os
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                         # scripts/
MODELS = HERE.parent                           # planners/imitatetrans_onnx
p = argparse.ArgumentParser()
p.add_argument("--name", default="new_imitate_1200")
p.add_argument("--dataset", type=Path, default=ROOT / "HIGHOPT_DATASET_NAIP_141026.pt")
p.add_argument("--dir", type=Path, default=MODELS)
p.add_argument("--n", type=int, default=1024)
args = p.parse_args()
os.environ["IMITATE_DATASET_PATH"] = str(args.dataset.resolve())

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic

from training.imitatetrans import ImitateTrans_headingremoved as m

fp32 = args.dir / f"{args.name}_fp32.onnx"
VARIANTS = {
    "int8": lambda o: quantize_dynamic(fp32, o, weight_type=QuantType.QUInt8, per_channel=True),
    "int8_per_tensor": lambda o: quantize_dynamic(fp32, o, weight_type=QuantType.QUInt8),
}
out = {}
t0 = time.time()
for label, fn in VARIANTS.items():
    path = args.dir / f"{args.name}_{label}.onnx"
    fn(str(path))
    ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])        # must load and be runnable
    out[label] = path
    print(f"built {label}: {path.stat().st_size / 1e6:.2f} MB", flush=True)
print(f"quantization took {time.time() - t0:.0f} s")

# ---------------- evaluation on real conditions ----------------
idx = np.random.default_rng(0).permutation(len(m.meanvarmarkermaps))[:args.n]
feed = {"meanvarmarker_map": m.meanvarmarkermaps[idx].numpy(), "current_position": m.conditions[idx].numpy(),
        "total_variance_condition": m.total_variance_conditions[idx].numpy()}
target = m.trajectories[idx].numpy()                                            # the expert's waypoints, normalized [B,3,8]
span = np.array([m.XY_SCALE / 2, m.XY_SCALE / 2, (m.Z_MAX - m.Z_MIN) / 2]).reshape(1, 3, 1)   # normalized units -> meters


def predict(path):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {k: v[i:i + 128] for k, v in feed.items()})[0] for i in range(0, len(idx), 128)])


def latency(path, threads=1, reps=200):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    one = {k: v[:1] for k, v in feed.items()}
    for _ in range(20): s.run(None, one)
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); s.run(None, one); ts.append(time.perf_counter() - a)
    return 1000 * float(np.median(ts))


ref = predict(fp32)
print(f"\n{len(idx)} real conditions. loss = MSE to the expert's waypoints (normalized units); error = distance in meters per waypoint vs the fp32 model\n")
print(f"{'variant':<17}{'size MB':>8}{'ms/call*':>9}{'loss':>8}{'err mean m':>11}{'err p95 m':>10}{'err max m':>10}")
results = {}
for label, path in [("fp32", fp32)] + list(out.items()):
    pred = ref if label == "fp32" else predict(path)
    loss = float(((pred - target) ** 2).mean())
    err = np.linalg.norm((pred - ref) * span, axis=1).ravel()                   # [B*8] meters
    results[label] = err
    print(f"{label:<17}{path.stat().st_size / 1e6:>8.2f}{latency(path):>9.1f}{loss:>8.4f}{err.mean():>11.2f}"
          f"{np.percentile(err, 95):>10.2f}{err.max():>10.2f}", flush=True)
print("* single-thread x86 laptop timing of one network call, batch 1; NOT representative of a Raspberry Pi")
np.savez(args.dir / f"{args.name}_quant_report.npz", labels=list(results), **{f"err_{k}": v for k, v in results.items()})
