"""
Quantize the exported fp32 ONNX imitation policy to 16 bit and measure what it costs in accuracy.

    python planners/imitatetrans_onnx/export/quantize_int16.py     ->  new_imitate_1200_int16.onnx

16-bit quantization is static (QDQ format): weights are int16 (per channel), activations uint16, and the activation
ranges are calibrated on real belief states, so unlike the dynamic INT8 files this needs the dataset. It uses the
com.microsoft Quantize/DequantizeLinear operators (16-bit is not in the standard opset used here).

The calibration set must be large: ranges measured on 256 conditions saturated on ~3% of unseen ones (worst waypoint
off by 14 m); with 2048 the worst case is 0.2 m. Calibration and evaluation use disjoint conditions.
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
p.add_argument("--n-calib", type=int, default=2048)
p.add_argument("--n-eval", type=int, default=1024)
args = p.parse_args()
os.environ["IMITATE_DATASET_PATH"] = str(args.dataset.resolve())

import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

from training.imitatetrans import ImitateTrans_headingremoved as m

fp32 = args.dir / f"{args.name}_fp32.onnx"
int16 = args.dir / f"{args.name}_int16.onnx"
perm = np.random.default_rng(0).permutation(len(m.meanvarmarkermaps))
calib_idx, eval_idx = perm[:args.n_calib], perm[-args.n_eval:]


def feed(idx):
    return {"meanvarmarker_map": m.meanvarmarkermaps[idx].numpy(), "current_position": m.conditions[idx].numpy(),
            "total_variance_condition": m.total_variance_conditions[idx].numpy()}


class Reader(CalibrationDataReader):
    def __init__(self, chunks=64):
        self.items = iter([feed(calib_idx[i::chunks]) for i in range(chunks)])

    def get_next(self):
        return next(self.items, None)


t0 = time.time()
quantize_static(str(fp32), str(int16), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QUInt16, weight_type=QuantType.QInt16,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={"UseQDQContribOps": True, "ActivationSymmetric": False})
ort.InferenceSession(str(int16), providers=["CPUExecutionProvider"])           # must load and be runnable
print(f"built {int16.name}: {int16.stat().st_size / 1e6:.2f} MB in {time.time() - t0:.0f} s (calibrated on {args.n_calib} conditions)")

# ---------------- evaluation on held-out real conditions ----------------
F = feed(eval_idx)
target = m.trajectories[eval_idx].numpy()
span = np.array([m.XY_SCALE / 2, m.XY_SCALE / 2, (m.Z_MAX - m.Z_MIN) / 2]).reshape(1, 3, 1)


def predict(path):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return np.concatenate([s.run(None, {k: v[i:i + 128] for k, v in F.items()})[0] for i in range(0, len(eval_idx), 128)])


def latency(path, reps=200):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    one = {k: v[:1] for k, v in F.items()}
    for _ in range(20): s.run(None, one)
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); s.run(None, one); ts.append(time.perf_counter() - a)
    return 1000 * float(np.median(ts))


ref = predict(fp32)
print(f"\n{len(eval_idx)} held-out real conditions. loss = MSE to the expert's waypoints (normalized); error = meters per waypoint vs the fp32 model\n")
print(f"{'variant':<9}{'size MB':>8}{'ms/call*':>9}{'loss':>8}{'err mean m':>11}{'err p95 m':>10}{'err max m':>10}")
rows = [("fp32", fp32), ("int16", int16)]
int8 = args.dir / f"{args.name}_int8.onnx"
if int8.exists():
    rows.append(("int8", int8))
for label, path in rows:
    pred = ref if label == "fp32" else predict(path)
    err = np.linalg.norm((pred - ref) * span, axis=1).ravel()
    print(f"{label:<9}{path.stat().st_size / 1e6:>8.2f}{latency(path):>9.1f}{float(((pred - target) ** 2).mean()):>8.4f}"
          f"{err.mean():>11.3f}{np.percentile(err, 95):>10.3f}{err.max():>10.2f}", flush=True)
print("* single-thread x86 laptop timing of one network call, batch 1; NOT representative of a Raspberry Pi")
