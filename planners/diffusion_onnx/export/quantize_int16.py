"""
Quantize the exported fp32 ONNX diffusion denoiser to 16 bit and measure what it costs in accuracy.

    python planners/diffusion_onnx/export/quantize_int16.py     ->  new_1200_int16.onnx

16-bit quantization is static (QDQ format): weights are int16 (per channel), activations uint16, and the activation
ranges are calibrated on real inputs (noised expert waypoints at random diffusion steps, with real belief maps), so unlike
the dynamic INT8 files this needs the dataset. It uses the com.microsoft Quantize/DequantizeLinear operators (16-bit is
not in the standard opset used here).

The calibration set must be large (ranges measured on a few hundred inputs saturate on unseen ones; see the imitation
model's quantize_int16.py: 256 -> worst waypoint off by 14 m, 2048 -> 0.2 m). Calibration and evaluation use disjoint
conditions.
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
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                         # scripts/
TRAIN = ROOT / "training" / "diffusion"        # the network, its checkpoints and normalization code
MODELS = HERE.parent                           # planners/diffusion_onnx: where the exported files live
p = argparse.ArgumentParser()
p.add_argument("--name", default="new_1200")
p.add_argument("--dataset", type=Path, default=ROOT / "HIGHOPT_DATASET_NAIP_141026.pt")
p.add_argument("--dir", type=Path, default=MODELS)
p.add_argument("--n-calib", type=int, default=4096)
p.add_argument("--n-eval", type=int, default=1024)
p.add_argument("--n-e2e", type=int, default=128)
args = p.parse_args()
os.environ["DIFFUSION_DATASET_PATH"] = str(args.dataset.resolve())
sys.path.insert(0, str(TRAIN)); sys.path.insert(0, str(args.dir)); os.chdir(TRAIN)

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType, quantize_static

import threeDSparseTransDiffusion_headingremoved as m
from onnx_sample_reference import Planner

fp32 = args.dir / f"{args.name}_fp32.onnx"
int16 = args.dir / f"{args.name}_int16.onnx"
deploy = args.dir / f"{args.name}_deploy.json"
perm = np.random.default_rng(0).permutation(len(m.meanvarmarkermaps))
calib_idx, eval_idx = perm[:args.n_calib], perm[-args.n_eval:]


def batch(indices, seed):
    """Real normalized conditions with expert waypoints noised to a random diffusion step: what the denoiser sees."""
    g = torch.Generator().manual_seed(seed)
    ix = torch.as_tensor(np.asarray(indices))
    traj, pos, mvm, tv = m.trajectories[ix], m.conditions[ix], m.meanvarmarkermaps[ix], m.total_variance_conditions[ix]
    t = torch.randint(0, m.T, (len(indices),), generator=g).long()
    noise = torch.randn(traj.shape, generator=g)
    a = m.sqrt_alphas_cumprod[t].view(-1, 1, 1); b = m.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
    x = a * traj + b * noise
    return dict(x=x.numpy(), t=t.numpy(), meanvarmarker_map=mvm.numpy(), current_position=pos.numpy(),
                total_variance_condition=tv.numpy()), noise.numpy()


class Reader(CalibrationDataReader):
    def __init__(self, chunks=64):
        self.items = iter([batch(calib_idx[i::chunks], seed=100 + i)[0] for i in range(chunks)])

    def get_next(self):
        return next(self.items, None)


t0 = time.time()
quantize_static(str(fp32), str(int16), Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                activation_type=QuantType.QUInt16, weight_type=QuantType.QInt16,
                calibrate_method=CalibrationMethod.MinMax,
                extra_options={"UseQDQContribOps": True, "ActivationSymmetric": False})
ort.InferenceSession(str(int16), providers=["CPUExecutionProvider"])           # must load and be runnable
print(f"built {int16.name}: {int16.stat().st_size / 1e6:.2f} MB in {time.time() - t0:.0f} s (calibrated on {args.n_calib} inputs)", flush=True)

# ---------------- evaluation on held-out real conditions ----------------
feed, noise = batch(eval_idx, seed=7)


def noise_metrics(path):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    pred = np.concatenate([s.run(None, {k: v[i:i + 128] for k, v in feed.items()})[0] for i in range(0, len(eval_idx), 128)])
    return pred, float(((pred - noise) ** 2).mean())


raw = torch.load(args.dataset, map_location="cpu", weights_only=False)
sel = np.random.default_rng(1).choice(len(raw["timestep"]), args.n_e2e, replace=False)
mm, vm, pw = raw["current_mean"][sel].float().numpy(), raw["current_var"][sel].float().numpy(), raw["current_position"][sel].float().numpy()
noises = np.random.default_rng(2).standard_normal((args.n_e2e, 1, 3, 8))
noises_b = np.random.default_rng(3).standard_normal((args.n_e2e, 1, 3, 8))


def e2e(path, nz):
    pl = Planner(path, deploy)
    res = []
    for i in range(args.n_e2e):
        maps, pos, tv = pl.build_inputs(mm[i], vm[i], pw[i])
        res.append(pl.denormalize(pl.sample_normalized(maps, pos, tv, initial_noise=nz[i], eta=0.0)))
    return np.stack(res)


def latency(path, reps=200):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    one = {k: v[:1] for k, v in feed.items()}
    for _ in range(20): s.run(None, one)
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); s.run(None, one); ts.append(time.perf_counter() - a)
    return 1000 * float(np.median(ts))


ref_pred, ref_loss = noise_metrics(fp32)
ref_wp = e2e(fp32, noises)
natural = np.linalg.norm(ref_wp - e2e(fp32, noises_b), axis=2)
print(f"\nfp32 reference: denoising loss {ref_loss:.4f}; natural spread of fp32 samples (same condition, different initial noise): "
      f"mean {natural.mean():.1f} m per waypoint\n")
print(f"{'variant':<9}{'size MB':>8}{'ms/call*':>9}{'loss':>8}{'noise MAE':>10}{'e2e mean m':>11}{'e2e p95 m':>10}{'e2e max m':>10}")
rows = [("fp32", fp32), ("int16", int16)]
int8 = args.dir / f"{args.name}_int8.onnx"
if int8.exists():
    rows.append(("int8", int8))
for label, path in rows:
    pred, loss = noise_metrics(path)
    wp = ref_wp if label == "fp32" else e2e(path, noises)
    err = np.linalg.norm(wp - ref_wp, axis=2).ravel()
    print(f"{label:<9}{path.stat().st_size / 1e6:>8.2f}{latency(path):>9.1f}{loss:>8.4f}{np.abs(pred - ref_pred).mean():>10.4f}"
          f"{err.mean():>11.3f}{np.percentile(err, 95):>10.3f}{err.max():>10.2f}", flush=True)
print("* single-thread x86 laptop timing of one network call, batch 1; NOT representative of a Raspberry Pi")
print("e2e = waypoint error in meters vs the fp32 model's own sample for the same condition and the same initial noise.")
