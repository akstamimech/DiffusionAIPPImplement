"""
Quantize the exported fp32 ONNX denoiser to INT8 and measure what it costs in accuracy.

    python quantize_onnx.py                  # builds all variants, prints a comparison, writes <name>_int8.onnx (best)

Variants (all built from ONNX/<name>_fp32.onnx, calibrated on real HIGHOPT conditions):
  dyn_matmul : dynamic quantization of MatMul/Gemm weights only (convolutions stay fp32)
  dyn_all    : dynamic quantization including Conv
  static_mm  : static QDQ INT8 (weights per-channel, activations calibrated with MinMax)
  static_pct : static QDQ INT8, activations calibrated with the 99.999th percentile (robust to outliers)

Accuracy is judged against fp32 on held-out real data, in the units that matter:
  1. denoising loss (the training objective) and noise-prediction error on 1024 held-out (sample, t, noise) draws
  2. END-TO-END: 20-step DDIM sampled waypoints for 128 real conditions, same initial noise, error in METERS,
     compared with the natural seed-to-seed spread of the fp32 model itself
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
p = argparse.ArgumentParser()
p.add_argument("--name", default="new_1200")
p.add_argument("--dataset", type=Path, default=HERE.parent / "HIGHOPT_DATASET_NAIP_141026.pt")
p.add_argument("--dir", type=Path, default=HERE.parent / "ONNX")
p.add_argument("--n-e2e", type=int, default=128)
args = p.parse_args()
os.environ["DIFFUSION_DATASET_PATH"] = str(args.dataset.resolve())
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(args.dir)); os.chdir(HERE)

import numpy as np
import torch
import onnxruntime as ort
from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)
import threeDSparseTransDiffusion_headingremoved as m
from onnx_sample_reference import Planner

fp32 = args.dir / f"{args.name}_fp32.onnx"; deploy = args.dir / f"{args.name}_deploy.json"
NAMES = ["x", "t", "meanvarmarker_map", "current_position", "total_variance_condition"]

# ---------------- data: normalized real inputs, disjoint calibration / evaluation sets ----------------
ds = m.TrajectoryDataset(m.trajectories, m.weights, m.meanvarmarkermaps, m.conditions, m.total_variance_conditions)
perm = np.random.default_rng(0).permutation(len(ds))
calib_idx, eval_idx = perm[:256], perm[256:256 + 1024]


def batch(indices, seed):
    g = torch.Generator().manual_seed(seed)
    traj, pos, mvm, tv, _ = (torch.stack(z) for z in zip(*[ds[int(i)] for i in indices]))
    t = torch.randint(0, m.T, (len(indices),), generator=g).long()
    noise = torch.randn(traj.shape, generator=g)
    a = m.sqrt_alphas_cumprod[t].view(-1, 1, 1); b = m.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
    x = a * traj + b * noise
    return dict(x=x.numpy(), t=t.numpy(), meanvarmarker_map=mvm.numpy(), current_position=pos.numpy(),
                total_variance_condition=tv.numpy()), noise.numpy()


class Reader(CalibrationDataReader):
    def __init__(self, chunks=16):
        self.items = iter([batch(calib_idx[i::chunks], seed=100 + i)[0] for i in range(chunks)])

    def get_next(self):
        return next(self.items, None)


# ---------------- variants ----------------
out = {}
t0 = time.time()
OUT_HEAD = ["/net/output/output.1/MatMul", "/net/output/output.3/MatMul"]       # final layers that produce the noise estimate
VARIANTS = {
    "dyn_matmul": lambda o: quantize_dynamic(fp32, o, weight_type=QuantType.QInt8, op_types_to_quantize=["MatMul", "Gemm"]),
    "dyn_mm_u8": lambda o: quantize_dynamic(fp32, o, weight_type=QuantType.QUInt8, op_types_to_quantize=["MatMul", "Gemm"]),
    "dyn_all": lambda o: quantize_dynamic(fp32, o, weight_type=QuantType.QUInt8),
    "dyn_all_headfp32": lambda o: quantize_dynamic(fp32, o, weight_type=QuantType.QUInt8, nodes_to_exclude=OUT_HEAD),
    "dyn_all_perch": lambda o: quantize_dynamic(fp32, o, weight_type=QuantType.QUInt8, per_channel=True),
    "static_mm": lambda o: quantize_static(fp32, o, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                                           activation_type=QuantType.QInt8, weight_type=QuantType.QInt8,
                                           calibrate_method=CalibrationMethod.MinMax),
}
for label, fn in VARIANTS.items():
    path = args.dir / f"{args.name}_int8_{label}.onnx"
    try:
        fn(str(path))
        ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])        # must load and be runnable
        out[label] = path
        print(f"built {label}: {path.stat().st_size / 1e6:.2f} MB", flush=True)
    except Exception as e:
        print(f"FAILED {label}: {type(e).__name__}: {str(e)[:160]}", flush=True)
print(f"quantization took {time.time() - t0:.0f} s")

# ---------------- evaluation ----------------
feed, noise = batch(eval_idx, seed=7)
w = np.ones(len(eval_idx))


def noise_metrics(path):
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    pred = np.concatenate([s.run(None, {k: v[i:i + 128] for k, v in feed.items()})[0] for i in range(0, len(eval_idx), 128)])
    return pred, float(((pred - noise) ** 2).mean())


ref_pred, ref_loss = noise_metrics(fp32)
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


ref_wp = e2e(fp32, noises)
other_seed = e2e(fp32, noises_b)
natural = np.linalg.norm(ref_wp - other_seed, axis=2)                      # distance per waypoint between two different-noise fp32 samples
print(f"\nfp32 reference: denoising loss {ref_loss:.4f} (training objective, held-out real data)")
print(f"natural spread of fp32 samples (same condition, different initial noise): mean {natural.mean():.1f} m per waypoint, median {np.median(natural):.1f} m\n")


def latency(path, threads=1, reps=200):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    one = {k: v[:1] for k, v in feed.items()}
    for _ in range(20): s.run(None, one)
    ts = []
    for _ in range(reps):
        a = time.perf_counter(); s.run(None, one); ts.append(time.perf_counter() - a)
    return 1000 * float(np.median(ts))


rows = [("fp32", fp32)] + list(out.items())
print(f"{'variant':<11}{'size MB':>8}{'ms/call*':>9}{'loss':>8}{'noise MAE':>10}{'e2e mean m':>11}{'e2e p95 m':>10}{'e2e max m':>10}")
results = {}
for label, path in rows:
    pred, loss = noise_metrics(path)
    wp = e2e(path, noises) if label != "fp32" else ref_wp
    err = np.linalg.norm(wp - ref_wp, axis=2).ravel()
    results[label] = (loss, float(np.abs(pred - ref_pred).mean()), err)
    print(f"{label:<11}{path.stat().st_size / 1e6:>8.2f}{latency(path):>9.1f}{loss:>8.4f}{np.abs(pred - ref_pred).mean():>10.4f}"
          f"{err.mean():>11.2f}{np.percentile(err, 95):>10.2f}{err.max():>10.2f}", flush=True)
print("* single-thread x86 laptop timing of one network call, batch 1; NOT representative of a Raspberry Pi")
print("e2e = waypoint error in meters vs the fp32 model's own sample for the same condition and the same initial noise;")
print("      compare with the natural spread above (how much two fp32 samples differ for the same condition).")
np.savez(args.dir / f"{args.name}_quant_report.npz", labels=[r[0] for r in rows], **{f"err_{k}": v[2] for k, v in results.items()})
