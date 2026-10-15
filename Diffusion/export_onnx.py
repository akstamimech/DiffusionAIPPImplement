"""
Export a trained no-heading diffusion noise predictor (threeDSparseTransDiffusion_headingremoved.NoisePredictor)
to ONNX (fp32), and write everything an external runtime needs to reproduce sampling:
the normalization constants, noise schedule and coordinate bounds (deploy JSON) and a few real
input/output test vectors.

    python export_onnx.py                       # new_1200.pth, trained on HIGHOPT_DATASET_NAIP_141026.pt
    python export_onnx.py --checkpoint checkpoints/other.pth --dataset ../OTHER_DATASET.pt --out ../ONNX

IMPORTANT: the training module derives its normalization constants (belief mean/variance centers and
scales, total-variance stats) from the dataset the model was trained on, at import time. --dataset must
be that dataset or the exported model will be fed wrongly scaled inputs. (new_1200.pth was matched to
HIGHOPT_DATASET_NAIP_141026.pt by recomputing its training loss: 0.159 vs 0.154 recorded, against
0.40 / 0.43 for the other candidate datasets.)

Only the single denoising pass is exported. The DDIM sampling loop, the position-marker rendering and
the cubic spline are described in the deploy JSON and implemented in onnx_sample_reference.py.
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=Path, default=HERE / "checkpoints" / "new_1200.pth")
parser.add_argument("--dataset", type=Path, default=HERE.parent / "HIGHOPT_DATASET_NAIP_141026.pt")
parser.add_argument("--out", type=Path, default=HERE.parent / "ONNX")
parser.add_argument("--name", default="new_1200")
parser.add_argument("--opset", type=int, default=17)
args = parser.parse_args()

# the training module reads this at import time to build its normalization constants
os.environ["DIFFUSION_DATASET_PATH"] = str(args.dataset.resolve())
sys.path.insert(0, str(HERE))
os.chdir(HERE)

import numpy as np
import torch
import torch.nn as nn

import threeDSparseTransDiffusion_headingremoved as m

args.out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(0)


class ExactAdaptiveAvgPool2d(nn.Module):
    """AdaptiveAvgPool2d(out) as two fixed matrix multiplies. Adaptive average pooling averages each
    rectangular bin [floor(i*n/out), ceil((i+1)*n/out)) and a rectangle average is the product of its row
    and column averages, so this is exactly equal to torch's op while also exporting to ONNX (the
    exporter cannot express adaptive pooling when the input size is not a multiple of the output size)."""

    def __init__(self, in_size, out_size):
        super().__init__()
        A = torch.zeros(out_size, in_size)
        for i in range(out_size):
            lo, hi = (i * in_size) // out_size, math.ceil((i + 1) * in_size / out_size)
            A[i, lo:hi] = 1.0 / (hi - lo)
        self.register_buffer("A", A)

    def forward(self, x):                                   # [B, C, H, W] -> [B, C, out, out]
        return torch.einsum("oh,bchw,pw->bcop", self.A, x, self.A)


class Denoiser(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x, t, meanvarmarker_map, current_position, total_variance_condition):
        return self.net(x, t, meanvarmarker_map, current_position, total_variance_condition)


net = m.NoisePredictor()
ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
net.load_state_dict(ck["model_state_dict"])
net.eval()

# --- exact pooling replacement, verified against torch's own op on real activations ---
pool = net.mean_var_marker_cnn.pool2
size_in = 25
ref_in = torch.randn(4, 128, size_in, size_in)
exact = ExactAdaptiveAvgPool2d(size_in, m.TOKEN_SPATIAL_DISC)
pool_err = (pool(ref_in) - exact(ref_in)).abs().max().item()
assert pool_err < 1e-6, f"pooling replacement is not exact ({pool_err})"
print(f"adaptive pool replacement: max abs difference vs torch = {pool_err:.2e}")
net.mean_var_marker_cnn.pool2 = exact
model = Denoiser(net).eval()

# --- real inputs from the dataset the model was trained on (already normalized by the training module) ---
ds = m.TrajectoryDataset(m.trajectories, m.weights, m.meanvarmarkermaps, m.conditions, m.total_variance_conditions)
idx = torch.randperm(len(ds))[:64]
traj, pos, mvm, tv, _ = (torch.stack(z) for z in zip(*[ds[int(i)] for i in idx]))
t = torch.randint(0, m.T, (len(idx),)).long()
x_noisy, noise = m.forward_diffusion_sample(traj, t)
with torch.no_grad():
    y_torch = model(x_noisy, t, mvm, pos, tv)
print("input shapes:", {k: tuple(v.shape) for k, v in dict(x=x_noisy, t=t, map=mvm, position=pos, total_variance=tv).items()}, "-> output", tuple(y_torch.shape))

onnx_path = args.out / f"{args.name}_fp32.onnx"
names_in = ["x", "t", "meanvarmarker_map", "current_position", "total_variance_condition"]
torch.onnx.export(
    model, (x_noisy[:1], t[:1], mvm[:1], pos[:1], tv[:1]), str(onnx_path),
    input_names=names_in, output_names=["noise_pred"], opset_version=args.opset, do_constant_folding=True,
    dynamic_axes={n: {0: "batch"} for n in names_in + ["noise_pred"]}, dynamo=False,
)
print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")

# --- parity: onnxruntime fp32 vs torch ---
import onnx
import onnxruntime as ort

onnx.checker.check_model(onnx.load(str(onnx_path)))
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
feed = {"x": x_noisy.numpy(), "t": t.numpy(), "meanvarmarker_map": mvm.numpy(),
        "current_position": pos.numpy(), "total_variance_condition": tv.numpy()}
y_ort = sess.run(None, feed)[0]
diff = np.abs(y_ort - y_torch.numpy())
print(f"onnxruntime fp32 vs torch on {len(idx)} real inputs: max abs diff {diff.max():.2e}, mean {diff.mean():.2e}")
single = sess.run(None, {k: v[:1] for k, v in feed.items()})[0]
print(f"batch-1 run vs batched run: max abs diff {np.abs(single - y_ort[:1]).max():.2e}")

# --- deploy metadata: everything the runtime needs besides the network ---
sched = {k: getattr(m, k).tolist() for k in ("betas", "alphas_cumprod", "sqrt_alphas_cumprod", "sqrt_one_minus_alphas_cumprod") if hasattr(m, k)}
meta = {
    "checkpoint": args.checkpoint.name, "epoch": int(ck["epoch"]), "dataset_used_for_normalization": args.dataset.name,
    "inputs": {
        "x": "noisy waypoints [B,3,8] (normalized xyz in [-1,1])",
        "t": "int64 diffusion step [B], 0..T-1",
        "meanvarmarker_map": "float32 [B,3,51,51]: channel 0 = (mean - mean_center)/mean_scale, 1 = (variance - var_center)/var_scale, 2 = one-hot agent cell",
        "current_position": "float32 [B,3] agent xyz normalized to [-1,1]",
        "total_variance_condition": "float32 [B,1] = (log1p(sum of belief variance) - total_variance_center)/total_variance_scale",
    },
    "output": "predicted noise [B,3,8]",
    "normalization": {k: float(getattr(m, k)) for k in ("mean_center", "mean_scale", "var_center", "var_scale", "total_variance_center", "total_variance_scale")},
    "world": {"xy_scale_m": float(m.XY_SCALE), "z_min_m": float(m.Z_MIN), "z_max_m": float(m.Z_MAX), "grid": 51,
              "marker": "one cell at round((xy / xy_scale) * 50) clamped to 0..50; row = y, column = x"},
    "diffusion": {"T": int(m.T), "num_waypoints": int(m.NUM_CONTROL_WAYPOINTS), "schedule": "cosine, betas clamped to [1e-4, 0.999]", **sched},
    "sampling": "DDIM (see onnx_sample_reference.py): start from N(0,1) noise [1,3,8], loop t from T-1 down to 0, clip x0 to [-1,1]",
}
(args.out / f"{args.name}_deploy.json").write_text(json.dumps(meta, indent=1))
np.savez(args.out / f"{args.name}_test_vectors.npz", **feed, expected_noise_pred_torch=y_torch.numpy())
print("wrote deploy JSON and test vectors")
