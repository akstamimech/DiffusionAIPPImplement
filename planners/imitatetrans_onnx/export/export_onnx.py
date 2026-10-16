"""
Export a trained imitation policy (ImitateTrans_headingremoved.WaypointPredictor) to ONNX (fp32), and write everything an
external runtime needs to reproduce it: the normalization constants and coordinate bounds (deploy JSON) and a few real
input/output test vectors.

    python planners/imitatetrans_onnx/export/export_onnx.py     # new_imitate_1200.pth, normalized with HIGHOPT_DATASET_NAIP_141026.pt

IMPORTANT: the training module derives its normalization constants (belief mean/variance centers and scales,
total-variance stats) from the dataset the model was trained on, at import time. --dataset must be that dataset or the
exported model will be fed wrongly scaled inputs.

Unlike the diffusion policy there is no sampling loop: ONE network call is the whole policy. It maps the belief maps,
the position and the total variance to the 8 control waypoints (normalized xyz in [-1, 1]); onnx_reference.py shows the
few steps around it (input normalization, the position marker, denormalization, the cubic spline).
"""
# --- repo root on sys.path (this file lives in a sub-package) ---
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parents[3]
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))
# ---
import argparse
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                         # scripts/
MODELS = HERE.parent                           # planners/imitatetrans_onnx: where the exported files live
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--checkpoint", type=Path, default=ROOT / "planners" / "imitatetrans" / "checkpoints" / "new_imitate_1200.pth")
parser.add_argument("--dataset", type=Path, default=ROOT / "HIGHOPT_DATASET_NAIP_141026.pt")
parser.add_argument("--out", type=Path, default=MODELS)
parser.add_argument("--name", default="new_imitate_1200")
parser.add_argument("--opset", type=int, default=17)
args = parser.parse_args()

# the training module reads this at import time to build its normalization constants
os.environ["IMITATE_DATASET_PATH"] = str(args.dataset.resolve())

import numpy as np
import torch
import torch.nn as nn

from training.imitatetrans import ImitateTrans_headingremoved as m

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


class Policy(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, meanvarmarker_map, current_position, total_variance_condition):
        return self.net(meanvarmarker_map, current_position, total_variance_condition)


net = m.WaypointPredictor()
ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
m.load_model_state_dict_compatible(net, m.remap_legacy_state_dict_keys(ck["model_state_dict"]))
net.eval()

# --- exact pooling replacement, verified against torch's own op on random activations ---
pool = net.mean_var_marker_cnn.pool2
size_in = 25
ref_in = torch.randn(4, 128, size_in, size_in)
exact = ExactAdaptiveAvgPool2d(size_in, m.TOKEN_SPATIAL_DISC)
pool_err = (pool(ref_in) - exact(ref_in)).abs().max().item()
assert pool_err < 1e-6, f"pooling replacement is not exact ({pool_err})"
print(f"adaptive pool replacement: max abs difference vs torch = {pool_err:.2e}")
net.mean_var_marker_cnn.pool2 = exact
model = Policy(net).eval()

# --- real inputs from the dataset the model was trained on (already normalized by the training module) ---
idx = torch.randperm(len(m.meanvarmarkermaps))[:64]
mvm, pos, tv = m.meanvarmarkermaps[idx], m.conditions[idx], m.total_variance_conditions[idx]
with torch.no_grad():
    y_torch = model(mvm, pos, tv)
print("input shapes:", {k: tuple(v.shape) for k, v in dict(map=mvm, position=pos, total_variance=tv).items()}, "-> output", tuple(y_torch.shape))

onnx_path = args.out / f"{args.name}_fp32.onnx"
names_in = ["meanvarmarker_map", "current_position", "total_variance_condition"]
torch.onnx.export(
    model, (mvm[:1], pos[:1], tv[:1]), str(onnx_path),
    input_names=names_in, output_names=["waypoints_normalized"], opset_version=args.opset, do_constant_folding=True,
    dynamic_axes={n: {0: "batch"} for n in names_in + ["waypoints_normalized"]}, dynamo=False,
)
print(f"wrote {onnx_path} ({onnx_path.stat().st_size / 1e6:.2f} MB)")

# --- parity: onnxruntime fp32 vs torch ---
import onnx
import onnxruntime as ort

onnx.checker.check_model(onnx.load(str(onnx_path)))
sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
feed = {"meanvarmarker_map": mvm.numpy(), "current_position": pos.numpy(), "total_variance_condition": tv.numpy()}
y_ort = sess.run(None, feed)[0]
diff = np.abs(y_ort - y_torch.numpy())
print(f"onnxruntime fp32 vs torch on {len(idx)} real inputs: max abs diff {diff.max():.2e}, mean {diff.mean():.2e}")
single = sess.run(None, {k: v[:1] for k, v in feed.items()})[0]
print(f"batch-1 run vs batched run: max abs diff {np.abs(single - y_ort[:1]).max():.2e}")

# --- deploy metadata: everything the runtime needs besides the network ---
meta = {
    "checkpoint": args.checkpoint.name, "epoch": int(ck["epoch"]), "dataset_used_for_normalization": args.dataset.name,
    "inputs": {
        "meanvarmarker_map": "float32 [B,3,51,51]: channel 0 = (mean - mean_center)/mean_scale, 1 = (variance - var_center)/var_scale, 2 = one-hot agent cell",
        "current_position": "float32 [B,3] agent xyz normalized to [-1,1]",
        "total_variance_condition": "float32 [B,1] = (log1p(sum of belief variance) - total_variance_center)/total_variance_scale",
    },
    "output": "waypoints_normalized float32 [B,3,8]: the 8 control waypoints, xyz normalized to [-1,1]",
    "normalization": {k: float(getattr(m, k)) for k in ("mean_center", "mean_scale", "var_center", "var_scale", "total_variance_center", "total_variance_scale")},
    "world": {"xy_scale_m": float(m.XY_SCALE), "z_min_m": float(m.Z_MIN), "z_max_m": float(m.Z_MAX), "grid": 51,
              "marker": "one cell at round((xy / xy_scale) * 50) clamped to 0..50; row = y, column = x"},
    "policy": {"num_waypoints": int(m.NUM_CONTROL_WAYPOINTS),
               "use": "ONE network call per plan (no sampling loop). meters = clamp(denormalize(output)): x, y = (v + 1) * xy_scale / 2 clamped to 0..xy_scale; "
                      "z = (v + 1) * (z_max - z_min) / 2 + z_min clamped to z_min..z_max. See onnx_reference.py."},
}
(args.out / f"{args.name}_deploy.json").write_text(json.dumps(meta, indent=1))
np.savez(args.out / f"{args.name}_test_vectors.npz", **feed, expected_waypoints_normalized_torch=y_torch.numpy())
print("wrote deploy JSON and test vectors")
