"""
Reference cases for validating an external re-implementation of the imitation planner: real belief states from the
training dataset with the waypoints (meters) the PyTorch fp32 policy predicts for them. An external runtime
(planners/imitatetrans_onnx/onnx_reference.py) must reproduce them.
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]                         # scripts/
MODELS = HERE.parent                           # planners/imitatetrans_onnx
p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, default=ROOT / "planners" / "imitatetrans" / "checkpoints" / "new_imitate_1200.pth")
p.add_argument("--dataset", type=Path, default=ROOT / "HIGHOPT_DATASET_NAIP_141026.pt")
p.add_argument("--out", type=Path, default=MODELS / "new_imitate_1200_reference_cases.npz")
p.add_argument("--n", type=int, default=16)
a = p.parse_args()
os.environ["IMITATE_DATASET_PATH"] = str(a.dataset.resolve())

import numpy as np
import torch

from training.imitatetrans import ImitateTrans_headingremoved as m

net = m.WaypointPredictor()
ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
m.load_model_state_dict_compatible(net, m.remap_legacy_state_dict_keys(ck["model_state_dict"]))
net.eval()
raw = torch.load(a.dataset, map_location="cpu", weights_only=False)
rounds = raw["timestep"].numpy(); rng = np.random.default_rng(0)
cand = np.flatnonzero(raw["parent_beam_index"].numpy() == 0)
pick = []                                                             # spread over the 16 rounds
for r in np.linspace(0, 15, a.n).astype(int):
    pool = cand[rounds[cand] == r]; pick.append(int(rng.choice(pool)))
mean_map, var_map = raw["current_mean"][pick].float(), raw["current_var"][pick].float()
pos_world = raw["current_position"][pick].float()
expected = []
with torch.no_grad():
    for i in range(len(pick)):
        mm = ((mean_map[i] - m.mean_center) / m.mean_scale)
        vm = ((var_map[i] - m.var_center) / m.var_scale)
        marker = m.make_position_marker_maps(pos_world[i:i + 1, :2], grid_size=51, marker_radius=2)[0, 0]
        maps = torch.stack([mm, vm, marker], 0).unsqueeze(0)
        tv = m.normalize_total_variance(var_map[i].reshape(1, -1).sum(dim=1, keepdim=True))
        pos_n = m.normalize_xyz(pos_world[i:i + 1])
        wp = net(maps, pos_n, tv)
        expected.append(m.extract_control_waypoints(wp[0]).T.numpy())   # [8,3] meters
a.out.parent.mkdir(parents=True, exist_ok=True)
np.savez(a.out, mean_map=mean_map.numpy(), var_map=var_map.numpy(), position_xyz=pos_world.numpy(),
         expected_waypoints_m=np.stack(expected))
print(f"wrote {a.out} with {len(pick)} cases from rounds {sorted(set(rounds[pick].tolist()))}")
