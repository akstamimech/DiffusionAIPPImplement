"""
Reference runtime for the exported diffusion planner: numpy + onnxruntime only (no PyTorch).

    pip install numpy onnxruntime
    python onnx_sample_reference.py --model new_1200_int8.onnx --selftest

The ONNX file contains ONE denoising pass (predict the noise in the waypoints). Everything around it is
here: input normalization, rendering the agent-position marker, the DDIM loop, denormalization to meters,
and the cubic spline through the waypoints. All constants come from new_1200_deploy.json.

Typical use on the robot (once per replan):

    planner = Planner("new_1200_int8.onnx", "new_1200_deploy.json")
    waypoints, dense_path = planner.plan(mean_map, var_map, position_xyz)
        mean_map, var_map : float arrays [51, 51], rows = y, columns = x (the GP posterior mean and variance
                            on the 2 m grid over the 100 m x 100 m field)
        position_xyz      : (x, y, z) of the agent in meters (z in 10..40)
    -> waypoints [8, 3] meters (what the network predicts), dense_path [41, 3] (spline through the current
       position and the 8 waypoints, 5 points per segment, to be flown point by point).

Calling the network 20 times per replan is the cost; the belief-map branch inside the graph is recomputed
each call.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent


class Planner:
    def __init__(self, model_path, deploy_json, threads=None, seed=None):
        self.meta = json.loads(Path(deploy_json).read_text())
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        d = self.meta["diffusion"]
        self.T = d["T"]
        self.alphas_cumprod = np.asarray(d["alphas_cumprod"], dtype=np.float64)
        self.rng = np.random.default_rng(seed)
        self.n = self.meta["normalization"]
        self.w = self.meta["world"]

    # ----- inputs -----
    def _normalize_position(self, p):
        w = self.w
        return np.array([p[0] * 2 / w["xy_scale_m"] - 1, p[1] * 2 / w["xy_scale_m"] - 1,
                         (p[2] - w["z_min_m"]) * 2 / (w["z_max_m"] - w["z_min_m"]) - 1], dtype=np.float32)

    def build_inputs(self, mean_map, var_map, position_xyz):
        n, w = self.n, self.w
        grid = w["grid"]
        marker = np.zeros((grid, grid), dtype=np.float32)
        cx = int(np.clip(np.round(position_xyz[0] / w["xy_scale_m"] * (grid - 1)), 0, grid - 1))
        cy = int(np.clip(np.round(position_xyz[1] / w["xy_scale_m"] * (grid - 1)), 0, grid - 1))
        marker[cy, cx] = 1.0                                         # row = y, column = x
        maps = np.stack([(np.asarray(mean_map, np.float32) - n["mean_center"]) / n["mean_scale"],
                         (np.asarray(var_map, np.float32) - n["var_center"]) / n["var_scale"], marker])[None]
        tv = (np.log1p(max(float(np.sum(var_map)), 0.0)) - n["total_variance_center"]) / n["total_variance_scale"]
        return (maps.astype(np.float32), self._normalize_position(position_xyz)[None],
                np.array([[tv]], dtype=np.float32))

    # ----- one network call -----
    def predict_noise(self, x, t, maps, pos, tv):
        return self.session.run(None, {"x": x.astype(np.float32), "t": np.array([t], dtype=np.int64),
                                       "meanvarmarker_map": maps, "current_position": pos,
                                       "total_variance_condition": tv})[0]

    # ----- DDIM loop (matches ddim_sample in the training code; eta=0 is deterministic) -----
    def sample_normalized(self, maps, pos, tv, initial_noise=None, eta=0.0, clip_x0=True, num_steps=None):
        a = self.alphas_cumprod
        x = (self.rng.standard_normal((1, 3, self.meta["diffusion"]["num_waypoints"])) if initial_noise is None
             else np.asarray(initial_noise, dtype=np.float64)).astype(np.float64)
        steps = list(range(self.T - 1, -1, -1)) if not num_steps or num_steps >= self.T else \
            list(dict.fromkeys(np.linspace(self.T - 1, 0, num_steps).astype(np.int64).tolist()))
        for k, i in enumerate(steps):
            prev = steps[k + 1] if k + 1 < len(steps) else -1
            eps = self.predict_noise(x, i, maps, pos, tv).astype(np.float64)
            x0 = (x - np.sqrt(1 - a[i]) * eps) / np.sqrt(a[i])
            if clip_x0:
                x0 = np.clip(x0, -1.0, 1.0)
            if prev < 0:
                return x0
            var = eta ** 2 * (1 - a[prev]) / (1 - a[i]) * (1 - a[i] / a[prev])
            x = np.sqrt(a[prev]) * x0 + np.sqrt(1 - a[prev] - var) * eps
            if var > 0:
                x = x + np.sqrt(var) * self.rng.standard_normal(x.shape)
        return x

    def denormalize(self, wp):                                       # [1,3,8] normalized -> [8,3] meters
        w = self.w
        xy = (wp[0, :2].T + 1.0) * (w["xy_scale_m"] / 2.0)
        z = (wp[0, 2] + 1.0) * ((w["z_max_m"] - w["z_min_m"]) / 2.0) + w["z_min_m"]
        out = np.concatenate([xy, z[:, None]], axis=1)
        out[:, :2] = np.clip(out[:, :2], 0.0, w["xy_scale_m"])
        out[:, 2] = np.clip(out[:, 2], w["z_min_m"], w["z_max_m"])
        return out

    @staticmethod
    def cubic_spline(start_xyz, waypoints, samples_per_segment=5, eps=1e-9):
        """Natural cubic spline through [start, waypoints], parametrized by cumulative chord length
        (same construction as the training code); returns [samples_per_segment * K + 1, 3]."""
        pts = np.vstack([np.asarray(start_xyz, float)[None], waypoints]).astype(np.float64)
        h = np.maximum(np.linalg.norm(np.diff(pts, axis=0), axis=1), eps)
        N = len(pts)
        A = np.zeros((N, N)); rhs = np.zeros((N, 3)); A[0, 0] = A[-1, -1] = 1.0
        for i in range(1, N - 1):
            A[i, i - 1], A[i, i], A[i, i + 1] = h[i - 1], 2 * (h[i - 1] + h[i]), h[i]
            rhs[i] = 6 * ((pts[i + 1] - pts[i]) / h[i] - (pts[i] - pts[i - 1]) / h[i - 1])
        M = np.linalg.solve(A, rhs)
        out = []
        for i in range(N - 1):
            tau = np.linspace(0, 1, samples_per_segment + 1)[:-1] * h[i]
            a_, b_ = (h[i] - tau)[:, None], tau[:, None]
            out.append(M[i] * a_ ** 3 / (6 * h[i]) + M[i + 1] * b_ ** 3 / (6 * h[i])
                       + (pts[i] - M[i] * h[i] ** 2 / 6) * (a_ / h[i]) + (pts[i + 1] - M[i + 1] * h[i] ** 2 / 6) * (b_ / h[i]))
        out.append(pts[-1:])
        return np.vstack(out)

    def plan(self, mean_map, var_map, position_xyz, eta=0.0):
        maps, pos, tv = self.build_inputs(mean_map, var_map, position_xyz)
        waypoints = self.denormalize(self.sample_normalized(maps, pos, tv, eta=eta))
        return waypoints, self.cubic_spline(position_xyz, waypoints)


def selftest(model, deploy, cases):
    import time
    planner = Planner(model, deploy)
    z = np.load(cases)
    errs = []
    t0 = time.time()
    for i in range(len(z["position_xyz"])):
        maps, pos, tv = planner.build_inputs(z["mean_map"][i], z["var_map"][i], z["position_xyz"][i])
        wp = planner.denormalize(planner.sample_normalized(maps, pos, tv, initial_noise=z["initial_noise"][i], eta=0.0))
        errs.append(np.abs(wp - z["expected_waypoints_m"][i]).max())
    dt = (time.time() - t0) / len(errs)
    print(f"{len(errs)} reference cases: max abs waypoint difference vs the PyTorch fp32 sampler {max(errs):.3f} m "
          f"(mean of per-case max {np.mean(errs):.3f} m); {dt:.2f} s per full plan on this machine "
          f"({planner.T} network calls)")
    return errs


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=str(HERE / "new_1200_int8.onnx"))
    p.add_argument("--deploy", default=str(HERE / "new_1200_deploy.json"))
    p.add_argument("--cases", default=str(HERE / "new_1200_reference_cases.npz"))
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        selftest(a.model, a.deploy, a.cases)
    else:
        print(__doc__)
