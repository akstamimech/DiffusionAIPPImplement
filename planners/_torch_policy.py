"""
What the two PyTorch planners (diffusion, imitation) share: turning the belief into the network's conditioning
tensors and turning the network's normalized waypoints into a clamped, splined flight plan.

Both network modules (the diffusion sampler module and ImitateTrans) expose the same helpers, because the imitation
model was written as a copy of the diffusion one: the belief normalization constants (derived from the training
dataset when the module is imported), the position marker map, waypoint extraction and the torch cubic spline.
"""
import time

import numpy as np
import torch

from aipp.config import GridSpec
from aipp.interfaces import FlightPlan


class TorchPolicyPlanner:
    def __init__(self, module, model, grid: GridSpec):
        self.m, self.model, self.grid = module, model, grid
        self.z_range = (float(module.Z_MIN), float(module.Z_MAX))

    def conditioning(self, belief, pose):
        """(belief maps with position marker [1, 3, 51, 51], normalized position [1, 3], total variance [1, 1],
        world position [1, 3]) as the network expects them."""
        m = self.m
        position_world = torch.as_tensor(pose, dtype=torch.float32, device=m.device).view(1, 3)
        position_model = m.normalize_xyz(position_world)
        current_mean = torch.tensor(belief.mean_map(), dtype=torch.float32, device=m.device)
        current_var = torch.tensor(belief.var_map(), dtype=torch.float32, device=m.device)
        mean_map = (current_mean - m.mean_center.to(m.device)) / m.mean_scale.to(m.device)
        total_variance_condition = m.normalize_total_variance(
            current_var.reshape(1, -1).sum(dim=1, keepdim=True)
        ).to(m.device)
        var_map = (current_var - m.var_center.to(m.device)) / m.var_scale.to(m.device)
        marker_map = m.make_position_marker_maps(
            position_world[:, :2].cpu(), grid_size=51, marker_radius=2,
        ).to(m.device)[0, 0]
        meanvarmarker_map = torch.stack([mean_map, var_map, marker_map], dim=0).unsqueeze(0)
        return meanvarmarker_map, position_model, total_variance_condition, position_world

    def to_plan(self, normalized_sample, position_world, started) -> FlightPlan:
        """Network output [3, 8] (normalized) -> control waypoints, spline through them, both clamped into the padded
        field; returned as arrays of shape [8, 3] and [41, 3]."""
        m, g = self.m, self.grid
        control_waypoints = m.extract_control_waypoints(normalized_sample)
        dense_traj = m.pytorch_cubic_spline(control_waypoints, current_position=position_world[0])[0]
        zmin, zmax = self.z_range
        padding = g.step * 2
        dense_traj[0] = dense_traj[0].clamp(g.xmin + padding, g.xmax - padding)
        dense_traj[1] = dense_traj[1].clamp(g.ymin + padding, g.ymax - padding)
        dense_traj[2] = dense_traj[2].clamp(zmin, zmax)
        control_waypoints[0] = control_waypoints[0].clamp(g.xmin + padding, g.xmax - padding)
        control_waypoints[1] = control_waypoints[1].clamp(g.ymin + padding, g.ymax - padding)
        control_waypoints[2] = control_waypoints[2].clamp(zmin, zmax)
        return FlightPlan(
            waypoints=control_waypoints.detach().cpu().numpy().T.copy(),
            path=dense_traj.detach().cpu().numpy().T.copy(),
            seconds=time.time() - started,
        )


def load_state_dict_into(module, model, checkpoint_path):
    """Load a checkpoint into `model` the way both training modules saved them (legacy key names are remapped)."""
    checkpoint = torch.load(checkpoint_path, map_location=module.device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = module.remap_legacy_state_dict_keys(state_dict)
    module.load_model_state_dict_compatible(model, state_dict)
    return model
