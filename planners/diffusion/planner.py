"""
The diffusion planner with the PyTorch network: each replan denoises 8 control waypoints with DDIM, conditioned on the
belief maps, the position and the total variance, then a natural cubic spline through the pose and the waypoints is the
plan. The network code and its normalization constants live in the diffusion training package.
"""
import os
import sys
import time
from pathlib import Path

import torch

from aipp.config import GridSpec
from aipp.interfaces import FlightPlan

from planners._torch_policy import TorchPolicyPlanner, load_state_dict_into

ROOT = Path(__file__).resolve().parents[2]
TRAINING_DIR = ROOT / "training" / "diffusion"                                # network, sampler and normalization (training code)
DEFAULT_DATASET = ROOT / "HIGHOPT_DATASET_NAIP_141026.pt"        # the normalization constants come from this dataset
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "new_1200.pth"


def import_diffusion_module(dataset_path=None):
    """The sampler module derives the belief normalization constants from the training dataset when imported."""
    os.environ.setdefault("DIFFUSION_DATASET_PATH", str(dataset_path or DEFAULT_DATASET))
    if str(TRAINING_DIR) not in sys.path:
        sys.path.insert(0, str(TRAINING_DIR))
    from sample_3d_sparse_trans_diffusion_headingremoved import diffusion
    return diffusion


class DiffusionPlanner(TorchPolicyPlanner):
    def __init__(self, grid: GridSpec, checkpoint_path=None, eta=0.0, num_steps=None, clip_x0=True, module=None):
        """`module`: the imported sampler module (import_diffusion_module()); pass it when something (seeding) has to
        happen between importing the network code and building the model."""
        diffusion = module or import_diffusion_module()
        model = diffusion.NoisePredictor().to(diffusion.device)
        load_state_dict_into(diffusion, model, checkpoint_path or DEFAULT_CHECKPOINT)
        model.eval()
        diffusion.model = model
        super().__init__(diffusion, model, grid)
        self.eta, self.num_steps, self.clip_x0 = eta, num_steps, clip_x0

    @torch.no_grad()
    def plan(self, belief, pose) -> FlightPlan:
        started = time.time()
        d = self.m
        self.model.eval()
        d.model = self.model
        meanvarmarker_map, position_model, total_variance_condition, position_world = self.conditioning(belief, pose)
        sparse_noise = torch.randn((1, *d.TARGET_SHAPE), device=d.device)
        sparse_sample, _, _ = d.ddim_sample(
            sparse_noise,
            meanvarmarker_map,
            current_position=position_model,
            total_variance_condition=total_variance_condition,
            num_steps=self.num_steps,
            clip_x0=self.clip_x0,
            eta=self.eta,
        )
        return self.to_plan(sparse_sample[0], position_world, started)
