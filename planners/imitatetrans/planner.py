"""
The imitation baseline: the same conditioning and the same transformer backbone as the diffusion policy, but it
regresses the 8 expert waypoints in a single forward pass (so it averages over the expert's modes instead of
sampling one). A natural cubic spline through the pose and the waypoints is the plan.
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
TRAINING_DIR = ROOT                                                  # ImitateTrans_headingremoved.py (network + training)
DEFAULT_DATASET = ROOT / "HIGHOPT_DATASET_NAIP_141026.pt"             # the normalization constants come from this dataset (what new_imitate_1200 was trained on)
# This checkpoint must come from the current (3D) architecture: checkpoints saved before the 2D -> 3D port will not
# load into WaypointPredictor and need to be retrained.
DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "checkpoints" / "new_imitate_1200.pth"


def import_imitation_module(dataset_path=None):
    """The ImitateTrans module derives the belief normalization constants from the training dataset when imported."""
    os.environ.setdefault("IMITATE_DATASET_PATH", str(dataset_path or DEFAULT_DATASET))
    if str(TRAINING_DIR) not in sys.path:
        sys.path.insert(0, str(TRAINING_DIR))
    from training.imitatetrans import ImitateTrans_headingremoved as imitate_trans
    return imitate_trans


class ImitationPlanner(TorchPolicyPlanner):
    def __init__(self, grid: GridSpec, checkpoint_path=None, module=None):
        imitate_trans = module or import_imitation_module()
        model = imitate_trans.WaypointPredictor().to(imitate_trans.device)
        load_state_dict_into(imitate_trans, model, checkpoint_path or DEFAULT_CHECKPOINT)
        model.eval()
        imitate_trans.model = model
        super().__init__(imitate_trans, model, grid)

    @torch.no_grad()
    def plan(self, belief, pose) -> FlightPlan:
        started = time.time()
        self.model.eval()
        self.m.model = self.model
        meanvarmarker_map, position_model, total_variance_condition, position_world = self.conditioning(belief, pose)
        normalized_waypoints = self.model(meanvarmarker_map, position_model, total_variance_condition)
        return self.to_plan(normalized_waypoints[0], position_world, started)
