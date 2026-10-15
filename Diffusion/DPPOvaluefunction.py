from symtable import Class

import numpy as np
import matplotlib

from threeDSparseTransDiffusion import MeanVarMarkerCNN

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from torch import dtype, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
from pathlib import Path
from tqdm import tqdm
from scipy.interpolate import CubicSpline
    


class ValueFunction(nn.Module):
    def __init__(self, token_dim = 196, base_channels = 64, hidden_dim = 512, pos_dim = 3, vel_dim = 3):
        super(ValueFunction, self).__init__()
        #reusing the map encoder architecture for the value function
        self.mean_var_marker_cnn = MeanVarMarkerCNN(
            input_channels=3, hidden_dim=base_channels, token_dim=token_dim
        )

        #for (cx, cy, cz) and (vx, vy, vz)
        self.state_mlp = nn.Sequential(
            nn.Linear(pos_dim + vel_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, token_dim),
        )

        #value head 
        self.value_head = nn.Sequential(
            nn.LayerNorm(token_dim),
            nn.Linear(token_dim, hidden_dim), 
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)) 
        

    def forward(self, meanvarmarker_map, current_position, initial_heading_velocity):
        map_tokens = self.mean_var_marker_cnn(meanvarmarker_map) #[B, 64, token_dim]
        map_feat = map_tokens.mean(dim = 1) #[B, token_dim]

        state_input = torch.cat([current_position, initial_heading_velocity], dim = -1) #[B, 6]
        state_feat = self.state_mlp(state_input) #[B, token_dim]

        combined_feat = map_feat + state_feat #[B, token_dim]
        value = self.value_head(combined_feat) #[B, 1]
        return value
        

