
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import matplotlib.pyplot as plt
import os
import sys
from pathlib import Path
import imageio.v2 as imageio
from matplotlib.patches import Rectangle
from gaussianprocesstraining import (
    build_correlated_noise_covariance,
    build_sensor_matrix,
    build_spline_trajectory_3d,
    create_plots_and_gifs,
    grid_measure,
    importance_filter,
    initialize_gp,
    kalman_update,
    noise_model,
    sample_correlated_sensor_noise,
)
from evalmetrics import compute_task_completion, compute_coverage_efficiency, compute_reconstruction_rmse, compute_rmse_time_metrics
import torch
import time
from CMAES_classic_singlemap import compute_fov, dynamics_3d, waypoint_3d

"""
gymnasium import
"""

import gymnasium as gym
from gymnasium import spaces 
from stable_baselines3 import PPO 
from stable_baselines3.common.env_checker import check_env 
import torch




SCRIPT_DIR = Path(__file__).resolve().parent
DIFFUSION_DIR = SCRIPT_DIR / "Diffusion"
if str(DIFFUSION_DIR) not in sys.path:
    sys.path.insert(0, str(DIFFUSION_DIR))

from sample_3d_sparse_trans_diffusion import diffusion, load_model, sample_sparse

from Diffusionplanner_singlemap import apply_measurement_update_3d, build_true_map_flat
SENSORNOISE_SEED = 123
MAPTYPE = os.environ.get("MAPTYPE", "grf")
CSV_PATH = SCRIPT_DIR / "csv"
TRUE_MAP_CACHE = {}

beta = 1.0
utility_threshold = 0.5
samplestep = 2.0
execution_steps_for_reward = 40

"""
def one_training_trial(): 
    initial_conditions = varmap, mean_map, (cx, cy), (vx, vy)
    initial_traj = sample_diffusion(varmap, mean_map, (cx, cy), (vx, vy))
    observation = initial_conditions, initial_traj
    residual_changes = PPO_policy.sample(observation)
    residual_changes = max_change * np.tanh(residual_changes)
    final_traj = initial_traj + residual_changes 
    final_conditions = simulate(final_traj)
    reward = reward_fn(
    initial_conditions, final_conditions, final_traj, residual_changes)

    PPO_store(observation, residual_changes, reward)

def_train(): 
    for i in range(number_of_trials): 
        one_training_trial()

    improve_PPO()


def main(): 

    

    residual_changes = PPO_network(initial_traj, initial_conditions) #PPO network can't be
      based on initial_reward, because we don't have that in the actual rollout, i think? 
    
    residual_changes = tanh() * residual_changes ##keeping the changes small

    final_traj = initial_traj + residual_changes 

    final_conditions = simulate(final_traj)

    reward = reward_fn(
    residual_changes, 
    initial traj,
    final_traj,
    initial_conditions, 
    final_conditions)

    improve_PPO(reward)


"""


'''
Map, planner, dynamics settings
'''
SCRIPT_DIR = Path(__file__).resolve().parent
TRAINING_MODULE_CANDIDATES = (
    SCRIPT_DIR / "threeDSparseTransDiffusion.py",
)
DEFAULT_CHECKPOINT = (
    DIFFUSION_DIR
    / "checkpoints"
    / "sparse_trans_waypoints_epoch_1900_multimodal_3d.pth"
)
CONDITON_INDEX = 1000
TRUTH_INDEX = 1000
SEED = None
CLIP = True
NUM_STEPS = None



def load_true_map_for_condition(condition_index, X_test, step):
    map_id = int(diffusion.map_ids[condition_index].detach().cpu().item())
    cache_key = (map_id, MAPTYPE)

    if cache_key in TRUE_MAP_CACHE:
        return TRUE_MAP_CACHE[cache_key]

    data = np.loadtxt(
        CSV_PATH / f"map_{map_id}_{MAPTYPE}_grid_counts.csv",
        delimiter=",",
        skiprows=1,
    )

    pts = data[:, 0:3]

    tol = 1e-9
    mask = (
        np.isclose(np.mod(pts[:, 0], step), 0.0, atol=tol)
        & np.isclose(np.mod(pts[:, 1], step), 0.0, atol=tol)
    )
    pts = pts[mask]

    true_map_flat = build_true_map_flat(pts, X_test)

    TRUE_MAP_CACHE[cache_key] = (true_map_flat, pts, map_id)

    return TRUE_MAP_CACHE[cache_key]






class ResidualDiffusionEnv(gym.Env):
    def __init__(self, diffusion_model): 
        super().__init__()

        self.diffusion_model = diffusion_model 
        (
            self.gp,
            self.X_test,
            self.mean,
            self.cov,
            self.xs,
            self.ys,
            self.X,
            self.Y,
            self.xmin,
            self.xmax,
            self.ymin,
            self.ymax,
            self.grid_step,
        ) = initialize_gp()

        self.action_space = spaces.Box( #normalized actions
            low = -1.0, 
            high = 1.0,
            shape = (24,),
            dtype = np.float32
        )

        obs_dim = 3*51*51 + 3 + 3 + 24 ##correct this, I don't think flattening everything would be the best

        self.observation_space = spaces.Box(low = -np.inf, 
                                            high = np.inf, 
                                            shape = (obs_dim,), 
                                            dtype = np.float32)
        
        self.max_delta = np.array([8.0, 8.0, 4.0], dtype = np.float32)



    def reset(self, seed = None, options = None): 
        super().reset(seed=seed)

        self.condition_index = CONDITON_INDEX
        self.meanvarmarker_map = diffusion.meanvarmarkermaps[self.condition_index:self.condition_index+1]
        self.current_position = diffusion.conditions[self.condition_index: self.condition_index+1]
        self.initial_heading_velocity = diffusion.initial_heading_velocities[self.condition_index : self.condition_index + 1]


        #sample initial proposal 
        self.initial_sample = sample_sparse(
                self.diffusion_model,
                condition_index=self.condition_index,
                seed=None,
                num_steps=NUM_STEPS,
                clip_x0=CLIP,
            )
        
        obs = self._make_obs()

        return obs, {}
    
    def _make_obs(self):
        obs = np.concatenate([
            self.meanvarmarker_map.detach().cpu().numpy().flatten(),
            self.current_position.detach().cpu().numpy().flatten(),
            self.initial_heading_velocity.detach().cpu().numpy().flatten(),
            self.initial_sample.detach().cpu().numpy().flatten(),
        ]).astype(np.float32)

        return obs
    
    def step(self, action): #training step

        #action shape was priorly flattened

        residual = action.reshape(8, 3)
        residual_world = residual * self.max_delta

        initial_traj_world = diffusion.extract_control_waypoints(self.initial_sample[0]).detach().cpu().numpy().T 

        final_traj_world = residual_world + initial_traj_world


        #clamping
        final_traj_world[:, 0] = np.clip(final_traj_world[:, 0], 0.0, diffusion.XY_SCALE)
        final_traj_world[:, 1] = np.clip(final_traj_world[:, 1], 0.0, diffusion.XY_SCALE)
        final_traj_world[:, 2] = np.clip(final_traj_world[:, 2], diffusion.Z_MIN, diffusion.Z_MAX)

        reward = simulate_compute_reward(initial_traj_world, final_traj_world, self.condition_index,
                                         self.gp, self.X_test, self.mean, self.cov, self.xs, self.ys, 
                                         self.X, self.Y, self.xmin, self.xmax, self.ymin, self.ymax, self.grid_step)

        terminated = True
        truncated = False
        obs = self._make_obs()

       
        info = {
            "initial_traj": initial_traj_world,
            "final_traj": final_traj_world,
        }

        return obs, float(reward), terminated, truncated, info
    





    





def train():
    diffusion_model = load_model(DEFAULT_CHECKPOINT, diffusion.device)

    env = ResidualDiffusionEnv(diffusion_model)

    check_env(env)

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=64,
        batch_size=32,
        n_epochs=10,
        gamma=1.0,
        verbose=1,
    )

    model.learn(total_timesteps=10)

    model.save("ppo_residual_policy")



def rollout_trajectory(
    traj_world,
    cx0,
    cy0,
    cz0,
    mu0,
    P0,
    true_map_flat,
    xs,
    ys,
    xmin,
    xmax,
    ymin,
    ymax,
    grid_step,
    rng,
    max_measurement_updates=None,
):
    mu = mu0.copy()
    P = P0.copy()
    cx, cy, cz = cx0, cy0, cz0

    dense_path = build_spline_trajectory_3d(
        cx,
        cy,
        cz,
        traj_world,
        samples_per_segment=5,
    )

    measurement_updates = 0

    for goal_x, goal_y, goal_z in dense_path:
        if (
            max_measurement_updates is not None
            and measurement_updates >= max_measurement_updates
        ):
            break

        grad_x, grad_y, grad_z, waypoint_reached = waypoint_3d(
            cx,
            cy,
            cz,
            goal_x=goal_x,
            goal_y=goal_y,
            goal_z=goal_z,
            step=grid_step,
        )

        if waypoint_reached:
            continue

        cx, cy, cz = dynamics_3d(
            cx,
            cy,
            cz,
            grad_x,
            grad_y,
            grad_z,
            samplestep,
            xmin,
            xmax,
            ymin,
            ymax,
            diffusion.Z_MIN,
            diffusion.Z_MAX,
            buffer=grid_step / 2,
        )

        mu, P = apply_measurement_update_3d(
            cx,
            cy,
            cz,
            mu,
            P,
            true_map_flat,
            xs,
            ys,
            rng,
        )
        measurement_updates += 1

    return mu, P



def simulate_compute_reward(initial_traj_world, final_traj_world, condition_index,
gp, X_test, mean, cov, xs, ys, X, Y, 
xmin, xmax, ymin, ymax, grid_step): 

    """
    At this stage we have the initial warm start traj, Post-PPO traj. 
    """

    current_mean = diffusion.means[condition_index].detach().cpu().numpy()
    current_var = diffusion.vars[condition_index].detach().cpu().numpy()
    
    mu = current_mean.flatten() 
    P = np.diag(current_var.flatten()) #P estimate since haven't stored covariances
    Pinit = P.copy()
    muinit = mu.copy()
    # gp, X_test, mean, cov, xs, ys, X, Y, xmin, xmax, ymin, ymax, grid_step = initialize_gp()

    true_map_flat, pts, map_id = load_true_map_for_condition(
        condition_index,
        X_test,
        grid_step,
    )

    cx0, cy0, cz0 = diffusion.raw_current_positions[condition_index].detach().cpu().numpy().tolist()

    initial_total_variance = np.trace(P)

    rng_ppo = np.random.default_rng(SENSORNOISE_SEED + map_id)
    mu_ppo, P_ppo = rollout_trajectory(
        final_traj_world,
        cx0,
        cy0,
        cz0,
        muinit,
        Pinit,
        true_map_flat,
        xs,
        ys,
        xmin,
        xmax,
        ymin,
        ymax,
        grid_step,
        rng_ppo,
        max_measurement_updates=execution_steps_for_reward,
    )

    final_PPO_variance = np.trace(P_ppo)
    variance_reduction_PPO = initial_total_variance - final_PPO_variance

    final_PPO_rmse_metrics = compute_reconstruction_rmse(
        mu=mu_ppo,
        pts=pts,
        xs=xs,
        ys=ys,
        step=grid_step,
        utility_threshold=utility_threshold,
        xmin=xmin,
        ymin=ymin,
    )

    final_global_PPO_rmse = final_PPO_rmse_metrics["global_rmse"]


    rng_diffusion = np.random.default_rng(SENSORNOISE_SEED + map_id)
    mu_diffusion, P_diffusion = rollout_trajectory(
        initial_traj_world,
        cx0,
        cy0,
        cz0,
        muinit,
        Pinit,
        true_map_flat,
        xs,
        ys,
        xmin,
        xmax,
        ymin,
        ymax,
        grid_step,
        rng_diffusion,
        max_measurement_updates=execution_steps_for_reward,
    )

    final_diffusion_variance = np.trace(P_diffusion)
    variance_reduction_diffusion = initial_total_variance - final_diffusion_variance

    final_diffusion_rmse_metrics = compute_reconstruction_rmse(
        mu=mu_diffusion,
        pts=pts,
        xs=xs,
        ys=ys,
        step=grid_step,
        utility_threshold=utility_threshold,
        xmin=xmin,
        ymin=ymin,
    )

    final_global_diffusion_rmse = final_diffusion_rmse_metrics["global_rmse"]


    rmse_improvement = final_global_diffusion_rmse - final_global_PPO_rmse

    occupied_rmse_improvement = final_diffusion_rmse_metrics["occupied_rmse"] - final_PPO_rmse_metrics["occupied_rmse"]

    residual_penalty = np.mean((final_traj_world - initial_traj_world)**2)




    # TODO: DEFINE REWARD FUNCTION AND CALCULATE REWARD
    reward = 10*rmse_improvement + 10*occupied_rmse_improvement - 0.01*residual_penalty

    print("rmse_improvement:", rmse_improvement,
    "occupied_rmse_improvement:", occupied_rmse_improvement,
    "residual_penalty:", residual_penalty,
    "reward:", reward)

    return float(reward)


if __name__ == "__main__": 
    train()
