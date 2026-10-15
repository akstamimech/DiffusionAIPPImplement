#!/bin/sh
#
#SBATCH --job-name="diffusion"
#SBATCH --partition=gpu-a100-small
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-AE-MSc-AE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load 2025
module load cuda
module load python

source ~/.bashrc
conda activate diffusion_env 


/scratch/ajain3/conda/envs/diffusion_env/bin/python -c "import torch; print(torch.__version__)"
/scratch/ajain3/conda/envs/diffusion_env/bin/python -u continue_sparse_trans_training.py --checkpoint "/scratch/ajain3/diffusion/checkpoints/sparse_trans_waypoints_epoch_1800.pth" --target-epochs 3000
