#!/bin/sh
#
#SBATCH --job-name="mapeval_multimodal"
#SBATCH --partition=memory
#SBATCH --time=24:00:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#SBATCH --account=Education-AE-MSc-AE
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

module load 2025
module load cuda
module load python
module load openmpi

source ~/.bashrc
conda activate diffusion_env


export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

export OMPI_MCA_opal_cuda_support=true

srun --mpi=pmix /scratch/ajain3/conda/envs/diffusion_env/bin/python -u DataCollector_3D_randomstart_CMAESregularized.py

