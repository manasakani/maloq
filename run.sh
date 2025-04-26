#!/bin/bash

#SBATCH --job-name=make_water
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --time=10:00
#SBATCH --error=error_file.err
#   SBATCH --partition=ocp

source activate pytorch

# export MODULEPATH=/opt/slurm/etc/files/modulesfiles/:$MODULEPATH 

module load cuda/11.6 \
    nccl/2.12.7-cuda.11.6 \
    nccl_efa/1.15.1-nccl.2.12.7-cuda.11.6

export CUDA_HOME=/opt/slurm/etc/files/modulesfiles/cuda/11.6
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# --> Dataset creation
# srun --cpu_bind=socket bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; 
# python make_water_cluster_dataset.py'

# --> Training script
srun --cpu_bind=socket bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; 
python train_water_clusters.py'