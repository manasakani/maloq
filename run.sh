#!/bin/bash

#SBATCH --job-name=make_water
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --error=error_file.err
#   SBATCH --partition=ocp

source activate pytorch
conda activate pytorch
# export MODULEPATH=/opt/slurm/etc/files/modulesfiles/:$MODULEPATH 

module load cuda/11.6 \
    nccl/2.12.7-cuda.11.6 \
    nccl_efa/1.15.1-nccl.2.12.7-cuda.11.6

export CUDA_HOME=/opt/slurm/etc/files/modulesfiles/cuda/11.6
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# --> Dataset creation
srun --cpu_bind=socket bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; 
python make_dataset.py -f /home/manasakani/water_molecules_small_flexible -o water_clusters_small_flexible_x800.db -m 100'
# m = number of structures to make per gpu (omit for all)

# --> Training script
# srun --cpu_bind=socket bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; 
# python train_QM7_water_clusters.py'