#!/bin/bash

#SBATCH --job-name=uracil
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:8
#SBATCH --time=08:00:00
#SBATCH --error=error_file.err
#SBATCH --output=out-train_water_uracil_5MP.out
#   SBATCH --qos=ocp_high
#   SBATCH --qos=alignment_shared
#   SBATCH --partition=ocp

source activate pytorch

module load cuda/11.6 \
    nccl/2.12.7-cuda.11.6 \
    nccl_efa/1.15.1-nccl.2.12.7-cuda.11.6

export CUDA_HOME=/opt/slurm/etc/files/modulesfiles/cuda/11.6

export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# --> Dataset creation
#srun --cpu_bind=socket bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; 
#python make_dataset.py -f /home/manasakani/ocp-modeling-dev/manasakani/fock_datasets/omol/omol_closedshell_25k_train -o omol_closedshell_25k_train_5.0.db -m 100000'
# m = number of structures to make per gpu (omit for all)

# --> Training script
# srun --cpu_bind=socket bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; 
# python run_nablaDFT_medium.py'

# --> Evals 
srun --cpu_bind=socket bash -c 'export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID; 
python run_QM7_uracil.py'
