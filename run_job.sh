#!/bin/bash -l
#SBATCH --job-name=train_omol
#SBATCH --account=lp16
#SBATCH --time=05:30:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#   SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=64
#SBATCH --partition=normal
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --output=slurm_output/output_%j.txt
#SBATCH --error=slurm_output/error_file_%j.err 

# Interactive: srun --account=lp82 --gres=gpu:1 --time=00:30:00 --partition=debug --pty bash

# START ENVIRONMENT SETUP - CSCS Alps
uenv start --view=modules prgenv-gnu/24.11:v1 
# uenv start pytorch/v2.6.0:v1 --view=default

module load cuda
module load gcc
module load meson
module load ninja
module load nccl 
module load cray-mpich
module load cmake
module load openblas
module load aws-ofi-nccl

export NCCL_ROOT=/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd 
export NCCL_LIB_DIR=$NCCL_ROOT/lib
export NCCL_INCLUDE_DIR=$NCCL_ROOT/include
# export NCCL_DEBUG=INFO

export CUDA_DIR=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$NCCL_ROOT/include:$CPATH
export LIBRARY_PATH=$NCCL_ROOT/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

export NCCL_NET='AWS Libfabric'
# Disable eager messages to avoid NCCL timeouts
export FI_CXI_RDZV_GET_MIN=0
export FI_CXI_RDZV_THRESHOLD=0
export FI_CXI_RDZV_EAGER_SIZE=0

export TRITON_CACHE_DIR="/capstor/scratch/cscs/mkanisel/triton_cache"

source /users/mkanisel/miniconda3/bin/activate helm_env
# END ENVIRONMENT SETUP

# Profile: nsys profile -o my_profile_report --trace=cuda,nvtx python run_nablaDFT.py

# --> Dataset creation for omol electronic structures
# python make_omol_database_raw.py -f /capstor/scratch/cscs/mkanisel/omol_electrolytes_unsolvated -o omol_electrolytes_10k.db --start_idx 0 --end_idx 10000 --job_id 0

# --> Training/Eval
srun --cpu-bind=socket bash -c 'export MPICH_GPU_SUPPORT_ENABLED=1; export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID;
python run_omol_csh.py'