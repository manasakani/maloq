#!/bin/bash -l
#SBATCH --job-name=test
#SBATCH --account=lp16
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=3
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=12
#SBATCH --partition=debug
#SBATCH --hint=nomultithread
#   SBATCH --hint=exclusive
#SBATCH --output=outputs_slurm/output_%j.txt
#SBATCH --error=error_file.err 

# srun --account=lp82 --gres=gpu:1 --time=00:30:00 --partition=debug --pty bash

# START ENVIRONMENT SETUP
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

export NCCL_ROOT=/user-environment/linux-sles15-neoverse_v2/gcc-13.3.0/nccl-2.22.3-1-4j6h3ffzysukqpqbvriorrzk2lm762dd  # Replace with your NCCL installation path
export NCCL_LIB_DIR=$NCCL_ROOT/lib
export NCCL_INCLUDE_DIR=$NCCL_ROOT/include

export CUDA_DIR=$CUDA_HOME
export CUDA_PATH=$CUDA_HOME
export CPATH=$CUDA_HOME/include:$CPATH
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CPATH=$NCCL_ROOT/include:$CPATH
export LIBRARY_PATH=$NCCL_ROOT/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=$NCCL_ROOT/lib:$LD_LIBRARY_PATH

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

source /users/mkanisel/miniconda3/bin/activate helm_env
# END ENVIRONMENT SETUP

# Profile: nsys profile -o my_profile_report --trace=cuda,nvtx python run_nablaDFT.py

# --> Dataset creation
# python make_omol_database_densitymat.py -f /capstor/scratch/cscs/mkanisel/omol_electrolytes_unsolvated -o omol_electrolytes_10k.db --start_idx 0 --end_idx 10000 --job_id 0

# --> Training/Eval
# srun --cpu-bind=socket bash -c 'export MPICH_GPU_SUPPORT_ENABLED=1; export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID;
# python run_omol_elytes_unsolvated.py'

# srun --cpu-bind=socket bash -c 'export MPICH_GPU_SUPPORT_ENABLED=1; export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID;
# python run_omol_elytes_redox.py'

# srun --cpu-bind=socket bash -c 'export MPICH_GPU_SUPPORT_ENABLED=1; export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID;
# python run_nablaDFT.py'

srun --cpu-bind=socket bash -c 'export MPICH_GPU_SUPPORT_ENABLED=1; export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID;
python run_QM7.py'