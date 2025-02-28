#!/bin/bash -l
#SBATCH --job-name=gnn-malon
#SBATCH --account=lp16
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=18
#SBATCH --partition=debug
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive
#SBATCH --output=outputs/output_%j.txt
#SBATCH --error=error_file.err 

uenv start --view=modules prgenv-gnu/24.11:v1 

module load cuda
module load gcc
module load meson
module load ninja
module load nccl 
module load cray-mpich
module load cmake
module load openblas

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

source /users/amaeder/miniconda3/etc/profile.d/conda.sh
conda activate ml

srun --cpu-bind=socket bash -c 'export MPICH_GPU_SUPPORT_ENABLED=1; export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID;
python train.py -f /capstor/scratch/cscs/amaeder/ > outputs/output_${SLURM_PROCID}_${SLURM_NTASKS}.txt'