#!/bin/bash -l
#SBATCH --job-name=amorphous_gnns_test
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --account=lp16
#SBATCH --hint=nomultithread
#SBATCH --hint=exclusive

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MPICH_MALLOC_FALLBACK=1

MASTER_NODE=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_ADDR=$MASTER_NODE
export MASTER_PORT=29500
# echo "Master node address: $MASTER_ADDR"

# debug mode
# TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTHONUNBUFFERED=1

srun python main-HfO2.py -f $SCRATCH
