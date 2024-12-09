#!/bin/bash
source /home/almaeder/miniconda3/etc/profile.d/conda.sh  # Ensures conda is initialized on all hosts
conda activate /home/almaeder/miniconda3/envs/ml2
python3 "$@"


# how to use: mpirun -np 2 -hostfile ./hosts ./run.sh train.py -f ../../..