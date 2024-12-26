#!/bin/bash
mpirun -np 4 ./run.sh train_dense_small.py -f ../../.. > output_4rank.txt
mpirun -np 3 ./run.sh train_dense_small.py -f ../../.. > output_3rank.txt
mpirun -np 2 ./run.sh train_dense_small.py -f ../../.. > output_2rank.txt
mpirun -np 1 ./run.sh train_dense_small.py -f ../../.. > output_1rank.txt