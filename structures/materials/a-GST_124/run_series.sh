#!/bin/bash
mpirun -np 4 ./run.sh train.py -f ../../.. > output_4rank.txt