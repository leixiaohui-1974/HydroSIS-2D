#!/bin/bash

# Multi-GPU circular dam break simulation
# Adjust -np based on number of GPUs available

NUM_GPUS=4

echo "Running circular dam break test (Multi-GPU with $NUM_GPUS GPUs)..."
mpirun -np $NUM_GPUS ./hydrosis --multi-gpu --nx 2048 --ny 2048 --cfl 0.5 --tend 10.0 --test 1

echo "Done!"
