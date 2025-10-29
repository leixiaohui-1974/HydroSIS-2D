#!/bin/bash

# Single GPU dam break simulation
echo "Running 1D dam break test (Single GPU)..."
./hydrosis --nx 1024 --ny 512 --cfl 0.5 --tend 5.0 --test 0

echo "Done!"
