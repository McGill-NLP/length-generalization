#!/bin/bash

source scripts/set_num_gpus.sh

# Check if $NUM_GPUS are higher than one and if so, use distributed launch
if [ "$NUM_GPUS" -gt 1 ]; then
  chmod +x scripts/distributed_manual_sweep_launch_best_run.sh
  ./scripts/distributed_manual_sweep_launch_best_run.sh
else
  chmod +x scripts/manual_sweep_launch_best_run.sh
  ./scripts/manual_sweep_launch_best_run.sh
fi