#!/bin/bash

# If using SLURM env
if [ -z "$SLURM_JOB_ID" ]; then
  # Not running within a SLURM job
  # First check if we can use CUDA_VISIBLE_DEVICES
  if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    # No CUDA_VISIBLE_DEVICES set, use nvidia-smi
    export NUM_GPUS=$(nvidia-smi -L | wc -l)
  else
    # CUDA_VISIBLE_DEVICES is set, use that
    export NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
  fi
else
  # Running within a SLURM job
  if [ -z "$SLURM_GPUS_ON_NODE" ]; then
    export NUM_GPUS=$(nvidia-smi -L | wc -l)
  else
    export NUM_GPUS=$SLURM_GPUS_ON_NODE
  fi
fi