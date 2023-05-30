#!/bin/bash

# If using SLURM env
if [ -z "$SLURM_JOB_ID" ]; then
  # Not running within a SLURM job
  export MASTER_ADDR="0.0.0.0"

  # Set a random master port
  export MASTER_PORT=$(shuf -i 10000-65535 -n 1)
else
  # Running within a SLURM job
  export MASTER_PORT=$(expr 20000 + $(echo -n $SLURM_JOBID | tail -c 4))
  export MASTER_ADDR=$(hostname)

  if [ -z "$SLURM_NTASKS_PER_NODE" ]; then
    export WORLD_SIZE=$((${SLURM_JOB_NUM_NODES:=1} * $SLURM_GPUS_ON_NODE))
  else
    export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
  fi
fi