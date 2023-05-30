#!/bin/bash

# Get project name from args
PROJECT_NAME=$1

# Assert project name is not empty
if [ -z "$PROJECT_NAME" ]; then
  echo "Project name is empty"
  exit 1
fi

# ------------------------------------------------------------------
# Edit this to point to a shared directory between
# the compute nodes and the login ınode
CLUSTER_SHARED_STORAGE_DIR="/dccstor/$USER/${PROJECT_NAME}_scratch"
# ------------------------------------------------------------------

REPO_DIR=$(pwd)
CONDA_ENV_NAME=pt_v7

# Check if conda env exists
if ! conda env list | grep -q $CONDA_ENV_NAME; then
  # Create conda env based on environment.yml
  conda env create -f environment.yml -n $CONDA_ENV_NAME
fi

# Activate conda env
conda activate $CONDA_ENV_NAME

mkdir -p $CLUSTER_SHARED_STORAGE_DIR
ln -snf $CLUSTER_SHARED_STORAGE_DIR ~/scratch

mkdir -p ~/scratch/logs
mkdir -p ~/scratch/scripts

mkdir -p ~/scratch/$PROJECT_NAME/experiments/hf_cache
mkdir -p ~/scratch/$PROJECT_NAME/experiments/hf_ds_cache
mkdir -p ~/scratch/$PROJECT_NAME/experiments/hf_module_cache
mkdir -p ~/scratch/$PROJECT_NAME/experiments/wandb_cache_dir

echo "alias launcher=\"conda activate $CONDA_ENV_NAME && python $REPO_DIR/scripts/launcher.py\"" >>~/.bashrc
