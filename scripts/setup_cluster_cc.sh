#!/bin/bash

# Get project name from args
PROJECT_NAME=$1

# Assert project name is not empty
if [ -z "$PROJECT_NAME" ]; then
  echo "Project name is empty"
  exit 1
fi

VENV_PATH=~/venv_pt_hf_base
REPO_DIR=$(pwd)

module load python/3.9
if [ ! -d "$VENV_PATH" ]; then
  python3 -m venv $VENV_PATH
fi
source $VENV_PATH/bin/activate
pip install --upgrade pip
pip install pika wandb PyGithub InquirerPy

mkdir -p ~/scratch/containers
cd ~/scratch/containers/
module load singularity
if [ ! -f "pt_v7.sif" ]; then
  singularity pull --arch amd64 library://kzmnjd/deeplr/pt:v7
fi

module load gcc arrow scipy-stack
source $VENV_PATH/bin/activate
pip install torch torchvision transformers==4.16.2 datasets sklearn sentencepiece seqeval
mkdir -p ~/scratch/$PROJECT_NAME/experiments/hf_cache
mkdir -p ~/scratch/$PROJECT_NAME/experiments/hf_ds_cache
mkdir -p ~/scratch/$PROJECT_NAME/experiments/hf_module_cache
mkdir -p ~/scratch/$PROJECT_NAME/experiments/wandb_cache_dir
export TRANSFORMERS_CACHE=~/scratch/$PROJECT_NAME/experiments/hf_cache
export HF_DATASETS_CACHE=~/scratch/$PROJECT_NAME/experiments/hf_ds_cache
export HF_MODULES_CACHE=~/scratch/$PROJECT_NAME/experiments/hf_module_cache

cd $REPO_DIR
python scripts/preload_hf_models.py

echo "alias launcher=\"source $VENV_PATH/bin/activate && python $REPO_DIR/scripts/launcher.py\"" >> ~/.bashrc