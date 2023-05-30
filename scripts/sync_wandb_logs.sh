#!/bin/bash

# Read the wandb src dir from the command line
SRC_WANDB_DIR=$1

# Read the wandb target dir to sync from the command line
TGT_WANDB_DIR=$2


# Sync the wandb dir
while true; do
  cp -r $SRC_WANDB_DIR/* $TGT_WANDB_DIR
  sleep 300
done

