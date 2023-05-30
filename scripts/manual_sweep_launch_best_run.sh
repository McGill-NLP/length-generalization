#!/bin/bash

set -e

if [ ! -f best_run.json ]; then
  # If best_run.json does not exist, then it means there were no sweeps
  # So, we just run the base config
  echo "best_run.json does not exist. Running base config."
  # Make sure HP_EXP_CONFIG is set
  if [ -z "$HP_EXP_CONFIG" ]; then
    echo "HP_EXP_CONFIG is not set"
    exit 1
  fi
  CONFIGS_STR="${HP_EXP_CONFIG},configs/hp_base.jsonnet,configs/final.jsonnet"
else
  CONFIGS_STR="best_run.json,configs/hp_base.jsonnet,configs/final.jsonnet"

  python scripts/manual_sweep.py \
    --sweep_name $SWEEP_NAME \
    --sweep_root_dir $SWEEP_ROOT_DIR \
    --sweep_configs $SWEEP_CONFIGS \
    fail_if_sweep_not_complete
fi

RUN_ID_PREFIX=$(python scripts/manual_sweep.py \
  --sweep_name $SWEEP_NAME \
  --sweep_root_dir "$SWEEP_ROOT_DIR" \
  --sweep_configs $SWEEP_CONFIGS \
  generate_deterministic_run_id --run_name "best_run")

SEEDS="256788 234054 146317"

for SEED in $SEEDS; do
  export APP_DIRECTORY=$SWEEP_ROOT_DIR/exps/
  export APP_EXPERIMENT_NAME="best_run_seed_${SEED}"
  export APP_SEED=$SEED
  export WANDB_JOB_TYPE=best_run_seed_exp
  export WANDB_RUN_ID="${RUN_ID_PREFIX}__${SEED}"

  python src/main.py --configs $CONFIGS_STR \
    full_step

done
