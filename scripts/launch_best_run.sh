#!/bin/bash

python scripts/create_config_for_best_run.py

sweep_id=$(python -c "import json;print(json.load(open('best_run_info.json'))['sweep_id'], end='')")
export WANDB_TAGS=sweep_"$sweep_id"_best_run,$WANDB_TAGS
export EXP_WANDB_TAGS=$WANDB_TAGS

CONFIGS_STR="best_run.json,configs/hp_base.jsonnet,configs/final.jsonnet"

SEEDS="256788 234054 146317"

for SEED in $SEEDS; do
  export APP_DIRECTORY=experiments/$WANDB_RUN_GROUP/training_runs
  export APP_EXPERIMENT_NAME=seed_$SEED
  export APP_SEED=$SEED
  export WANDB_JOB_TYPE=best_run_seed_exp
  export WANDB_RUN_ID=sweep_"$sweep_id"_training_run__seed_$SEED

  python src/main.py --configs $CONFIGS_STR \
    train --eval_split valid

  python src/main.py --configs $CONFIGS_STR \
    predict

  python src/main.py --configs $CONFIGS_STR \
    combine_pred

  python src/main.py --configs $CONFIGS_STR \
    analyze_all

  python src/main.py --configs $CONFIGS_STR \
    predict --split valid

  python src/main.py --configs $CONFIGS_STR \
    combine_pred --split valid

  python src/main.py --configs $CONFIGS_STR \
    analyze_all --split valid

done
