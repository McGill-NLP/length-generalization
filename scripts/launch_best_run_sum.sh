#!/bin/bash

python scripts/create_config_for_best_run.py

sweep_id=$(python -c "import json;print(json.load(open('best_run_info.json'))['sweep_id'], end='')")
export WANDB_TAGS=sweep_"$sweep_id"_best_run
export EXP_WANDB_TAGS=$WANDB_TAGS


SEEDS="256788 234054 146317"

for SEED in $SEEDS; do
	export APP_DIRECTORY=experiments/$WANDB_RUN_GROUP/training_runs
	export APP_EXPERIMENT_NAME=seed_$SEED
	export APP_SEED=$SEED
	export WANDB_JOB_TYPE=best_run_seed_exp
	export WANDB_RUN_ID=sweep_"$sweep_id"_training_run__seed_$SEED

	python src/main.py --configs 'best_run.json,configs/hp_base.jsonnet,configs/final_sum.jsonnet' \
	       train --eval_split valid

  python src/main.py --configs 'best_run.json,configs/hp_base.jsonnet,configs/final_sum.jsonnet' \
	       predict

  python src/main.py --configs 'best_run.json,configs/hp_base.jsonnet,configs/final_sum.jsonnet' \
	       combine_pred

  python src/main.py --configs 'best_run.json,configs/hp_base.jsonnet,configs/final_sum.jsonnet' \
	       analyze_all

	python src/main.py --configs 'best_run.json,configs/hp_base.jsonnet,configs/final_sum.jsonnet' \
	       predict --split valid

	python src/main.py --configs 'best_run.json,configs/hp_base.jsonnet,configs/final_sum.jsonnet' \
	       combine_pred --split valid

  python src/main.py --configs 'best_run.json,configs/hp_base.jsonnet,configs/final_sum.jsonnet' \
	       analyze_all --split valid

done