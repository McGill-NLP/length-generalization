#!/bin/bash

# Parse command line name arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --dataset)
      DS="$2"
      shift # past argument
      shift # past value
      ;;
    --split)
      SPLIT="$2"
      shift # past argument
      shift # past value
      ;;
    --configs)
      CONFIGS="$2"
      shift # past argument
      shift # past value
      ;;
    --sweep_configs)
      SWEEP_CONFIGS="$2"
      shift # past argument
      shift # past value
      ;;
    --commands)
      COMMANDS="$2"
      shift # past argument
      shift # past value
      ;;
    --env)
      ENV_STR="$2"
      shift # past argument
      shift # past value
      ;;
    *)    # unknown option
      echo "Unknown option $1"
      exit 1
      ;;
  esac
done

if [ -z "$COMMANDS" ]; then
  COMMANDS="hp_step --eval_split valid"
fi

if [ -z "$ENV_STR" ]; then
  ENV_STR=""
fi

PROJECT_NAME=$(python -c "import json;print(json.load(open('configs/project_name.json'))['project_name'], end='')")
ENTITY_NAME=$(python -c "import json;print(json.load(open('configs/entity_name.json'))['entity_name'], end='')")
ENV_STR="$ENV_STR",APP_DS_SPLIT="$SPLIT"

echo "------------------"
echo "Dataset: $DS"
echo "Split: $SPLIT"
echo "Configs: $CONFIGS"
echo "Sweep configs: $SWEEP_CONFIGS"
echo "Commands: $COMMANDS"
echo "Project name: $PROJECT_NAME"
echo "Entity name: $ENTITY_NAME"
echo "Env str: $ENV_STR"
echo "------------------"

FINAL_CONFIGS_STR="$CONFIGS,configs/data/$DS.jsonnet,configs/sweep.jsonnet"
DS_ID=data-"$DS"-"$SPLIT"


BASE_EXP_NAME=$(python scripts/upload_experiment.py --configs "$FINAL_CONFIGS_STR" -c "$COMMANDS" -d $DS_ID --output_only_name)
RANDOM_UUID=$(python -c "import uuid;print(uuid.uuid4().hex[:10], end='')")
GROUP_NAME="SW"-"$BASE_EXP_NAME"-"$RANDOM_UUID"

echo "Group name: $GROUP_NAME"
echo "------------------"

python scripts/launch_sweep.py --name $GROUP_NAME "$SWEEP_CONFIGS" >>slog.txt 2>&1

SWEEP=$(python -c "import re;p=re.compile(r'wandb: Created sweep with ID: ((\d|\w)+)');print( re.search(p, open('slog.txt').read() ).group(1),end='')")
echo "Wandb Sweep URL: https://wandb.ai/$ENTITY_NAME/$PROJECT_NAME/sweeps/$SWEEP"

python scripts/upload_experiment.py \
  --configs $FINAL_CONFIGS_STR \
  -c "$COMMANDS" \
  -d $DS_ID \
  --env "$ENV_STR" \
  --sweep "$ENTITY_NAME/$PROJECT_NAME/$SWEEP" \
  --post_script ./scripts/launch_best_run_sum.sh  \
  --group $GROUP_NAME

rm slog.txt