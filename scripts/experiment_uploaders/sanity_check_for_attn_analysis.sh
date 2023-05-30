#!/bin/bash

python scripts/experiment_uploaders/sanity_check.py \
  --dataset_name "s2s_reverse" \
  --ds_split "mc2x_tr20_ts40" \
  --tags 'sanityCh_for_attn'

python scripts/experiment_uploaders/sanity_check.py \
  --dataset_name "s2s_reverse" \
  --ds_split "mc_tr20_ts40" \
  --tags 'sanityCh_for_attn'