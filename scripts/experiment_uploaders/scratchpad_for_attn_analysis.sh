#!/bin/bash


# Addition task with full scratchpad format
python scripts/experiment_uploaders/scratchpad.py \
  --dataset_name "s2s_addition" \
  --tgt_scratchpad_cfg "i1_c1_o1_v1_r1" \
  --tags 'scratch_for_attn'