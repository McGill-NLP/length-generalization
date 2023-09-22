# The Impact of Positional Encoding on Length Generalization in Transformers
Paper:

https://arxiv.org/abs/2305.19466

Abstract:

*Length generalization, the ability to generalize from small training context sizes to larger ones, is a critical challenge in the development of Transformer-based language models. Positional encoding (PE) has been identified as a major factor influencing length generalization, but the exact impact of different PE schemes on extrapolation in downstream tasks remains unclear. In this paper, we conduct a systematic empirical study comparing the length generalization performance of decoder-only Transformers with five different position encoding approaches including Absolute Position Embedding (APE), T5's Relative PE, ALiBi, and Rotary, in addition to Transformers without positional encoding (NoPE). Our evaluation encompasses a battery of reasoning and mathematical tasks. Our findings reveal that the most commonly used positional encoding methods, such as ALiBi, Rotary, and APE, are not well suited for length generalization in downstream tasks. More importantly, NoPE outperforms other explicit positional encoding methods while requiring no additional computation. We theoretically demonstrate that NoPE can represent both absolute and relative PEs, but when trained with SGD, it mostly resembles T5's relative PE attention patterns. Finally, we find that scratchpad is not always helpful to solve length generalization and its format highly impacts the model's performance. Overall, our work suggests that explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences.*

## Updates
- (Sept 22) Paper got accepted at NeurIPS 2023
## Quick Start
This section provides a quick start guide to use the codebase. 

### Install the required packages:
We provide two options to prepare the environment.
1. Using conda (install from `environment.yml`):
```bash
conda env create -f environment.yml
conda activate pt_v7
```
2. Using the singularity container:
```bash
singularity pull library://kzmnjd/deeplr/pt:v7
```

### Download the data
```bash
chmod a+x scripts/download_and_prepare_datasets.sh
./scripts/download_and_prepare_datasets.sh
```

### Create the experiment script
The following script provides a full training and evaluation scenario for training the model on a given dataset.

Use `run.sh.template` to create an experiment for different datasets and models. 
```bash
cp run.sh.template run.sh
```

Edit the `run.sh` file to set the required parameters. 
```bash
#!/bin/bash

set -e


#-------------------- EDIT THIS PART --------------------#
PE=pe_abs_sin # Select from pe_none, pe_t5, pe_alibi, pe_rotary, pe_abs_sin
DS=scan # See data/ for available datasets
export APP_DS_SPLIT=mdlen_tr25_ts48 # See data/$DS for available splits
export WANDB_ENTITY="<YOUR_WANDB_ENTITY>"
#-------------------- EDIT THIS PART --------------------#


export WANDB_RUN_GROUP="SW-t5_dec_base_${PE}_scan_sweep___data-${DS}-${APP_DS_SPLIT}"
export WANDB_TAGS="classic,classic_${DS}"
export WANDB_PROJECT="len_gen"

RUN_ID_PREFIX="run__${DS}__${PE}"

CONFIGS_STR="configs/t5_dec_base.jsonnet,\
configs/models/${PE}.jsonnet,\
configs/data/${DS}.jsonnet,\
configs/sweep.jsonnet,\
configs/hp_base.jsonnet,\
configs/final.jsonnet"

SEEDS="256788 234054 146317"

for SEED in $SEEDS; do
  export APP_DIRECTORY="experiments/${WANDB_RUN_GROUP}"
  export APP_EXPERIMENT_NAME="seed_${SEED}"
  export APP_SEED=$SEED

  export WANDB_JOB_TYPE=best_run_seed_exp
  export WANDB_RUN_ID="${RUN_ID_PREFIX}__${SEED}"
  # Training, Evaluation, and Analysis all in one command
  python src/main.py --configs $CONFIGS_STR \
    full_step

  export WANDB_JOB_TYPE=attn_analysis2
  export WANDB_RUN_ID="${RUN_ID_PREFIX}_2_${SEED}"
  export WANDB_TAGS=attention_analysis,$WANDB_TAGS
  python src/main.py --configs $CONFIGS_STR,configs/attn_analysis.jsonnet \
    analyze_all --split test

  export WANDB_JOB_TYPE=attn_analysis_aggr
  export WANDB_RUN_ID="${RUN_ID_PREFIX}_agg_${SEED}"
  export WANDB_TAGS=attention_aggr_analysis,$WANDB_TAGS
  python src/main.py --configs $CONFIGS_STR,configs/attn_aggr_analysis.jsonnet \
    analyze_all --split test
done
```

### Run the experiment:
1. Using conda:
```bash
mkdir -p experiments
conda activate pt_v7
chmod a+x run.sh
./run.sh
```

2. Using the singularity container:
```bash
mkdir -p experiments
chmod a+x run.sh
singularity exec --nv \
	-H $(pwd):$HOME \
	-B $(pwd)/experiments:$HOME/experiments \
	/path/to/singularity/image/pt_v7.sif \
	./run.sh
```

Note that this script will heavily make use of the [wandb](https://wandb.ai/) platform to log the results.


## Reproducibility
Work-in-progress. The instruction to fully replicate the results will be released in the coming weeks. 

## Acknowledgement
This repository is based on https://github.com/kazemnejad/pt_hf_base

## Citation
```bibtex
@misc{kazemnejad2023:ImpactOfPeOnLengthGen,
      title={The Impact of Positional Encoding on Length Generalization in Transformers}, 
      author={Amirhossein Kazemnejad and Inkit Padhi and Karthikeyan Natesan Ramamurthy and Payel Das and Siva Reddy},
      year={2023},
      eprint={2305.19466},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
