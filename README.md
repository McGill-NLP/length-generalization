# The Impact of Positional Encoding on Length Generalization in Transformers

**Table of Content**

- [Paper](#paper)
- [Abstract](#abstract)
- [Quick Start](#quick-start)
  - [Install the required packages](#install-the-required-packages)
  - [Download the data](#download-the-data)
  - [Create the experiment script](#create-the-experiment-script)
  - [Run the experiment](#run-the-experiment)
- [1B Scale Pretrained Models](#1b-scale-pretrained-models)
  - [Dataset](#dataset)
  - [Model and Training](#model-and-training)
  - [Example](#example)
  - [Important Note](#important-note)
- [Code Structure](#code-structure)
- [Acknowledgement](#acknowledgement)
- [Citation](#citation)
  
## Paper

https://arxiv.org/abs/2305.19466

## Abstract

*Length generalization, the ability to generalize from small training context sizes to larger ones, is a critical challenge in the development of Transformer-based language models. Positional encoding (PE) has been identified as a major factor influencing length generalization, but the exact impact of different PE schemes on extrapolation in downstream tasks remains unclear. In this paper, we conduct a systematic empirical study comparing the length generalization performance of decoder-only Transformers with five different position encoding approaches including Absolute Position Embedding (APE), T5's Relative PE, ALiBi, and Rotary, in addition to Transformers without positional encoding (NoPE). Our evaluation encompasses a battery of reasoning and mathematical tasks. Our findings reveal that the most commonly used positional encoding methods, such as ALiBi, Rotary, and APE, are not well suited for length generalization in downstream tasks. More importantly, NoPE outperforms other explicit positional encoding methods while requiring no additional computation. We theoretically demonstrate that NoPE can represent both absolute and relative PEs, but when trained with SGD, it mostly resembles T5's relative PE attention patterns. Finally, we find that scratchpad is not always helpful to solve length generalization and its format highly impacts the model's performance. Overall, our work suggests that explicit position embeddings are not essential for decoder-only Transformers to generalize well to longer sequences.*

## Updates
- (Feb 18, 2024) Added the pretained models ([1B Scale Pretrained Models](#1b-scale-pretrained-models))
- (Dec 13, 2023) Presented as poster.
- (Sept 22, 2023) Paper got accepted at NeurIPS 2023.


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


## 1B Scale Pretrained Models

Following our submission, we pretrained a 1B-scale decoder-only style CodeLLM on 30B tokens, experimenting with three different positional encodings: **NoPE**, **Rotary**, and **ALiBi**. 
These models were pretrained using the exact same configuration to enable a fair comparison across the different positional encoding techniques (see Appendix F of paper for more details).

Find our pretrained 1B LLMs on 🤗 Huggingface:
- [`McGill-NLP/codellm_1b_nope`](https://huggingface.co/McGill-NLP/codellm_1b_nope)
- [`McGill-NLP/codellm_1b_rotary`](https://huggingface.co/McGill-NLP/codellm_1b_rotary)
- [`McGill-NLP/codellm_1b_alibi`](https://huggingface.co/McGill-NLP/codellm_1b_alibi)

### Dataset
We compiled a dataset by collecting 30M source code files from the StarCoder corpus ([Li et al., 2023](https://arxiv.org/abs/2305.06161)), totaling 30B tokens. The dataset composition is as follows:

- 40% Python
- 25% Java
- 25% JavaScript
- 5% GitHub issues
- 5% GitHub commits

### Model and Training

The configuration used is as follows:
- Decoder-only architecture, trained using next-token prediction.
- 1.3 billion parameters.
- Context size of 1024 tokens.
- Batch size of 256.
- `d_model` = 1024, `d_kv` = 128, `d_ff` = 16384, with 32 attention heads.
- Training duration was set to one epoch.
- For detailed hyperparameters parameters, refer to [Allah et al., 2023](https://arxiv.org/abs/2301.03988).

### Example

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Available models: `McGill-NLP/codellm_1b_nope`, `McGill-NLP/codellm_1b_rotary`, `McGill-NLP/codellm_1b_alibi`
model_name = "McGill-NLP/codellm_1b_rotary"

# Important: `trust_remote_code=True` is required due to the custom architecture supporting different positional encodings, necessitating the download of the model implementation from Huggingface
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(model.config.position_encoding_type)
# Outputs: `rotary`

prompt = "def print_hello_world():"
input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
input_ids = torch.cat([torch.tensor([[tokenizer.bos_token_id]], device="cuda"), input_ids], dim=1)  # Prepend <bos> token

output = model.generate(input_ids, do_sample=True, temperature=0.2, max_length=16)
print(tokenizer.decode(output[0]))
```
### Important Note
Please note that these models significantly differ from the small model used to produce the main results reported in our paper, particularly in terms of size and training context size. As such, they are not directly suitable for evaluation on the datasets described in the paper. To effectively utilize these models, one should consider recreating or adapting the datasets accordingly.

## Code Structure

Here's a brief overview of the key components of our codebase:

- [`configs`](https://github.com/McGill-NLP/length-generalization/tree/main/configs): Contains Jsonnet files for configuring experiment settings.
- [`notebooks`](https://github.com/McGill-NLP/length-generalization/tree/main/notebooks): A collection of ad-hoc Jupyter notebooks, primarily for visualization and plotting.
- [`src`](https://github.com/McGill-NLP/length-generalization/tree/main/src): The main directory for source code, encompassing:
    - [`models`](https://github.com/McGill-NLP/length-generalization/tree/main/src/models): Houses model implementations, with [`custom_t5_decoder_only.py`](https://github.com/McGill-NLP/length-generalization/blob/main/src/models/custom_t5_decoder_only.py) being the central piece for our custom models.
    - [`data`](https://github.com/McGill-NLP/length-generalization/tree/main/src/data): Manages the data pipeline. The [`s2s_dl_factory.py`](https://github.com/McGill-NLP/length-generalization/blob/main/src/data/s2s_dl_factory.py) script is key for tokenization and preparing data for sequence-to-sequence tasks.
    - [`trainers`](https://github.com/McGill-NLP/length-generalization/tree/main/trainers): Contains trainer classes, with [`decoder_only_trainer.py`](https://github.com/McGill-NLP/length-generalization/blob/main/src/trainers/decoder_only_trainer.py) being a custom trainer based on the 🤗 Trainer.
    - [`runtime`](https://github.com/McGill-NLP/length-generalization/tree/main/runtime): Integrates components and implements training and evaluation procedures. The [`seq2seq_runtime.py`](https://github.com/McGill-NLP/length-generalization/blob/main/src/runtime/seq2seq_runtime.py) script is specifically for sequence-to-sequence tasks.



## Acknowledgement
This repository is based on https://github.com/kazemnejad/pt_hf_base

## Citation
```bibtex
@inproceedings{kazemnejad2023:ImpactOfPeOnLengthGen,
      title={The Impact of Positional Encoding on Length Generalization in Transformers},
      author={Amirhossein Kazemnejad and Inkit Padhi and Karthikeyan Natesan and Payel Das and Siva Reddy},
      booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
      year={2023},
      url={https://openreview.net/forum?id=Drrl2gcjzl}
}
```
