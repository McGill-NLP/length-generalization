import datetime
import json
import os
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Dict, Tuple, List, Any

import fire
import numpy as np
import pandas as pd
from jsonlines import jsonlines
from transformers import set_seed

import os

current_path = os.path.dirname(os.path.abspath(__file__))
util_path = os.path.join(current_path, "util.py")
import importlib.util

spec = importlib.util.spec_from_file_location("util", util_path)
util = importlib.util.module_from_spec(spec)
spec.loader.exec_module(util)


def main(
    dataset_name: str = None,
    pe: str = None,
    base_config: str = None,
    sweep_config: str = None,
    launcher: str = None,
    dry_run: bool = False,
    force_re_upload_finished: bool = False,
):
    results_path = Path("results/scratchpad_f.jsonl")
    assert results_path.exists(), "Results file does not exist"

    results = pd.concat(
        [
            util.load_dataframe_from_jsonlines(results_path),
            util.load_dataframe_from_jsonlines(Path("results/classic.jsonl")),
            util.load_dataframe_from_jsonlines(Path("results/sanity_check.jsonl")),
        ]
    )
    results = results[results["job_type"] != "agent"]

    with open("scripts/attention_analysis_status.json", "r") as f:
        attention_analysis_status = json.load(f)

    run_group_to_cluster_name = util.get_run_group_to_cluster_name(results)

    currently_available_groups = util.read_currently_available_models()

    if base_config is None:
        base_config = "configs/t5_dec_base.jsonnet"

    if sweep_config is None:
        sweep_config = "configs/sweeps/no_sweep.jsonnet"

    if launcher is None:
        launcher = "upload_experiment_with_manual_hp_attn_analysis.sh"

    exp_ids: Dict[str, Tuple[str, str, str]] = {}

    if pe is not None:
        pos_encoding_list = pe.split(",")
    else:
        pos_encoding_list = util.POSITIONAL_ENCODINGS

    if dataset_name is not None:
        ds_list = dataset_name.split(",")
        ds_list = [
            (ds, ds_split) for ds in ds_list for ds_split in util.DS_TO_DS_SPLITS[ds]
        ]
    else:
        ds_list = util.DS_WITH_SCRATCHPAD_SUPPORT

    with open("scripts/scratchpad_stats.json", "r") as f:
        # Example:
        # {
        #     "ds,ds_plit": {
        #         ...
        #        "i1_c1_o1_v0_r1": 300,
        #        "i1_c1_o1_v1_r1": 300,
        #        ...
        #     }
        #   ...
        # }
        max_length_approx: Dict[str, Dict[str, int]] = json.load(f)

    GPU_CLASSES = [
        "slow_low_mem",  # Length range: 0-200,
        "slow_med_mem",  # Length range: 200-300
        "fast_med_mem",  # Length range: 300-400
        "fast_high_mem",  # Length range: > 400
    ]

    def get_gpu_class(length):
        if length <= 200:
            return "slow_low_mem"
        elif length <= 300:
            return "slow_med_mem"
        elif length <= 400:
            return "fast_med_mem"
        else:
            return "fast_high_mem"

    gpu_class_to_experiment_ids = defaultdict(list)
    experiment_id_to_info = {}

    for ds, ds_split in ds_list:
        print(f"Generating configs for {ds} {ds_split}...")
        for pe in pos_encoding_list:
            for scratchpad_config in util.generate_all_scratchpad_configs(ds, ds_split):
                if scratchpad_config["include_scratchpad"]:
                    scratchpad_config_str = ",configs/data/w_scratchpad_f.jsonnet"
                    scratchpad_config_filename = util.get_file_name(scratchpad_config)
                    scratchpad_config_str += f",configs/data/unified_scratchpad_configs/ufs__{scratchpad_config_filename}.jsonnet"
                else:
                    scratchpad_config_str = ""
                    scratchpad_config_filename = "no_scratchpad"

                if not force_re_upload_finished:
                    # key example: scan--mdlen_tr25_ts48$$$pe_rotary$$$no_scratchpad
                    attn_status_key = "$$$".join(
                        [f"{ds}--{ds_split}", pe, scratchpad_config_filename]
                    )

                    if attn_status_key in attention_analysis_status:
                        status_dict = attention_analysis_status[attn_status_key]
                        if status_dict is not None:
                            keys_to_check = [
                                "is_done__attn_analysis2",
                                "is_done__attn_analysis_aggr",
                            ]
                            if all([status_dict[k] == True for k in keys_to_check]):
                                print(
                                    f"Skipping {ds} {ds_split} {pe} {scratchpad_config_filename} because it is already done"
                                )
                                continue

                            keys_to_check = [
                                "is_running__attn_analysis2",
                                "is_running__attn_analysis_aggr",
                            ]
                            if any([status_dict[k] == True for k in keys_to_check]):
                                print(
                                    f"Skipping {ds} {ds_split} {pe} {scratchpad_config_filename} because it is already running"
                                )
                                continue

                # Get the length approx of the dataset
                length_approx_key = f"{ds},{ds_split}"
                length_approx = max_length_approx[length_approx_key][
                    scratchpad_config_filename
                ]

                cmd = f"scripts/{launcher}"
                cmd += f" --dataset {ds}"
                cmd += f" --split {ds_split}"
                cmd += f' --configs "{base_config},configs/models/{pe}.jsonnet{scratchpad_config_str}"'
                cmd += f" --sweep_configs {sweep_config}"
                cmd += f' --commands "hp_step --eval_split valid"'
                cmd += f' --env "APP_SEED=42"'
                cmd += (
                    f" --tags attention_analysis,attention_analysis_{ds},"
                    f"attention_aggr_analysis,attention_aggr_analysis_{ds}",
                )
                cmd += (
                    f" --post_script scripts/manual_sweep_launch_best_run_analyze_attn_all.sh",
                )

                if not dry_run:
                    output = os.popen(cmd).read()
                else:
                    random_exp_id = str(uuid.uuid4())[0:8]
                    output = f"Exp Key: {random_exp_id}\n"

                # Get the experiment id from the output using a regex
                try:
                    exp_id = re.search(r"Exp Key: (.*)\n", output).group(1)
                    exp_ids[exp_id] = (ds_split, pe, scratchpad_config_filename)

                    group_name = re.search(
                        r"Group name / Sweep Name: (.*)\n", output
                    ).group(1)
                except Exception as e:
                    print(f"Failed to get exp_id from output: {output}")
                    raise e

                print(f"Experiment id: {exp_id}")

                # Make sure the group name is available on the cluster
                if group_name not in run_group_to_cluster_name:
                    print(f"Skipping {exp_id} because {group_name} is not available")
                    continue
                cluster_name = run_group_to_cluster_name[group_name]
                groups_available_on_cluster = currently_available_groups[cluster_name]
                if group_name not in groups_available_on_cluster:
                    print(f"Skipping {exp_id} because {cluster_name} is not available")
                    continue

                potential_gpu_class = get_gpu_class(length_approx)
                gpu_class_to_experiment_ids[potential_gpu_class].append(exp_id)

                experiment_id_to_info[exp_id] = {
                    "length_approx": length_approx,
                    "gpu_class": potential_gpu_class,
                    "scratchpad_config": scratchpad_config_filename,
                    "ds": ds,
                    "ds_split": ds_split,
                    "pe": pe,
                    "group_name": group_name,
                }

    # class to provider and their weights
    GPU_CLASS_TO_PROVIDERS = {
        "slow_low_mem": {
            "cc_cedar": 1.0,
        },
        "slow_med_mem": {
            "mila": 0.6,
            "cc_narval": 0.4,
        },
        "fast_med_mem": {
            "mila": 0.2,
            "cc_narval": 0.3,
            "ibm": 0.5,
        },
        "fast_high_mem": {
            "mila": 0.2,
            "ibm": 0.8,
        },
    }

    set_seed(42)

    provider_to_gpu_class_to_exp_ids = defaultdict(lambda: defaultdict(list))
    # Now we have all the experiment ids, we can assign them to the right gpu provider
    for gpu_class, experiment_ids in gpu_class_to_experiment_ids.items():
        for experiment_id in experiment_ids:
            group_name = experiment_id_to_info[experiment_id]["group_name"]
            provider = run_group_to_cluster_name[group_name]
            experiment_id_to_info[experiment_id]["provider"] = provider
            provider_to_gpu_class_to_exp_ids[provider][gpu_class].append(experiment_id)

    # Now, restructure experiment_id_to_info to be "ds,ds_split" -> "pe" -> "scratchpad_config" -> {"id":..., "provider":..., "length_approx":...}
    ds_to_pe_to_scratchpad_config_to_info = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for exp_id, info in experiment_id_to_info.items():
        ds_key = f"{info['ds']},{info['ds_split']}"
        ds_to_pe_to_scratchpad_config_to_info[ds_key][info["pe"]][
            info["scratchpad_config"]
        ] = {
            "id": exp_id,
            "provider": info["provider"],
            "length_approx": info["length_approx"],
        }

    # Now, we can generate the commands to run the experiments for each provider based on the GPU class
    # "provider" -> "gpu_class" -> "cmd"
    provider_to_gpu_class_to_cmd = defaultdict(lambda: defaultdict(list))
    for provider, gpu_class_to_exp_ids in provider_to_gpu_class_to_exp_ids.items():
        for gpu_class, exp_ids in gpu_class_to_exp_ids.items():
            exp_ids_str = ",".join(exp_ids)
            if provider == "cc_cedar":
                cmd = (
                    f"export WANDB_PROJECT=len_gen && "
                    f"launcher -p cc  --image_name pt_v7.sif "
                    f'-s "--gres=gpu:v100l:1 -t 72:00:00 -c 6 --mem=64G" '
                    f"--nodup {exp_ids_str}"
                )
            elif provider == "cc_narval":
                cmd = (
                    f"export WANDB_PROJECT=len_gen && "
                    f"launcher -p cc  --image_name pt_v7.sif "
                    f'-s "--gres=gpu:1 -t 72:00:00 -c 6 --mem=64G" '
                    f"--nodup {exp_ids_str}"
                )
            elif provider == "mila":
                cmd = (
                    f"export WANDB_PROJECT=len_gen && "
                    f"launcher -p mila  --image_name pt_v7.sif "
                    f'-s "--partition=long '
                    f"--gres=gpu:a100l.3g.40gb:1 -t 72:00:00 -c 6 --mem=64G "
                    f'" '
                    f"--nodup {exp_ids_str}"
                )
            elif provider == "ibm":
                assert gpu_class == "fast_med_mem" or gpu_class == "fast_high_mem"
                if gpu_class == "fast_med_mem":
                    gpu_model = "a100"
                elif gpu_class == "fast_high_mem":
                    gpu_model = "a100l"
                else:
                    raise ValueError(f"Unknown gpu class {gpu_class}")

                cmd = (
                    f"launcher -p ibm -n 3 "
                    f'-s "-mem 30g  -cores 6+1 -q x86_24h -require {gpu_model}" '
                    f"{exp_ids_str}"
                )
            else:
                raise ValueError(f"Unknown provider {provider}")

            provider_to_gpu_class_to_cmd[provider][gpu_class].append(cmd)

    # Now, print the commands
    for provider, gpu_class_to_cmd in provider_to_gpu_class_to_cmd.items():
        print(f"----------------------------------------")
        print(f"Provider: {provider}")
        for gpu_class, cmds in gpu_class_to_cmd.items():
            print(f"\tGPU class: {gpu_class}")
            for cmd in cmds:
                print("\t", cmd)
            print()

    # Save the upload results to a file timestamped with the current time
    now = datetime.datetime.now()
    dt_string = now.strftime("%d-%m-%Y_%H-%M-%S")
    save_dir = Path("experiments")
    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / f"attn_results_{dt_string}.json", "w") as f:
        json.dump(
            {
                "exp_info": ds_to_pe_to_scratchpad_config_to_info,
                "commands": provider_to_gpu_class_to_cmd,
                "id_to_info": experiment_id_to_info,
            },
            f,
            indent=4,
            sort_keys=True,
        )


if __name__ == "__main__":
    fire.Fire(main)
