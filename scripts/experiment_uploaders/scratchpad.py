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
from transformers import set_seed

POSITIONAL_ENCODINGS = [
    "pe_t5",
    "pe_none",
    "pe_abs_sin",
    "pe_alibi",
    "pe_rotary",
    # "pe_rotary_rerun",
    # "pe_newRot",
]


# List of (DS, DS_SPLIT) tuples
DS_WITH_SCRATCHPAD_SUPPORT = [
    ["s2s_addition", "len_tr8_ts16"],
    ["s2s_poly", "n_terms_tr8_ts16"],
    ["s2s_sort", "len_sngd_tr8_ts16"],
    ["s2s_sort", "len_mltd_tr8_ts16"],
    ["s2s_parity", "len_tr8_ts16"],
    ["s2s_lego", "len_tr8_ts16"],
    ["s2s_lego", "len_tr8_ts16_perm"],
    ["s2s_sum", "len_tr8_ts16"],
    ["clutrr", "bwd_short"],
]


DS_TO_DS_SPLITS = defaultdict(list)
for ds, ds_split in DS_WITH_SCRATCHPAD_SUPPORT:
    DS_TO_DS_SPLITS[ds].append(ds_split)

# Sort DS_SPLITS
for ds in DS_TO_DS_SPLITS:
    DS_TO_DS_SPLITS[ds] = sorted(DS_TO_DS_SPLITS[ds])


def generate_boolean_configs(config):
    configs = []
    for i in range(len(config)):
        new_config = config.copy()
        new_config[i] = not config[i]
        configs.append(new_config)
    return configs


def modify_array(arr: List[Any], idx: int, val: Any) -> List[Any]:
    arr[idx] = val
    return arr


def generate_all_scratchpad_configs(ds_name, split_name) -> List[Dict[str, Any]]:
    configs = []
    for include_scratchpad in [
        True,
        False,
    ]:
        if not include_scratchpad:
            configs.append({"include_scratchpad": include_scratchpad})
            continue

        include_all = [
            True,  # include_input
            True,  # include_computation
            True,  # include_output
            True,  # include_intermediate_variables
            True,  # include_remaining_input
        ]
        include_only_computation = [
            False,  # include_input
            True,  # include_computation
            True,  # include_output
            False,  # include_intermediate_variables
            False,  # include_remaining_input
        ]
        all_bool_configs = [
            include_all.copy(),
            include_only_computation.copy(),
        ] + generate_boolean_configs(include_all)

        if ds_name in ["s2s_sort", "s2s_lego"]:
            # Don't include intermediate variables for sort
            all_bool_configs = [modify_array(c, 3, False) for c in all_bool_configs]

        if ds_name in ["clutrr"]:
            # Only include computation for clutrr
            all_bool_configs = [
                [
                    False,  # include_input
                    True,  # include_computation
                    False,  # include_output
                    False,  # include_intermediate_variables
                    False,  # include_remaining_input
                ]
            ]

        # Make the configs unique by converting each to tuple
        # and then the list of configs to a set
        all_bool_configs = [tuple(c) for c in all_bool_configs]
        all_bool_configs = sorted(set(all_bool_configs))

        for config in all_bool_configs:
            configs.append(
                {
                    "include_scratchpad": include_scratchpad,
                    "include_input": config[0],
                    "include_computation": config[1],
                    "include_output": config[2],
                    "include_intermediate_variables": config[3],
                    "include_remaining_input": config[4],
                }
            )

    return configs


def get_file_name(config: Dict[str, bool]) -> str:
    keys_to_abrv = {
        "include_input": "i",
        "include_computation": "c",
        "include_output": "o",
        "include_intermediate_variables": "v",
        "include_remaining_input": "r",
    }
    keys_in_order = [
        "include_input",
        "include_computation",
        "include_output",
        "include_intermediate_variables",
        "include_remaining_input",
    ]
    filename = ""
    for key in keys_in_order:
        abrv = keys_to_abrv[key]
        if key not in config:
            print(config)
            raise ValueError(f"Key {key} not in config")
        val = str(int(config[key]))
        filename += f"{abrv}{val}_"

    return filename[:-1]


def main(
    dataset_name: str = None,
    ds_split: str = None,
    pe: str = None,
    tgt_scratchpad_cfg: str = None,
    base_config: str = None,
    sweep_config: str = None,
    launcher: str = None,
    dry_run: bool = False,
    force: bool = False,
    tags: str = None,
):
    if base_config is None:
        base_config = "configs/t5_dec_base.jsonnet"

    if sweep_config is None:
        sweep_config = "configs/sweeps/no_sweep.jsonnet"

    if launcher is None:
        launcher = "upload_experiment_with_manual_hp_post_script.sh"

    exp_ids: Dict[str, Tuple[str, str, str]] = {}

    if pe is not None:
        pos_encoding_list = pe.split(",")
    else:
        pos_encoding_list = POSITIONAL_ENCODINGS

    if dataset_name is not None:
        ds_list = dataset_name.split(",")
        ds_list = [(ds, split) for ds in ds_list for split in DS_TO_DS_SPLITS[ds]]
        if ds_split is not None:
            ds_list = [(ds, split) for ds, split in ds_list if split == ds_split]
    else:
        ds_list = DS_WITH_SCRATCHPAD_SUPPORT

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
            for scratchpad_config in generate_all_scratchpad_configs(ds, ds_split):
                if scratchpad_config["include_scratchpad"]:
                    scratchpad_config_str = ",configs/data/w_scratchpad_f.jsonnet"
                    scratchpad_config_filename = get_file_name(scratchpad_config)
                    scratchpad_config_str += f",configs/data/unified_scratchpad_configs/ufs__{scratchpad_config_filename}.jsonnet"
                else:
                    scratchpad_config_str = ""
                    scratchpad_config_filename = "no_scratchpad"

                if (
                    tgt_scratchpad_cfg is not None
                    and scratchpad_config_filename != tgt_scratchpad_cfg
                ):
                    continue

                # Get the length approx of the dataset
                length_approx_key = f"{ds},{ds_split}"
                length_approx = max_length_approx[length_approx_key][
                    scratchpad_config_filename
                ]

                if tags is None:
                    tags = f"scratchpad,scratchpad_f,scratchpad_{ds}"


                cmd = f"scripts/{launcher}"
                cmd += f" --dataset {ds}"
                cmd += f" --split {ds_split}"
                cmd += f' --configs "{base_config},configs/models/{pe}.jsonnet{scratchpad_config_str}"'
                cmd += f" --sweep_configs {sweep_config}"
                cmd += f' --commands "hp_step --eval_split valid"'
                cmd += f' --env "APP_SEED=42"'
                cmd += f" --tags {tags}"
                if not force:
                    cmd += f" --post_script scripts/manual_sweep_launch_best_run_all.sh"
                else:
                    cmd += f" --post_script scripts/manual_sweep_launch_best_run_all_force.sh"

                if not dry_run:
                    output = os.popen(cmd).read()
                else:
                    random_exp_id = str(uuid.uuid4())[0:8]
                    output = f"Exp Key: {random_exp_id}\n"

                # Get the experiment id from the output using a regex
                try:
                    exp_id = re.search(r"Exp Key: (.*)\n", output).group(1)
                    exp_ids[exp_id] = (ds_split, pe, scratchpad_config_filename)
                except Exception as e:
                    print(f"Failed to get exp_id from output: {output}")
                    raise e

                print(f"Experiment id: {exp_id}")

                potential_gpu_class = get_gpu_class(length_approx)
                gpu_class_to_experiment_ids[potential_gpu_class].append(exp_id)

                experiment_id_to_info[exp_id] = {
                    "length_approx": length_approx,
                    "gpu_class": potential_gpu_class,
                    "scratchpad_config": scratchpad_config_filename,
                    "ds": ds,
                    "ds_split": ds_split,
                    "pe": pe,
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
            "mila": 0.3,
            "cc_narval": 0.7,
        },
        "fast_high_mem": {
            "mila": 1,
        },
    }

    set_seed(42)

    provider_to_gpu_class_to_exp_ids = defaultdict(lambda: defaultdict(list))
    # Now we have all the experiment ids, we can assign them to the right gpu provider
    for gpu_class, experiment_ids in gpu_class_to_experiment_ids.items():
        providers = GPU_CLASS_TO_PROVIDERS[gpu_class]
        provider_names = sorted(providers.keys())
        provider_weights = [providers[p] for p in provider_names]
        for experiment_id in experiment_ids:
            provider = np.random.choice(provider_names, p=provider_weights)
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
                assert gpu_class == "slow_low_mem"
                cmd = (
                    f"export WANDB_PROJECT=len_gen && "
                    f"launcher -p cc  --image_name pt_v7.sif "
                    f'-s "--gres=gpu:v100l:1 -t 72:00:00 -c 6 --mem=64G" '
                    f"--nodup {exp_ids_str}"
                )
            elif provider == "cc_narval":
                assert gpu_class == "slow_med_mem" or gpu_class == "fast_med_mem"
                cmd = (
                    f"export WANDB_PROJECT=len_gen && "
                    f"launcher -p cc  --image_name pt_v7.sif "
                    f'-s "--gres=gpu:1 -t 72:00:00 -c 6 --mem=64G" '
                    f"--nodup {exp_ids_str}"
                )
            elif provider == "mila":
                assert (
                    gpu_class == "slow_med_mem"
                    or gpu_class == "fast_med_mem"
                    or gpu_class == "fast_high_mem"
                )
                if gpu_class == "slow_med_mem":
                    gpu_model = "rtx8000"
                elif gpu_class == "fast_med_mem":
                    gpu_model = "a100"
                elif gpu_class == "fast_high_mem":
                    gpu_model = "a100l"
                else:
                    raise ValueError(f"Unknown gpu class {gpu_class}")

                cmd = (
                    f"export WANDB_PROJECT=len_gen && "
                    f"launcher -p mila  --image_name pt_v7.sif "
                    f'-s "--partition=long '
                    f"--gres=gpu:{gpu_model}:1 -t 72:00:00 -c 6 --mem=64G "
                    f"-x 'cn-g[005-012,017-026]'"
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
    with open(save_dir / f"scratchpad_exp_results_{dt_string}.json", "w") as f:
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
