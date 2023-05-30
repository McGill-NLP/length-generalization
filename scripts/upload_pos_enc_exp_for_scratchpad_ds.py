import os
import re
from typing import Dict, Tuple, List, Any

import fire

POSITIONAL_ENCODINGS = ["pe_t5", "pe_none", "pe_abs_sin", "pe_alibi", "pe_rotary"]


datasets_with_scratchpad_support = [
    ["s2s_addition", "len_tr8_ts16"],
    ["s2s_poly", "n_terms_tr8_ts16"],
    ["s2s_sort", "len_mltd_tr8_ts16"],
    ["s2s_sort", "len_sngd_tr8_ts16"],
    ["s2s_lego", "len_tr8_ts16"],
    ["s2s_lego", "len_tr8_ts16_perm"],
    ["s2s_parity", "len_tr8_ts16"],
    ["sum", "len_tr8_ts16"],
]

DS_TO_SPLITS = {
    "s2s_parity": [
        # "len_tr20_ts40",
    ],
    "s2s_lego": [
        "len_tr8_ts16",
        "len_tr8_ts16_perm",
    ],
    "s2s_addition": [
        "len_tr8_ts16",
    ],
    "s2s_sort": [
        "len_mltd_tr8_ts16",
        "len_sngd_tr8_ts16",
    ],
    "s2s_poly": [
        "n_terms_tr8_ts16",
    ],
    "s2s_sum": [
        # "len_tr20_ts40",
    ],
}


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
            False,  # include_output
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
    dataset_name: str,
    pe: str = None,
    base_config: str = None,
    sweep_config: str = None,
    launcher: str = None,
):

    if base_config is None:
        base_config = "configs/t5_dec_base.jsonnet"

    if sweep_config is None:
        sweep_config = "configs/sweeps/training_scratch.jsonnet"

    if launcher is None:
        launcher = "upload_experiment_with_hp.sh"

    exp_ids: Dict[str, Tuple[str, str, str]] = {}

    if pe is not None:
        pos_encoding_list = pe.split(",")
    else:
        pos_encoding_list = POSITIONAL_ENCODINGS


    for pe in pos_encoding_list:
        for split in DS_TO_SPLITS[dataset_name]:
            for scratchpad_config in generate_all_scratchpad_configs(
                dataset_name, split
            )[:1]:
                if scratchpad_config["include_scratchpad"]:
                    scratchpad_config_str = ",configs/data/w_scratchpad_d.jsonnet"
                    scratchpad_config_filename = get_file_name(scratchpad_config)
                    # scratchpad_config_str += f",configs/data/unified_scratchpad_configs/ufs__{scratchpad_config_filename}.jsonnet"
                    scratchpad_config_str += f",configs/data/unified_scratchpad_configs/ufs__i1_c1_o1_v1_r1.jsonnet"
                else:
                    scratchpad_config_str = ""
                    scratchpad_config_filename = "no_scratchpad"
                    # continue

                cmd = f"scripts/{launcher}"
                cmd += f" --dataset {dataset_name}"
                cmd += f" --split {split}"
                cmd += f' --configs "{base_config},configs/models/{pe}.jsonnet{scratchpad_config_str}"'
                cmd += f" --sweep_configs {sweep_config}"
                cmd += f' --commands "hp_step --eval_split valid"'
                cmd += f' --env "APP_SEED=42"'
                cmd += f" --tags scratchpad,scratchpad_{dataset_name}"

                output = os.popen(cmd).read()

                # Get the experiment id from the output using a regex
                try:
                    exp_id = re.search(r"Exp Key: (.*)\n", output).group(1)
                    exp_ids[exp_id] = (split, pe, scratchpad_config_filename)
                except Exception as e:
                    print(f"Failed to get exp_id from output: {output}")
                    raise e

                print((split, pe, scratchpad_config_filename))

    # Print out all experiment ids
    for exp_id, (split, pe, scratchpad_format) in exp_ids.items():
        print(f"{exp_id}: {split} {pe} {scratchpad_format}")

    # Print out all experiment ids seperated by commas
    print("\nExperiment Keys")
    print(",".join(exp_ids.keys()))


if __name__ == "__main__":
    fire.Fire(main)
