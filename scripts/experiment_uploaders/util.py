import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any

import jsonlines
import pandas as pd

POSITIONAL_ENCODINGS = [
    "pe_t5",
    "pe_none",
    "pe_abs_sin",
    "pe_alibi",
    "pe_rotary",
    # "pe_newRot",
]


# List of (DS, DS_SPLIT) tuples
DS_WITH_SCRATCHPAD_SUPPORT = [
    ["s2s_addition", "len_tr8_ts16"],
    ["s2s_poly", "n_terms_tr8_ts16"],
    # ["s2s_sort", "len_sngd_tr8_ts16"],
    ["s2s_parity", "len_tr8_ts16"],
    ["s2s_sum", "len_tr8_ts16"],
    ["s2s_lego", "len_tr8_ts16"],
    ["sum", "len_tr8_ts16"],
    ["s2s_sort", "len_mltd_tr8_ts16"],
    ["clutrr", "bwd_short"],
    # ["s2s_lego", "len_tr8_ts16_perm"],
    ("scan", "len_tr25_ts48"),
]

DS_TO_DS_SPLITS = defaultdict(list)
for ds, ds_split in DS_WITH_SCRATCHPAD_SUPPORT:
    DS_TO_DS_SPLITS[ds].append(ds_split)

# Sort DS_SPLITS
for ds in DS_TO_DS_SPLITS:
    DS_TO_DS_SPLITS[ds] = sorted(DS_TO_DS_SPLITS[ds])


def get_compute_cluster(host: str):
    if "cedar" in host:
        return "cc_cedar"
    elif "narval" in host:
        return "cc_narval"
    elif host.startswith("cn-"):
        return "mila"
    else:
        raise ValueError(f"Unknown host {host}")


def load_dataframe_from_jsonlines(path: Path) -> pd.DataFrame:
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    return pd.DataFrame.from_records(data)


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

        if ds_name in ["scan"]:
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


def read_currently_available_models():
    current_path = Path(__file__).parent
    models_path = current_path / "available_groups.json"

    with open(models_path, "r") as f:
        data = json.load(f)

    data["cc_narval"] = data["narval"]
    data["cc_cedar"] = data["cedar"]

    return data


def get_run_group_to_cluster_name(results: pd.DataFrame) -> Dict[str, str]:
    # Map run group to their hostname
    run_group_to_cluster_name = {}
    multiple_hosts = set()
    for _, row in results.iterrows():
        hostname = row["host"]
        run_group = row["run_group"]

        cluster_name = get_compute_cluster(hostname)
        # If the run group is already in the dict, make sure the hostname is the same

        if run_group in run_group_to_cluster_name:
            # assert run_group_to_cluster_name[run_group] == cluster_name, (
            #     f"Run group {run_group} has multiple hostnames: "
            #     f"{run_group_to_cluster_name[run_group]} and {cluster_name}"
            # )
            if run_group_to_cluster_name[run_group][0] != cluster_name:
                multiple_hosts.add(run_group)
                # Resolve the conflict by choosing the most recent one based on the
                # 'created_at' field in the row. Which has values like '2023-01-16T03:51:53'
                previous_row = run_group_to_cluster_name[run_group][1]
                previous_created_at = previous_row["created_at"]
                current_created_at = row["created_at"]

                # Convert to datetime
                if datetime.datetime.fromisoformat(
                    previous_created_at
                ) < datetime.datetime.fromisoformat(current_created_at):
                    run_group_to_cluster_name[run_group] = (cluster_name, row)

        else:
            run_group_to_cluster_name[run_group] = (cluster_name, row)

    # Remove row tuple from run_group_to_cluster_name
    run_group_to_cluster_name = {k: v[0] for k, v in run_group_to_cluster_name.items()}
    return run_group_to_cluster_name
