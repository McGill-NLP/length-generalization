import os
import re
import sys
from typing import Dict, Tuple

import fire

POSITIONAL_ENCODINGS = [
    "pe_t5",
    "pe_none",
    "pe_abs_sin",
]

SCRATCHPAD_FORMATS = [
    None,
    "w_scratchpad",
]

DS_TO_SPLITS = {
    "parity": [
        "len_tr20_ts40",
    ],
    "s2s_lego": [
        "len_tr8_ts16",
        # "len_tr8_ts16_perm",
    ],
    "s2s_copy": [
        "rsc_tr20_ts40", "rdc_tr20_ts40",
        "rsc2x_tr15_ts30",
        "cmc_tr20_ts40",
    ],
    "sc_count": [
        "sc_tr20_ts40", "mc_tr20_ts40"
    ],
    "sc_count_mod": [
        "sc_tr20_ts40", "mc_tr20_ts40"
    ],
    "sum": ["len_tr20_ts40"],
    "sum_mod": ["len_tr20_ts40"],
}

SKIP_SCRATCHPAD = ["s2s_copy", "sc_count", "sc_count_mod", "sum", "sum_mod"]


def main(dataset_name: str, pe: str = None, base_config: str = None, sweep_config: str = None, launcher: str = None):

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
        for scratchpad_format in SCRATCHPAD_FORMATS:
            if scratchpad_format is not None:
                if dataset_name in SKIP_SCRATCHPAD:
                    continue

            for split in DS_TO_SPLITS[dataset_name]:
                scratchpad_config = ""
                if scratchpad_format is not None:
                    scratchpad_config = f",configs/data/{scratchpad_format}.jsonnet"

                cmd = f"scripts/{launcher}"
                cmd += f" --dataset {dataset_name}"
                cmd += f" --split {split}"
                cmd += f' --configs "{base_config},configs/models/{pe}.jsonnet{scratchpad_config}"'
                cmd += f" --sweep_configs {sweep_config}"
                cmd += f' --commands "hp_step --eval_split valid"'
                cmd += f' --env "APP_SEED=42"'

                output = os.popen(cmd).read()

                # Get the experiment id from the output using a regex
                try:
                    exp_id = re.search(r"Exp Key: (.*)\n", output).group(1)
                    exp_ids[exp_id] = (split, pe, scratchpad_format)
                except Exception as e:
                    print(f"Failed to get exp_id from output: {output}")
                    raise e

                print((split, pe, scratchpad_format))

    # Print out all experiment ids
    for exp_id, (split, pe, scratchpad_format) in exp_ids.items():
        print(f"{exp_id}: {split} {pe} {scratchpad_format}")

    # Print out all experiment ids seperated by commas
    print("\nExperiment Keys")
    print(",".join(exp_ids.keys()))


if __name__ == "__main__":
    fire.Fire(main)
