import os
import re
import sys
from typing import Dict, Tuple

POSITIONAL_ENCODINGS = [
    "rpe_none",
    "rpe_abs",
]

SCRATCHPAD_FORMATS = [
    None,
    "w_scratchpad",
]

DS_TO_SPLITS = {
    "sum": ["len_tr20_ts40"],
    "sum_mod": ["len_tr20_ts40"],
}

SKIP_SCRATCHPAD = ["sum", "sum_mod"]

METRIC = {
    "sum": "mse",
    "sum_mod": "accuracy",
}

def main():
    dataset_name = sys.argv[1]

    exp_ids: Dict[str, Tuple[str, str, str]] = {}

    for pe in POSITIONAL_ENCODINGS:
        for scratchpad_format in SCRATCHPAD_FORMATS:
            if scratchpad_format is not None:
                if dataset_name in SKIP_SCRATCHPAD:
                    continue

            for split in DS_TO_SPLITS[dataset_name]:
                scratchpad_config = ""
                if scratchpad_format is not None:
                    scratchpad_config = f",configs/data/{scratchpad_format}.jsonnet"

                metric = METRIC[dataset_name]

                cmd = f"scripts/upload_experiment_with_hp_sum.sh"
                cmd += f" --dataset {dataset_name}"
                cmd += f" --split {split}"
                cmd += f' --configs "configs/robertaLargeArc_cls_base.jsonnet,configs/models/{pe}.jsonnet{scratchpad_config}"'
                cmd += f" --sweep_configs configs/sweeps/training_scratch_seq_cls_{metric}.jsonnet"
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
    main()
