import os
import re
import uuid
from typing import Dict, Tuple

import fire

POSITIONAL_ENCODINGS = [
    "pe_t5",
    "pe_none",
    "pe_abs_sin",
    "pe_alibi",
    "pe_rotary",
    "pe_newRot",
]


def main(
    pe: str = None,
    base_config: str = None,
    dry_run: bool = False,
):

    if base_config is None:
        base_config = "configs/t5_dec_base.jsonnet,configs/efficiency_measure.jsonnet"

    exp_ids: Dict[str, Tuple[str]] = {}

    if pe is not None:
        pos_encoding_list = pe.split(",")
    else:
        pos_encoding_list = POSITIONAL_ENCODINGS

    # This experiments does not depend on any datasets
    # But we need to specify a dataset since the code is not flexible enough
    dataset_name = "s2s_addition"
    dataset_split = "len_tr8_ts16"

    for pe in pos_encoding_list:
        cmd = f"python scripts/upload_experiment.py"
        cmd += f" --dataset data-{dataset_name}-{dataset_split}"
        cmd += f' --configs "{base_config},configs/models/{pe}.jsonnet,configs/data/{dataset_name}.jsonnet"'
        cmd += f' --commands "analyze_all --eval_split valid"'
        cmd += f' --env "APP_SEED=42,APP_DS_SPLIT={dataset_split}"'
        cmd += f" --tags runtime_efficiency,runtime_efficiency__{pe}"

        if not dry_run:
            output = os.popen(cmd).read()
        else:
            random_exp_id = str(uuid.uuid4())[0:8]
            output = f"Exp Key: {random_exp_id}\n"

        # Get the experiment id from the output using a regex
        try:
            exp_id = re.search(r".*Exp Key: (.*)\n", output).group(1)
            exp_ids[exp_id] = (pe,)
        except Exception as e:
            print(f"Failed to get exp_id from output: {output}")
            raise e

        print(f"Experiment id: {exp_id}")

    exp_ids_str = ",".join(sorted(exp_ids.keys()))

    # We run these experiments on Mila's V100 GPUs
    cmd = (
        f"export WANDB_PROJECT=len_gen && "
        f"launcher -p mila  --image_name pt_v7.sif "
        f'-s "--partition=long '
        f"--gres=gpu:v100:1 -t 24:00:00 -c 16 --mem=64G "
        f"-x 'kepler5'"
        f'" '
        f"--nodup {exp_ids_str}"
    )

    print(f"Run the following command to launch the experiments:")
    print(cmd)


if __name__ == "__main__":
    fire.Fire(main)
