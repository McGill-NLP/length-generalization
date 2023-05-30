import copy
import itertools
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import wandb


def get_entity_name() -> str:
    config_dr = Path(__file__).parent.parent / "configs"
    with (config_dr / "entity_name.json").open() as f:
        return json.load(f)["entity_name"]


def get_project_name() -> str:
    config_dr = Path(__file__).parent.parent / "configs"
    with (config_dr / "project_name.json").open() as f:
        return json.load(f)["project_name"]


def is_run_complete(run: wandb.apis.public.Run) -> bool:
    # Assert training is done
    try:
        max_steps = run.config["trainer"]["max_steps"]
    except Exception as e:
        return False

    h = run.history(samples=500, keys=["train/loss"], x_axis="train/global_step")
    if h is None or len(h) == 0:
        return False
    last_step = h.iloc[-1]["train/global_step"]

    if abs(last_step - max_steps) > 0.01 * max_steps:
        return False

    # Assert validation is done
    if (
        "pred/valid_acc_overall" not in run.summary
        or run.summary["pred/valid_acc_overall"] is None
    ):
        return False

    return True


def get_param(
    run: wandb.apis.public.Run, params: List[Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    for param in params:
        all_match = True
        for p_name, p_value in param.items():
            p_name_parts = p_name.split(".")[::-1]
            value_from_run = copy.deepcopy(run.config)
            while len(p_name_parts) > 0:
                key = p_name_parts.pop()
                value_from_run = value_from_run[key]

            if str(value_from_run) != str(p_value):
                all_match = False
                break

        if all_match:
            return param

    return None


def main():
    sweep_id = os.environ["SWEEP_ID"]

    api = wandb.Api(
        overrides={"entity": get_entity_name(), "project": get_project_name()}
    )

    try:
        sweep_id_parts = sweep_id.split("/")
        if len(sweep_id_parts) == 1:
            sweep = api.sweep(f"{get_entity_name()}/{get_project_name()}/{sweep_id}")
        elif len(sweep_id_parts) == 2:
            sweep = api.sweep(f"{get_entity_name()}/{sweep_id}")
        else:
            sweep = api.sweep(sweep_id)
    except Exception as e:
        print(f"Failed to get sweep: {e}")
        exit(1)

    sweep_params_dict = {p: v["values"] for p, v in sweep.config["parameters"].items()}
    keys, values = zip(*sweep_params_dict.items())
    sweep_params: List[Dict[str, Any]] = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]
    sweep_params_keys = sorted(keys)

    def get_param_hash(params: Dict[str, Any]) -> Tuple[Any, ...]:
        return tuple(params[k] for k in sweep_params_keys)

    run_groups: Dict[Tuple[Any, ...], List[wandb.apis.public.Run]] = defaultdict(list)
    for run in sweep.runs:
        run_param = get_param(run, sweep_params)
        if run_param is None:
            continue

        run_groups[get_param_hash(run_param)].append(run)

    for sweep_param in sweep_params:
        param_hash = get_param_hash(sweep_param)
        if len(run_groups[param_hash]) == 0:
            print(f"Failed to find run for {sweep_param}")
            exit(1)

        is_group_complete = False
        for run in run_groups[param_hash]:
            if is_run_complete(run):
                is_group_complete = True
                break

        if not is_group_complete:
            print(f"No run is complete for {sweep_param}")
            exit(1)


if __name__ == "__main__":
    main()
