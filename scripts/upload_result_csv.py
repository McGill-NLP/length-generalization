import copy
import csv
import json
import os
import re

import wandb
from wandb.apis.public import Run


def get_test_phase_shifts(summary_key: str) -> int:
    if "bias=" in summary_key:
        return int(re.search(r".*_bias=((\d)+).*", summary_key).group(1))
    elif "_random_" in summary_key:
        return -1
    else:
        raise ValueError()


def get_metric_name(summary_key: str) -> str:
    return re.search(r".*_(bias=(\d+)|(random))_(.*)", summary_key).group(4)


def get_train_shift(cfg):
    return cfg["dataset"]["value"]["train_shift"]["shift_bias_list"][0]


def get_dataset_name(cfg):
    return cfg["dataset"]["value"]["task_name"]


def get_model_name(cfg):
    return cfg["model"]["value"]["hf_model_name"]


def get_summary_key(test_phase_shift: int, metric: str):
    if test_phase_shift == -1:
        ph_shift = "random"
    else:
        ph_shift = f"bias={test_phase_shift}"

    return f"pred/_PS_{ph_shift}_{metric}"


def get_seed(cfg):
    return cfg["global_vars"]["value"]["seed"]


if __name__ == "__main__":
    with open("configs/project_name.json") as f:
        proj_name = json.load(f)["project_name"]

    with open("configs/entity_name.json") as f:
        entity_name = json.load(f)["entity_name"]

    tag_name = os.environ["EXP_WANDB_TAGS"]
    assert len(tag_name.split(",")) == 1

    api = wandb.Api(
        overrides={"entity": entity_name, "project": proj_name}, timeout=120
    )

    query = api.runs(f"{entity_name}/{proj_name}", filters={"tags": tag_name})
    runs = []
    while True:
        try:
            runs.append(query.next())
        except StopIteration:
            break

    if len(runs) == 0:
        exit()

    first_run = runs[0]
    first_run_config = json.loads(first_run.json_config)

    fieldnames = [
        "dataset",
        "model",
        "train_phase_shift",
        "test_phase_shift",
        "seed",
        "metric",
    ]

    ignore_keys = [
        "runtime",
        "loss",
        "samples_per_second",
        "num_samples",
        "steps_per_second",
        "hp_metric",
    ]
    pred_keys = [
        k
        for k in first_run.summary.keys()
        if k.startswith("pred/_PS")
        and not any(k.endswith(f"_{t}") for t in ignore_keys)
    ]
    other_metrics = sorted(set(map(get_metric_name, pred_keys)))
    fieldnames += other_metrics

    all_test_phase_shifts = sorted(set(map(get_test_phase_shifts, pred_keys)))

    with open("best_run_info.json") as f:
        info = json.load(f)
        sweep_name, sweep_id = info["sweep_name"], info["sweep_id"]

    filename = f"{sweep_id}_{sweep_name}.csv"
    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for run in runs:
            run: Run
            cfg = json.loads(run.json_config)
            base_d = {
                "dataset": get_dataset_name(cfg),
                "model": get_model_name(cfg),
                "train_phase_shift": get_train_shift(cfg),
                "seed": get_seed(cfg),
            }

            for test_ps in all_test_phase_shifts:
                d = copy.deepcopy(base_d)
                d["test_phase_shift"] = test_ps
                for metric in ["hp_metric"] + other_metrics:
                    key = get_summary_key(test_ps, metric)
                    value = run.summary[key]
                    metric_name = "metric" if metric == "hp_metric" else metric
                    d[metric_name] = value

                writer.writerow(d)

    with wandb.init(
        project=proj_name,
        entity=entity_name,
        job_type="result_upload",
        name="result_uploader",
        group=first_run.group,
    ) as run:
        artifact = wandb.Artifact(name=f"result_{sweep_id}_{sweep_name}", type="result")
        artifact.add_file(filename)
        run.log_artifact(artifact)
