import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Dict, Tuple, Any, Callable

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from wandb import Api
from wandb.apis.public import Run

from notebooks.plotting.utils import (
    wandb_result_to_dataframe,
    get_entity_name,
    get_project_name,
)


def get_generalization_ranks(
    acc_list: List[Tuple[int, int, int]]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    acc_dict = defaultdict(lambda: dict())
    for step, key, acc in acc_list:
        acc_dict[key][step] = acc

    earliest_gen = defaultdict(lambda: -1)

    for key in acc_dict.keys():
        best_step = 1e10
        for step, acc in acc_dict[key].items():
            if acc >= 0.9:
                if step < best_step:
                    best_step = step

        earliest_gen[key] = best_step

    earliest_gen_lst = list(earliest_gen.items())
    earliest_gen_lst.sort(key=lambda x: x[1])

    step_to_rank = {}
    ranks = {}
    for key, step in earliest_gen_lst:
        if step == 1e10:
            continue
        if step not in step_to_rank:
            step_to_rank[step] = len(step_to_rank)
        ranks[key] = step_to_rank[step]

    return ranks, earliest_gen


def plot_iid_generalization_ranks(
    runs: Optional[List[Run]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    title_kwargs: Optional[dict] = None,
    wandb_api: Optional[Api] = None,
):
    key_to_rankCount = defaultdict(lambda: defaultdict(lambda: 0))
    key_to_genStep = defaultdict(lambda: defaultdict(lambda: 0))
    for run in runs:
        artifact = wandb_api.artifact(
            f"run-{run.id}-evaluated_accvalidduring_training_table:latest"
        )
        path = artifact.get_path(
            "evaluated_acc/valid/during_training_table.table.json"
        ).download()
        with open(path) as f:
            validation_accs = json.load(f)["data"]
        ranks, earliest_generalization = get_generalization_ranks(validation_accs)

        for key, rank in ranks.items():
            key_to_rankCount[key][rank] += 1

        for key, step in earliest_generalization.items():
            key_to_genStep[key][step] += 1

    sorted_keys = sorted(key_to_rankCount.keys())

    # Create a heatmap of the ranks
    rank_counts = np.zeros((len(sorted_keys), len(sorted_keys)))
    for i, key in enumerate(sorted_keys):
        for rank in range(len(sorted_keys)):
            rank_counts[i, rank] = key_to_rankCount[key][rank] / sum(
                key_to_rankCount[key].values()
            )

    # Create axis if not provided
    if ax is None:
        sns.set_theme(style="white")
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 5), dpi=300)

    if title_kwargs is None:
        title_kwargs = {}

    ax.imshow(rank_counts, cmap="Blues")
    ax.set_xticks(range(len(sorted_keys)))
    ax.set_yticks(range(len(sorted_keys)))
    ax.set_yticklabels(sorted_keys)

    for i, key in enumerate(sorted_keys):
        for j in range(len(sorted_keys)):
            text = ax.text(
                j, i, key_to_rankCount[key][j], ha="center", va="center", color="w"
            )

    ax.set_title("Generalization Rank", **title_kwargs)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Length bucket")


def plot_iid_generalization_steps(
    runs: Optional[List[Run]] = None,
    fig: Optional[Figure] = None,
    ax: Optional[Axes] = None,
    title_kwargs: Optional[dict] = None,
    wandb_api: Optional[Api] = None,
):
    key_to_rankCount = defaultdict(lambda: defaultdict(lambda: 0))
    key_to_genStep = defaultdict(lambda: defaultdict(lambda: 0))
    for run in runs:
        artifact = wandb_api.artifact(
            f"run-{run.id}-evaluated_accvalidduring_training_table:latest"
        )
        path = artifact.get_path(
            "evaluated_acc/valid/during_training_table.table.json"
        ).download()
        with open(path) as f:
            validation_accs = json.load(f)["data"]
        ranks, earliest_generalization = get_generalization_ranks(validation_accs)

        for key, rank in ranks.items():
            key_to_rankCount[key][rank] += 1

        for key, step in earliest_generalization.items():
            key_to_genStep[key][step] += 1

    sorted_keys = sorted(key_to_rankCount.keys())

    # Create a heatmap of the ranks
    rank_counts = np.zeros((len(sorted_keys), len(sorted_keys)))
    for i, key in enumerate(sorted_keys):
        for rank in range(len(sorted_keys)):
            rank_counts[i, rank] = key_to_rankCount[key][rank] / sum(
                key_to_rankCount[key].values()
            )

    # Create axis if not provided
    if ax is None:
        sns.set_theme(style="white")
        fig, axes = plt.subplots(
            nrows=(len(sorted_keys)),
            ncols=1,
            sharex="row",
            figsize=(5, 3 * len(sorted_keys)),
            dpi=300,
        )
    else:
        axes = ax

    for ax_idx, key in enumerate(sorted_keys):
        sorted_steps = sorted(key_to_genStep[key].keys())
        xs = sorted_steps
        ys = [key_to_genStep[key][step] for step in sorted_steps]

        ax = axes[ax_idx]

        sns.barplot(x=xs, y=ys, hue=None, ax=ax)
        ax.set_title(key)


def save_all_len_gen_experiments(
    wandb_api: Api,
    save_path: Optional[Path] = None,
    group_prefix: str = "SW-",
    hp_metric_name: str = "pred/valid_seq_acc",
    ignore_created_at: bool = False,
    min_seed_runs: int = 3,
    min_hp_runs: int = 5,
    skip_unfinished_runs: bool = True,
):
    def is_valid_run_group(runs: List[Run]) -> bool:
        seed_runs = []
        hp_runs = []

        if len(runs) == 0:
            return False

        group_name = runs[0].group
        for run in runs:
            if skip_unfinished_runs and run.state != "finished":
                continue

            if "w_scratchpad" in group_name:
                try:
                    if not run.config["dataset"]["instance_processor"].get("include_scratchpad", False):
                        print("Skipping run", run.id, f"because it doesn't have scratchpad in group {group_name}")
                        continue
                except KeyError as e:
                    print(e)
                    continue


            if run.job_type == "best_run_seed_exp":
                seed_runs.append(run)
            elif run.job_type == "hp_exp":
                hp_runs.append(run)

        if len(seed_runs) < min_seed_runs or len(hp_runs) < min_hp_runs:
            print(
                "Invalid group:",
                runs[0].group,
                f"link: https://wandb.ai/{get_entity_name()}/{get_project_name()}/groups/{runs[0].group}",
                len(seed_runs),
                len(hp_runs),
            )
            return False

        return True

    def aggregate_group_fn(runs: List[Run]) -> Dict[str, Any]:
        seed_runs = []
        hp_runs = []
        for run in runs:
            if run.job_type == "best_run_seed_exp":
                seed_runs.append(run)
            elif run.job_type == "hp_exp":
                hp_runs.append(run)

        # Get the last run based on creation time
        last_run = sorted(runs, key=lambda run: run.created_at)[-1]
        dataset = last_run.config["dataset"]["name"]
        dataset_split = last_run.config["dataset"]["split"]
        include_scratchpad = "n/a"
        if (
            "instance_processor" in last_run.config["dataset"]
            and "include_scratchpad" in last_run.config["dataset"]["instance_processor"]
        ):
            include_scratchpad = last_run.config["dataset"]["instance_processor"][
                "include_scratchpad"
            ]

        o = {
            "group_name": runs[0].group,
            "created_at": last_run.created_at,
            "dataset": dataset,
            "split": dataset_split,
            "include_scratchpad": include_scratchpad,
            "url": f"https://wandb.ai/{get_entity_name()}/{get_project_name()}/groups/{runs[0].group}",
            "num_seed_runs": len(seed_runs),
            "num_hp_runs": len(hp_runs),
        }

        hp_numeric_summary_keys = set()
        for run in hp_runs:
            for key in run.summary.keys():
                if isinstance(run.summary[key], (int, float, np.ndarray)):
                    hp_numeric_summary_keys.add(key)

        seed_numeric_summary_keys = set()
        for run in seed_runs:
            for key in run.summary.keys():
                if isinstance(run.summary[key], (int, float, np.ndarray)):
                    seed_numeric_summary_keys.add(key)

        hp_numeric_summary_keys = list(hp_numeric_summary_keys)
        seed_numeric_summary_keys = list(seed_numeric_summary_keys)

        # Aggregate with max over all hp runs
        for k in hp_numeric_summary_keys:
            metric_values = [run.summary[k] for run in hp_runs if k in run.summary]
            if len(metric_values) == 0:
                continue
            metric_max = np.max(metric_values)
            o[f"hp__{k}"] = metric_max

        # Get the best config from the hp runs
        metric_values = [run.summary.get(hp_metric_name, None) for run in hp_runs]
        if any(v is None for v in metric_values):
            o["best_hp_config"] = seed_runs[0].json_config
        else:
            best_hp_run = hp_runs[np.argmax(metric_values)]
            o["best_hp_config"] = best_hp_run.json_config

        # Aggregate with mean and std over all seed runs
        for k in seed_numeric_summary_keys:
            metric_values = [run.summary[k] for run in seed_runs if k in run.summary]
            if len(metric_values) == 0:
                continue
            metric_mean = np.mean(metric_values)
            metric_std = np.std(metric_values)
            o[f"seed__{k}"] = metric_mean
            o[f"seedStd__{k}"] = metric_std

            # Create a text column for viewing the seed run
            o[f"seed_txt__{k}"] = f"{metric_mean:.4f}±{metric_std:.4f}"

        return o

    repo_dir = Path(__file__).parent.parent.parent
    results_dir = repo_dir / "results"
    results_dir.mkdir(exist_ok=True, parents=True)
    dataframe_path = results_dir / "len_gen_experiments.csv"
    if dataframe_path.exists():
        previously_finished_jobs = pd.read_csv(str(dataframe_path))
        print("Number of previously finished jobs:", len(previously_finished_jobs))
    else:
        previously_finished_jobs = []

    if not ignore_created_at and len(previously_finished_jobs) > 0:
        last_check_datetime = previously_finished_jobs.iloc[-1]["created_at"]
    else:
        last_check_datetime = datetime.datetime(
            year=2010, month=1, day=1, tzinfo=datetime.timezone.utc
        ).isoformat()

    print("Last check datetime:", last_check_datetime)

    newly_finished_jobs = wandb_result_to_dataframe(
        wandb_api,
        group_prefix=group_prefix,
        is_valid_group_fn=is_valid_run_group,
        aggregate_group_fn=aggregate_group_fn,
        last_check_datatime=last_check_datetime,
    )
    if len(newly_finished_jobs) > 0:
        newly_finished_jobs["created_at_o"] = pd.to_datetime(
            newly_finished_jobs["created_at"]
        )
        newly_finished_jobs.sort_values(by="created_at_o", inplace=True)
        newly_finished_jobs.drop(columns=["created_at_o"], inplace=True)

    print("Number of newly finished jobs:", len(newly_finished_jobs))

    if len(previously_finished_jobs) > 0:
        data_dfs = pd.concat(
            [previously_finished_jobs, pd.DataFrame(newly_finished_jobs)]
        )
    else:
        data_dfs = pd.DataFrame(newly_finished_jobs)

    data_dfs.to_csv(str(dataframe_path), index=False)


def plot_len_generalization_positional_encoding_result(
    df: pd.DataFrame,
    metric_column: str,
    metric_label: Optional[str] = None,
    get_positional_encoding_type_fn: Callable[[Dict[str, Any]], str] = None,
    ax=None,
):
    sns.set_theme(style="whitegrid")

    if metric_label is None:
        metric_label = metric_column

    def extract_pos_enc_from_config(x: Dict[str, Any]) -> str:
        config = json.loads(x["best_hp_config"])
        return config["model"]["value"]["position_encoding_type"]

    if get_positional_encoding_type_fn is None:
        get_positional_encoding_type_fn = extract_pos_enc_from_config

    df["Pos. Enc."] = df.apply(get_positional_encoding_type_fn, axis=1)

    # Sort by
    df = df.sort_values(by=["Pos. Enc."])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    df.plot(
        x="Pos. Enc.",
        y=f"seed__{metric_column}",
        yerr=f"seedStd__{metric_column}",
        kind="bar",
        ax=ax,
    )
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Positional Encoding")

def plot_len_generalization_positional_encoding_result_per_bucket(
    df: pd.DataFrame,
    metric_column_sep: str,
    metric_label: Optional[str] = None,
    get_positional_encoding_type_fn: Callable[[Dict[str, Any]], str] = None,
    ax=None,
):
    sns.set_theme(style="whitegrid")

    if metric_label is None:
        metric_label = metric_column_sep

    def extract_pos_enc_from_config(x: Dict[str, Any]) -> str:
        config = json.loads(x["best_hp_config"])
        return config["model"]["value"]["position_encoding_type"]

    if get_positional_encoding_type_fn is None:
        get_positional_encoding_type_fn = extract_pos_enc_from_config

    df["Pos. Enc."] = df.apply(get_positional_encoding_type_fn, axis=1)

    target_columns = [c for c in df.columns if f"seed__{metric_column_sep}" in c]
    target_columns.sort()

    xdf_data = []

    for _, row in df.iterrows():
        for c in target_columns:
            xdf_data.append({
                "Pos. Enc.": row["Pos. Enc."],
                "Bucket": c.replace(f"seed__{metric_column_sep}", "")[-1],
                metric_label: row[c],
            })

    xdf = pd.DataFrame(xdf_data)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    sns.barplot(
        x="Pos. Enc.",
        y=metric_label,
        hue="Bucket",
        data=xdf,
        ax=ax,
    )
    ax.set_ylabel(metric_label)
    ax.set_xlabel("Positional Encoding")

if __name__ == "__main__":
    import wandb
    wandb_api = wandb.Api(
        overrides={
            "project": get_project_name(),
            "entity": get_entity_name(),
        },
    )
    save_all_len_gen_experiments(wandb_api, min_seed_runs=1, min_hp_runs=1, skip_unfinished_runs=False)
