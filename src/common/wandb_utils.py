import datetime
import json
from pathlib import Path
import os
from typing import List, Dict

import jsonlines
import pandas as pd
from tqdm.auto import tqdm

from common import nest


CUSTOMIZED_RUN_FRAGMENT = """fragment RunFragment on Run {
    id
    tags
    name
    displayName
    sweepName
    state
    config
    group
    jobType
    commit
    readOnly
    createdAt
    heartbeatAt
    description
    notes
    runInfo {
        program
        args
        os
        python
        gpuCount
        gpu
    }
    host
    systemMetrics
    summaryMetrics
    historyLineCount
    user {
        name
        username
    }
    historyKeys
}"""

import wandb.apis.public as wandb_public_api

wandb_public_api.RUN_FRAGMENT = CUSTOMIZED_RUN_FRAGMENT
from wandb_gql import Client, gql

wandb_public_api.Runs.QUERY = gql(
    """
    query Runs($project: String!, $entity: String!, $cursor: String, $perPage: Int = 50, $order: String, $filters: JSONString) {
        project(name: $project, entityName: $entity) {
            runCount(filters: $filters)
            readOnly
            runs(filters: $filters, after: $cursor, first: $perPage, order: $order) {
                edges {
                    node {
                        ...RunFragment
                    }
                    cursor
                }
                pageInfo {
                    endCursor
                    hasNextPage
                }
            }
        }
    }
    %s
    """
    % CUSTOMIZED_RUN_FRAGMENT
)
import wandb
from wandb.apis.public import Run

# Taken from https://www.oreilly.com/library/view/python-cookbook/0596001673/ch04s23.html
def add_sys_path(new_path):
    import sys, os

    # Avoid adding nonexistent paths
    if not os.path.exists(new_path):
        return -1

    # Standardize the path. Windows is case-insensitive, so lowercase
    # for definiteness.
    new_path = os.path.abspath(new_path)
    if sys.platform == "win32":
        new_path = new_path.lower()

    # Check against all currently available paths
    for x in sys.path:
        x = os.path.abspath(x)
        if sys.platform == "win32":
            x = x.lower()
        if new_path in (x, x + os.sep):
            return 0
    sys.path.append(new_path)
    return 1


def get_entity_name() -> str:
    config_dr = get_repo_dir() / "configs"
    with (config_dr / "entity_name.json").open() as f:
        return json.load(f)["entity_name"]


def get_project_name() -> str:
    config_dr = get_repo_dir() / "configs"
    with (config_dr / "project_name.json").open() as f:
        return json.load(f)["project_name"]


def get_repo_dir() -> Path:
    return Path(__file__).parent.parent.parent


def get_wandb_api() -> wandb.Api:
    return wandb.Api(
        overrides={
            "project": get_project_name(),
            "entity": get_entity_name(),
        },
        timeout=120,
    )


def get_pretty_scratchpad_config_name_for_sort(config: str) -> str:
    config_names = {
        "i1_c0_o1_v0_r1": 'Removing "Step Input"',
        "i1_c1_o1_v0_r0": 'Removing "Remaining Parts"',
        "i0_c1_o1_v0_r0": "Minimal (Only Compute+Output)",
        "i1_c1_o1_v0_r1": "Full",
        "i1_c1_o0_v0_r1": 'Removing "Step Output"',
        "i0_c1_o1_v0_r1": 'Removing "Step Input"',
        "no_scratchpad": "No Scratchpad",
    }
    return config_names[config]


def get_pretty_scratchpad_config_name_for_lego(config: str) -> str:
    config_names = {
        "i1_c0_o1_v0_r1": 'Removing "Step Input"',
        "i1_c1_o1_v0_r0": 'Removing "Remaining Parts"',
        "i0_c1_o1_v0_r0": "Minimal (Only Compute+Output)",
        "i1_c1_o1_v0_r1": "Full",
        "i1_c1_o0_v0_r1": 'Removing "Step Output"',
        "i0_c1_o1_v0_r1": 'Removing "Step Input"',
        "no_scratchpad": "No Scratchpad",
    }
    return config_names[config]


def get_pretty_scratchpad_config_name(config: str, ds: str, ds_split: str) -> str:
    if ds == "s2s_sort":
        return get_pretty_scratchpad_config_name_for_sort(config)
    if ds == "s2s_lego":
        return get_pretty_scratchpad_config_name_for_lego(config)
    else:
        if config == "no_scratchpad":
            return "No Scratchpad"
        elif config == "i0_c1_o1_v0_r0":
            return "Minimal (Only Compute+Output)"
        elif config == "i0_c1_o1_v1_r1":
            return 'Removing "Step Input"'
        elif config == "i1_c1_o1_v1_r1":
            return "Full"
        elif config == "i1_c0_o1_v1_r1":
            return 'Removing "Step Computation"'
        elif config == "i1_c1_o0_v1_r1":
            return 'Removing "Step Output"'
        elif config == "i1_c1_o1_v0_r1":
            return 'Removing "Intermediate Vars"'
        elif config == "i1_c1_o1_v1_r0":
            return 'Removing "Remaining Parts"'
    raise ValueError(config, ds)


def df_prettify_scratchpad_config_name(row):
    return get_pretty_scratchpad_config_name(
        row["scratchpad_config"], row["ds"], row["ds_split"]
    )


def get_pretty_scratchpad_config_order() -> List[str]:
    configs = [
        "No Scratchpad",
        "Minimal (Only Compute+Output)",
        'Removing "Step Input"',
        'Removing "Step Computation"',
        'Removing "Step Output"',
        'Removing "Intermediate Vars"',
        'Removing "Remaining Parts"',
        "Full",
    ]
    return configs


def get_pretty_scratchpad_config_order2() -> List[str]:
    configs = [
        "No Scratchpad",
        "Minimal (Only Compute+Output)",
        "Full",
        'Removing "Step Input"',
        'Removing "Step Computation"',
        'Removing "Step Output"',
        'Removing "Intermediate Vars"',
        'Removing "Remaining Parts"',
    ]
    return configs

def get_pretty_scratchpad_config_order3() -> List[str]:
    configs = [
        "No Scratchpad",
        "Full",
        'Removing "Step Input"',
        'Removing "Step Computation"',
        'Removing "Step Output"',
        'Removing "Intermediate Vars"',
        'Removing "Remaining Parts"',
        "Minimal (Only Compute+Output)",
    ]
    return configs


def get_pretty_pe_name(pe_name: str) -> str:
    pe_name_to_pretty_pe_name = {
        "abs_sinusoid": "Absolute Sinusoid",
        "t5_relative_bias": "T5's Relative Bias",
        "alibi": "ALiBi",
        "none": "No PE",
        "rotary": "Rotary",
        "new_rotary": "Rotary (Fixed)",
    }

    return pe_name_to_pretty_pe_name.get(pe_name, pe_name)


def get_pretty_pe_order() -> List[str]:
    pe_order = [
        "none",
        "abs_sinusoid",
        "rotary",
        "new_rotary",
        "alibi",
        "t5_relative_bias",
    ]

    return [get_pretty_pe_name(pe) for pe in pe_order]


def get_pretty_dataset_name(dataset: str, split: str) -> str:
    if dataset == "s2s_addition":
        return "Addition"
    if dataset == "s2s_sort":
        if split == "len_sngd_tr8_ts16":
            return "Sort (Single Token)"
        return "Sort (Multi Digit)"
    if dataset == "s2s_poly":
        return "Polynomial Eval."
    if dataset == "s2s_lego":
        if split == "len_tr8_ts16":
            return "LEGO"
        return "LEGO (Perm.)"
    if dataset == "s2s_copy":
        if split == "rsc_tr20_ts40":
            return "Copy (Repeat)"
        if split == "rdc_tr20_ts40":
            return "Copy (Repeat+Change)"
        if split == "rsc2x_tr20_ts40":
            return "Copy (Repeat 2x)"
        if split == "cmc_tr20_ts40":
            return "Copy"
        if split == "cmc2x_tr20_ts40":
            return "Copy (2x)"
    if dataset == "s2s_reverse":
        if split == "mc_tr20_ts40":
            return "Reverse"
        if split == "mc2x_tr20_ts40":
            return "Reverse (2x)"
        if split == "mcrv_tr20_ts40":
            return "Reverse+Copy"
    if dataset == "s2s_parity":
        return "Parity"
    if dataset == "scan":
        return "SCAN"
    if dataset == "pcfg":
        return "PCFG (Productivity)"

    return f"{dataset} ({split})"


def get_best_performing_scratchpad_config(
    df: pd.DataFrame, ds: str, ds_split: str, pe: str
) -> str:
    # Dataframe columns:
    # pe, ds, ds_split, scratchpad_config, seed, output_acc, length (always == -1)
    xdf = df[
        (df["seq_length"] == -1)
        & (df["prediction_split"] == "test")
        & (df["output_acc"].notnull())
    ]
    xdf = xdf[(xdf["ds"] == ds) & (xdf["ds_split"] == ds_split) & (xdf["pe"] == pe)]

    if len(xdf) == 0:
        raise ValueError(f"No data for {ds}, {ds_split}, {pe}")

    xdf = xdf.copy()

    # Compute the mean and std of the output accuracy
    xdf = xdf.groupby("scratchpad_config").agg({"output_acc": ["mean", "std"]})
    xdf.columns = ["_".join(x) for x in xdf.columns]
    xdf = xdf.reset_index()
    xdf = xdf.sort_values("output_acc_mean", ascending=False)
    return xdf.iloc[0]["scratchpad_config"]


def get_aggr_dataframe_with_mean_std(
    df: pd.DataFrame, aggr_cols: List[str], metric_name: str
) -> pd.DataFrame:
    xdf = df.groupby(aggr_cols).agg({metric_name: ["mean", "std"]})
    xdf.columns = ["_".join(x) for x in xdf.columns]
    xdf = xdf.reset_index()
    return xdf


def create_mask(df: pd.DataFrame, key_values: Dict[str, str]) -> pd.Series:
    # First, produce mask that is True for all rows based on df
    mask = pd.Series([True] * len(df), index=df.index)
    for key, value in key_values.items():
        mask = mask & (df[key] == value)
    return mask


def get_loaded_runtime(wandb_run_id: str, wandb_api: wandb.Api):
    run = wandb_api.run(f"{get_entity_name()}/{get_project_name()}/{wandb_run_id}")

    # Download the dataset artifact

    # Find the agent run
    runs = wandb_api.runs(
        f"{get_entity_name()}/{get_project_name()}",
        {
            "$and": [
                {"group": {"$eq": f"{run.group}"}},
            ],
        },
    )
    runs = [r for r in runs if r.job_type == "agent"]
    agent_run = sorted(
        runs, key=lambda x: datetime.datetime.fromisoformat(x.created_at)
    )[-1]
    dataset_artifact = [a for a in agent_run.used_artifacts() if a.type == "dataset"][0]
    dataset_artifact.download("data")

    run_config = run.config

    experiment_dir = Path(run_config["directory"])
    experiment_name = Path(run_config["exp_name"])

    assert (
        experiment_dir / experiment_name
    ).exists(), (
        f"Experiment directory {experiment_dir / experiment_name} does not exist."
    )

    config_path = experiment_dir / experiment_name / "config.json"

    add_sys_path(f"{get_repo_dir()}/src")
    from main import EntryPoint

    os.environ.update({"APP_EXPERIMENT_NAME": run_config["exp_name"]})

    ep = EntryPoint(str(config_path), debug_mode=True)
    return ep._exp, ep._config


def get_result_name(tags: List[str]) -> str:
    return "_".join(tags)


def get_launcher_id(run: Run) -> str:
    if run.job_type == "agent":
        return run.id

    launcher_tag = [t for t in run.tags if t.startswith("launched_by_")]
    if len(launcher_tag) == 0:
        return None
    launcher_tag = launcher_tag[0]
    return launcher_tag.split("launched_by_")[1]


def load_dataframe_from_jsonlines(path: Path) -> pd.DataFrame:
    data = []
    with jsonlines.open(path) as reader:
        for obj in reader:
            data.append(obj)
    return pd.DataFrame.from_records(data)


def download_and_load_results(
    tags: List[str],
    force_download: bool = False,
    key: str = None,
) -> pd.DataFrame:
    if key is None:
        key = get_result_name(tags)

    result_dir = get_repo_dir() / "results"
    result_dir.mkdir(exist_ok=True, parents=True)

    result_path = result_dir / f"{key}.jsonl"
    if result_path.exists() and not force_download:
        return load_dataframe_from_jsonlines(result_path)

    wandb_api = get_wandb_api()
    runs = wandb_api.runs(
        f"{get_entity_name()}/{get_project_name()}",
        filters={"tags": {"$in": tags}},
    )

    df_data = []
    for run in tqdm(runs, total=len(runs)):
        config = run.config
        config = nest.flatten(config, separator=".")
        config = {f"cfg__{k}": v for k, v in config.items()}

        summary = run.summary._json_dict
        summary = nest.flatten(summary, separator="#")
        summary = {f"sum__{k}": v for k, v in summary.items()}

        run_info = run.run_info
        if run_info is None:
            run_info = {}
        run_info = nest.flatten(run_info, separator=".")
        run_info = {f"runInfo__{k}": v for k, v in run_info.items()}

        group = run.group
        df_data.append(
            {
                "run_group": group,
                "job_type": run.job_type,
                "tags": run.tags,
                "launcher_id": get_launcher_id(run),
                "state": run.state,
                "id": run.id,
                "group_url": f"https://wandb.ai/kzmnjd/len_gen/groups/{group}",
                "run_url": run.url,
                "created_at": run.created_at,
                "host": run.host,
                **run_info,
                **config,
                **summary,
            }
        )

    print("Building dataframe...")
    df = pd.DataFrame(df_data)

    print("Saving results to", result_path)

    # Save the results as jsonlines
    with jsonlines.open(result_path, mode="w") as writer:
        writer.write_all(df_data)

    return df


if __name__ == "__main__":
    df = download_and_load_results(
        tags=["scratchpad_f"],
        force_download=True,
    )
    len(df)
