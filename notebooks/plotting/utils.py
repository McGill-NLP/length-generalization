import datetime
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Callable, Dict, Any

import numpy as np
import pandas as pd
from datasets import tqdm
from wandb import Api
from wandb.apis.public import Run


def get_entity_name() -> str:
    config_dr = Path(__file__).parent.parent.parent / "configs"
    with (config_dr / "entity_name.json").open() as f:
        return json.load(f)["entity_name"]


def get_project_name() -> str:
    config_dr = Path(__file__).parent.parent.parent / "configs"
    with (config_dr / "project_name.json").open() as f:
        return json.load(f)["project_name"]


def get_repo_dir() -> Path:
    return Path(__file__).parent.parent.parent


def wandb_result_to_dataframe(
    wandb_api: Api,
    group_prefix: str = "",
    is_valid_group_fn: Optional[Callable[[List[Run]], bool]] = None,
    aggregate_group_fn: Callable[[List[Run]], Dict[str, Any]] = None,
    last_check_datatime: str = None,
) -> pd.DataFrame:
    entity_name = get_entity_name()
    project_name = get_project_name()

    # Get all runs with their group starting with `group_prefix`
    runs = wandb_api.runs(
        f"{entity_name}/{project_name}",
        {
            "$and": [
                {"created_at": {"$gt": last_check_datatime}},
                {"group": {"$regex": f"{group_prefix}.*"}},
            ],
        },
    )
    runs = list(runs)

    print("Number of received runs:", len(runs))

    # Group runs by their group name
    runs_by_group = defaultdict(list)
    for run in runs:
        runs_by_group[run.group].append(run)

    print("Number of groups:", len(runs_by_group))

    invalid_groups = []
    valid_groups = []
    for group_name, runs_group in runs_by_group.items():
        try:
            if is_valid_group_fn is not None:
                if not is_valid_group_fn(runs_group):
                    invalid_groups.append(group_name)
                    continue
            valid_groups.append(group_name)
        except Exception as e:
            print(f"Error in group {group_name}: {e}")
            continue

    print("Number of valid groups:", len(valid_groups))
    data_frame_data = []
    for group_name in tqdm(valid_groups):
        try:
            runs_group = runs_by_group[group_name]
            data_frame_data.append(aggregate_group_fn(runs_group))
        except Exception as e:
            print(f"Error in group {group_name}: {e}")
            continue

    return pd.DataFrame(data_frame_data)



if __name__ == "__main__":
    wandb_api = Api(
        overrides={
            "project": get_project_name(),
            "entity": get_entity_name(),
        }
    )

    save_all_len_gen_experiments(wandb_api)
