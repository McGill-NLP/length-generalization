import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List
from tqdm.auto import tqdm

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

PATH_REPLACEMENTS = {
    "cc-cedar": {
        "/home/kzmnjd/experiments/": "/home/kzmnjd/scratch/len_gen/experiments/",
    },
    "cc-narval": {
        "/home/kzmnjd/experiments/": "/home/kzmnjd/scratch/len_gen/experiments/",
    },
    "mila": {
        "/home/mila/a/amirhossein.kazemnejad/experiments/": "/home/mila/a/amirhossein.kazemnejad/scratch/len_gen/experiments/"
    },
}


def get_true_remote_path(cluster: str, remote_path: str) -> str:
    for k, v in PATH_REPLACEMENTS[cluster].items():
        if remote_path.startswith(k):
            return remote_path.replace(k, v)

    return remote_path


def get_cluster_name(host: str) -> str:
    if "cedar" in host:
        return "cc-cedar"
    elif "narval" in host:
        return "cc-narval"
    elif host.startswith("cn-"):
        return "mila"
    else:
        return host


def list_metadata(download_dir, run) -> List[Dict[str, Any]]:
    metadata_lst: List[Dict[str, Any]] = []
    metadata_files = list(run.files())
    for file in tqdm(metadata_files, desc="Downloading metadata"):
        if file.name.startswith("attn_metadata_"):
            category = file.name.split("_")[-1].split(".")[0]
            local_path = download_dir / category / file.name
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                _ = file.download(root=local_path.parent, replace=True)
            metadata_lst.append(json.load(local_path.open()))
    return metadata_lst


def download_the_entire_run(localhost: str, run: Run, root_dir: Path = None) -> Path:
    if root_dir is None:
        # Read from the environment variable
        root_dir = Path(os.environ["ATTENTION_ANALYSIS_ROOT"])

    download_dir = root_dir / run.id
    download_dir.mkdir(parents=True, exist_ok=True)

    remote_host = get_cluster_name(run.host)

    metadata_lst = list_metadata(download_dir, run)

    if localhost != remote_host:
        # Download all attention files from the remote host
        attention_analyzer_dir = Path(metadata_lst[0]["dataset"]).parent.parent
        remote_run_dir = get_true_remote_path(remote_host, str(attention_analyzer_dir))
        remote_run_dir = f"{remote_run_dir}/*"
        remote_run_dir = str(remote_run_dir).replace("/experiments/", "/experiments/./")
        local_dst_path = Path().home() / "scratch" / "len_gen" / "experiments"

        cmd = f"rsync -avzR {remote_host}:{remote_run_dir} {local_dst_path}/"
        print(cmd)
        os.system(cmd)

    for metadata in tqdm(metadata_lst, desc="Downloading data"):
        metadata_download_dir = download_dir / str(metadata["category"])
        metadata_download_dir.mkdir(parents=True, exist_ok=True)

        remote_dir = get_true_remote_path(
            remote_host, str(Path(metadata["dataset"]).parent)
        )

        non_existing_files = []
        for file_name, file_path in metadata.items():
            if file_name == "category":
                continue
            local_file_path = metadata_download_dir / Path(file_path).name
            if not local_file_path.exists():
                non_existing_files.append(file_path)

        if len(non_existing_files) > 0:
            # If remote_host is localhost, create a symbolic link
            if remote_host == localhost:
                local_dir = Path(remote_dir)
            else:
                remote_prefix = PATH_REPLACEMENTS[remote_host]
                local_prefix = PATH_REPLACEMENTS[localhost]
                local_dir = Path(remote_dir.replace(
                    list(remote_prefix.values())[0],
                    list(local_prefix.values())[0]
                ))

            for file in local_dir.glob("*"):
                destination_link = metadata_download_dir / file.name
                if destination_link.exists():
                    continue
                destination_link.symlink_to(file)

    return download_dir


def get_localhost() -> str:
    host = os.uname()[1]
    if "cedar" in host:
        return "cc-cedar"
    elif "narval" in host:
        return "cc-narval"
    return "mila"


if __name__ == "__main__":
    # Read the first argument as list of wandb run ids
    # run_ids = "sw_d57b3a11b8a0a1eee7805d747aa3f7ab_2_146317"
    run_ids = sys.argv[1]
    run_ids = [r.strip() for r in run_ids.split(",")]
    print("Run ids: ", run_ids)

    analysis_root_dir = (
        Path.home() / "scratch" / "len_gen" / "experiments" / "attention_analysis_data"
    )

    # Get the localhost name
    localhost = get_localhost()

    wandb_api = wandb.Api(
        overrides={
            "entity": "kzmnjd",
            "project": "len_gen",
        },
        timeout=200,
    )

    for run_id in run_ids:
        run = wandb_api.run(f"kzmnjd/len_gen/{run_id}")
        download_the_entire_run(localhost, run, root_dir=analysis_root_dir)
