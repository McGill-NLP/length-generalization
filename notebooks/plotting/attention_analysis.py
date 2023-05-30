import json
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List

import torch
from datasets import Dataset
from wandb.apis.public import Run

PATH_REPLACEMENTS = {
    "cedar": {
        "/home/kzmnjd/experiments/": "/home/kzmnjd/scratch/len_gen/experiments/",
    },
    "narval": {
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
        return "cc_cedar"
    elif "narval" in host:
        return "cc_narval"
    elif host.startswith("cn-"):
        return "mila"
    else:
        return host


def load_download_dir(download_dir: Path) -> Dict[str, Any]:
    output = {}
    for file in download_dir.glob("*"):
        name = file.name
        if name == "dataset.jsonl":
            ds = Dataset.from_json(file, lines=True, compression="gzip")
            output["dataset"] = ds
        elif name.endswith(".pt"):
            output[name] = torch.load(file)
        elif name.endswith(".pkl"):
            # Load the pickled object
            with open(file, "rb") as f:
                output[name] = pickle.load(f)
        else:
            raise ValueError(f"Unexpected file: {file}")

    return output


def download_and_load_attention_analysis(
    run: Run, category: str, root_dir: Path = None
) -> Dict[str, Any]:
    if root_dir is None:
        # Read from the environment variable
        root_dir = Path(os.environ["ATTENTION_ANALYSIS_ROOT"])

    root_dir.mkdir(parents=True, exist_ok=True)
    download_dir = root_dir / run.id / category

    output = {}
    if download_dir.exists():
        return load_download_dir(download_dir)

    download_dir.mkdir(parents=True, exist_ok=True)

    # Download the files
    # First, download the metadata file
    metadata = run.file(f"attn_metadata_{category}.json").download(
        root=download_dir, replace=True
    )
    metadata = json.load(metadata)

    host = run.host

    experiment_cluster = get_cluster_name(host)

    # Download the files
    for file_name, file_path in metadata.items():
        download_file(experiment_cluster, file_path, download_dir)

    return load_download_dir(download_dir)


def download_file(
    remote_cluster: str, remote_file_path: str, download_dir: Path
) -> Path:
    """Download a file from a remote host to the local machine."""
    # Check if the file exists in the same host
    localhost = os.uname()[1]

    local_cluster = get_cluster_name(localhost)
    remote_cluster = get_cluster_name(remote_cluster)

    if local_cluster == remote_cluster:
        true_remote_path = get_true_remote_path(remote_cluster, remote_file_path)
        # Copy the file to download_dir
        local_file_path = Path(true_remote_path)
        # Create a symbolic link to the source file in the destination directory
        destination_link = download_dir / local_file_path.name
        destination_link.symlink_to(local_file_path)
        return destination_link

    # Download the file from the remote host using SSH
    remote_file_path = get_true_remote_path(remote_cluster, remote_file_path)

    # Password less SSH, with host names already saved in ssh config
    ssh_url = f"{remote_cluster}"

    # Copy the file to the destination
    os.system(f"scp {ssh_url}:{remote_file_path} {download_dir}/")

    # Make sure the file is downloaded
    local_file_path = download_dir / Path(remote_file_path).name
    assert local_file_path.exists(), f"File {local_file_path} does not exist!"

    return local_file_path


def download_directory(
    remote_cluster: str, remote_dir: str, download_dir: Path
) -> Path:
    """Download a file from a remote host to the local machine."""
    # Check if the file exists in the same host
    localhost = os.uname()[1]

    local_cluster = get_cluster_name(localhost)
    remote_cluster = get_cluster_name(remote_cluster)

    print(f"Local cluster: {local_cluster}, remote cluster: {remote_cluster}")

    if local_cluster == remote_cluster:
        # Create a symbolic link to the source file in the destination directory
        local_dir = Path(get_true_remote_path(remote_cluster, remote_dir))
        for file in local_dir.glob("*"):
            destination_link = download_dir / file.name
            destination_link.symlink_to(file)

        return local_dir

    # Download the file from the remote host using SSH
    remote_dir = get_true_remote_path(remote_cluster, remote_dir)

    # Password less SSH, with host names already saved in ssh config
    ssh_url = f"{remote_cluster}"

    # Copy the file to the destination
    print(f"scp {ssh_url}:{remote_dir}/* {download_dir}/")
    os.system(f"scp {ssh_url}:{remote_dir}/* {download_dir}/")

    # Make sure the file is downloaded
    local_dir = download_dir
    assert len(list(local_dir.glob("*"))) > 1, f"Directory {local_dir} is empty!"

    return local_dir


def download_the_entire_run(localhost: str, run: Run, root_dir: Path = None) -> Path:
    if root_dir is None:
        # Read from the environment variable
        root_dir = Path(os.environ["ATTENTION_ANALYSIS_ROOT"])

    download_dir = root_dir / run.id
    download_dir.mkdir(parents=True, exist_ok=True)

    remote_host = get_cluster_name(run.host)

    metadata_lst: List[Dict[str, Any]] = []
    metadata_files = list(run.files())
    for file in metadata_files:
        if file.name.startswith("attn_metadata_"):
            category = file.name.split("_")[-1].split(".")[0]
            local_path = download_dir / category / "metadata.json"
            local_path.parent.mkdir(parents=True, exist_ok=True)
            if not local_path.exists():
                _ = file.download(root=local_path.parent, replace=True)
            metadata_lst.append(json.load(local_path.open()))

    # Download the files
    for metadata in metadata_lst:
        metadata_download_dir = download_dir / metadata["category"]
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
                for file in local_dir.glob("*"):
                    destination_link = metadata_download_dir / file.name
                    destination_link.symlink_to(file)
            else:
                # Download the remote directory content to metadata_download_dir using rsync
                cmd = (
                    f"rsync -avz {remote_host}:{remote_dir}/* {metadata_download_dir}/"
                )
                os.system(cmd)

                # Make sure the files are downloaded
                for file_name, file_path in metadata.items():
                    if file_name == "category":
                        continue
                    local_file_path = metadata_download_dir / Path(file_path).name
                    assert (
                        local_file_path.exists()
                    ), f"File {local_file_path} does not exist!"

    return download_dir
