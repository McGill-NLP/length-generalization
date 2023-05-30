import json
import os
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple

import torch
from datasets import Dataset
from wandb.apis.public import Run

from tqdm.auto import tqdm

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


def load_download_dir(download_dir: Path) -> Dict[str, Any]:
    output = {}
    for file in download_dir.glob("*"):
        name = file.name
        if name == "dataset.jsonl":
            ds = Dataset.from_json(str(file))
            output["dataset"] = ds
        elif name.endswith(".pt"):
            output[name.replace(".pt", "")] = torch.load(file)
        elif name.endswith(".pkl"):
            # Load the pickled object
            with open(file, "rb") as f:
                output[name.replace(".pkl", "")] = pickle.load(f)
        elif name.startswith("attn_metadata"):
            continue
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

    print(list(download_dir.glob("*")))
    output = {}
    if download_dir.exists() and len(list(download_dir.glob("*"))) > 1:
        return load_download_dir(download_dir)

    download_dir.mkdir(parents=True, exist_ok=True)

    # Download the files
    # First, download the metadata file
    metadata = run.file(f"attn_metadata_{category}.json").download(
        root=download_dir, replace=True
    )
    metadata = json.load(metadata)

    assert (
        metadata["category"] == category
    ), f"Category mismatch: {metadata['category']} != {category}"

    host = run.host

    experiment_cluster = get_cluster_name(host)

    remote_dir = Path(metadata["dataset"]).parent
    download_remote_dir(experiment_cluster, str(remote_dir), download_dir)

    # # Download the files
    # for file_name, file_path in metadata.items():
    #     if file_name == "category":
    #         continue
    #     download_file(experiment_cluster, file_path, download_dir)

    return load_download_dir(download_dir)


def download_file(
    remote_cluster: str, remote_file_path: str, download_dir: Path
) -> Path:
    """Download a file from a remote host to the local machine."""
    # Check if the file exists in the same host
    localhost = os.uname()[1]

    local_cluster = get_cluster_name(localhost)
    remote_cluster = get_cluster_name(remote_cluster)

    print(f"Local cluster: {local_cluster}, remote cluster: {remote_cluster}")

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
    print(f"scp {ssh_url}:{remote_file_path} {download_dir}/")
    os.system(f"scp {ssh_url}:{remote_file_path} {download_dir}/")

    # Make sure the file is downloaded
    local_file_path = download_dir / Path(remote_file_path).name
    assert local_file_path.exists(), f"File {local_file_path} does not exist!"

    return local_file_path


def download_remote_dir(
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

    metadata_lst = list_metadata(download_dir, run)

    # Download the files
    for metadata in tqdm(metadata_lst, desc="Downloading data"):
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
                    if destination_link.exists():
                        continue
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


def download_the_entire_run2(localhost: str, run: Run, root_dir: Path = None) -> Path:
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
            else:
                remote_prefix = PATH_REPLACEMENTS[remote_host]
                local_prefix = PATH_REPLACEMENTS[localhost]
                local_dir = Path(remote_dir.replace(remote_prefix, local_prefix))

            for file in local_dir.glob("*"):
                destination_link = metadata_download_dir / file.name
                if destination_link.exists():
                    continue
                destination_link.symlink_to(file)

    return download_dir


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


def rsync_download_remote_dir(
    remote_cluster: str, local_destination: Path, remote_dir: str, file_list: List[str]
):
    """Download a list of file from remote directory to local destination using rsync."""
    # Check if the file exists in the same host
    localhost = os.uname()[1]

    local_cluster = get_cluster_name(localhost)
    remote_cluster = get_cluster_name(remote_cluster)

    print(f"Local cluster: {local_cluster}, remote cluster: {remote_cluster}")

    if local_cluster == remote_cluster:
        # Create a symbolic link to the source file in the destination directory
        local_dir = Path(get_true_remote_path(remote_cluster, remote_dir))
        for file in local_dir.glob("*"):
            destination_link = local_destination / file.name
            destination_link.symlink_to(file)

        return local_dir

    # Download the file from the remote host using SSH
    remote_dir = get_true_remote_path(remote_cluster, remote_dir)

    # Password less SSH, with host names already saved in ssh config
    ssh_url = f"{remote_cluster}"

    # Copy the file to the destination
    # Only include the files in the file list
    # Example: rsync -av --include 'file1.txt' --include 'file2.txt' --exclude '*' /path/to/source/ /path/to/destination/
    include_files = " ".join([f"--include '{f}'" for f in file_list])
    exclude_files = "--exclude '*'"
    print(
        f"rsync -e ssh -avz {include_files} {exclude_files} {ssh_url}:{remote_dir}/* {local_destination}/"
    )
    os.system(
        f"rsync -e ssh -avz {include_files} {exclude_files} {ssh_url}:{remote_dir}/* {local_destination}/"
    )

    # Make sure the file is downloaded
    local_dir = local_destination
    assert len(list(local_dir.glob("*"))) > 1, f"Directory {local_dir} is empty!"

    return local_dir


def download_dataset_for_all_categories(
    attention_analysis_run: Run, root_dir: Path
) -> Tuple[Dict[str, Dataset], Dict[str, Any]]:
    """Download the dataset for all categories."""
    # List all metadata files
    dataset_name = attention_analysis_run.config["dataset"]["name"]
    dataset_split = attention_analysis_run.config["dataset"]["split"]

    dataset_dir = root_dir / "datasets" / dataset_name / dataset_split
    dataset_dir.mkdir(parents=True, exist_ok=True)

    metadata_lst = list_metadata(root_dir, attention_analysis_run)

    import wandb

    all_datasets = {}
    all_tokenization_info = {}
    for m in metadata_lst:
        category = m["category"]
        dataset_path = m["dataset"]
        tokenization_info_path = m["tokenization_info"]

        local_dir = dataset_dir / category
        local_dir.mkdir(parents=True, exist_ok=True)

        files_to_download = []
        if not (local_dir / Path(dataset_path).name).exists():
            files_to_download.append(Path(dataset_path).name)

        if not (local_dir / Path(tokenization_info_path).name).exists():
            files_to_download.append(Path(tokenization_info_path).name)

        if len(files_to_download) > 0:
            rsync_download_remote_dir(
                attention_analysis_run.host,
                local_dir,
                str(Path(dataset_path).parent),
                files_to_download,
            )

        # Load the dataset
        if dataset_path.endswith(".json"):
            dataset = Dataset.from_json(local_dir / Path(dataset_path).name)
        elif dataset_path.endswith(".parquet"):
            dataset = Dataset.from_parquet(local_dir / Path(dataset_path).name)
        else:
            raise ValueError(f"Dataset format {dataset_path} not supported.")

        # Load the tokenization info (pickle)
        tok_info = pickle.load(
            (local_dir / Path(tokenization_info_path).name).open("rb")
        )

        all_datasets[category] = dataset
        all_tokenization_info[category] = tok_info

    return all_datasets, all_tokenization_info
