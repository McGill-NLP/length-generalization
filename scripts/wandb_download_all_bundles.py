import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import List

import fire
import jsonlines
import wandb
from tqdm import tqdm


def get_repo_dir() -> Path:
    return Path(__file__).parent.parent


def get_all_code_bundles_ids() -> List[str]:
    # Read the results
    results_filenames = [
        "scratchpad_f.jsonl",
        "sanity_check.jsonl",
        "classic.jsonl",
        "attention_aggr_analysis_final.jsonl",
        "scratch_for_attn.jsonl",
        "final_kl.jsonl",
    ]

    bundle_ids = set()
    # Read the results and add id to the set
    for filename in results_filenames:
        with jsonlines.open(get_repo_dir() / "results" / filename) as reader:
            for obj in reader:
                if obj["job_type"] != "agent":
                    continue

                bundle_ids.add(obj["id"])

    return sorted(list(bundle_ids))


def download_and_save_code_bundle(
    bundle_id: str, wandb_api: wandb.Api, download_root: Path
):
    (download_root / "bundles").mkdir(parents=True, exist_ok=True)
    zip_filename = download_root / "bundles" / f"{bundle_id}.zip"
    if zip_filename.exists():
        return

    save_dir = download_root / bundle_id
    save_dir.mkdir(parents=True, exist_ok=True)

    try:
        code_bundle = wandb_api.artifact(f"bundle-{bundle_id}:latest")
    except wandb.errors.CommError:
        print(f"Failed to download {bundle_id}. Skipping...")
        shutil.rmtree(save_dir)
        return
    code_bundle.download(root=save_dir)

    # Make sure that the code bundle is downloaded
    assert (save_dir / "src").exists()

    # Dump artifact metadata to a file
    metadata = code_bundle.metadata
    with open(save_dir / ".wandb_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(save_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, save_dir))

    # Delete the folder using shutil
    shutil.rmtree(save_dir)


def download_all_dataset_bundles(wandb_api: wandb.Api, download_root: Path):
    # Get all the dataset bundles. i.e. artifacts with name data-*
    dataset_collections = wandb_api.artifact_type("dataset").collections()
    for col in dataset_collections:
        if "clutrr" in col.name:
            continue

        artifact_id = col.name
        zip_filename = download_root / "datasets" / f"{artifact_id}.zip"
        if zip_filename.exists():
            continue

        save_dir = download_root / artifact_id
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            dataset_bundle = wandb_api.artifact(f"{artifact_id}:latest")
        except wandb.errors.CommError:
            print(f"Failed to download {artifact_id}. Skipping...")
            continue
        dataset_bundle.download(root=save_dir)

        # Make sure that the dataset bundle is downloaded
        assert len(list(save_dir.glob("*"))) > 0

        # Zip the folder
        with zipfile.ZipFile(zip_filename, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(save_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    zipf.write(file_path, os.path.relpath(file_path, save_dir))

        # Delete the folder using shutil
        shutil.rmtree(save_dir)


def main():
    repo_dir = get_repo_dir()

    wandb_project_name = json.load((repo_dir / "configs" / "project_name.json").open())[
        "project_name"
    ]
    wandb_entity_name = json.load((repo_dir / "configs" / "entity_name.json").open())[
        "entity_name"
    ]

    wandb_api = wandb.Api(
        overrides={
            "entity": wandb_entity_name,
            "project": wandb_project_name,
        },
        timeout=300,
    )

    bundle_ids = get_all_code_bundles_ids()

    for bundle_id in tqdm(bundle_ids):
        download_and_save_code_bundle(
            bundle_id=bundle_id,
            wandb_api=wandb_api,
            download_root=repo_dir / "experiments" / "wandb_export_root",
        )

    download_all_dataset_bundles(
        wandb_api=wandb_api,
        download_root=repo_dir / "experiments" / "wandb_export_root",
    )

    # Create a release zip
    shutil.make_archive(
        base_name=str(repo_dir / "experiments" / "wandb_export_root"),
        format="zip",
        root_dir=repo_dir / "experiments",
        base_dir="wandb_export_root",
    )


if __name__ == "__main__":
    fire.Fire(main)
