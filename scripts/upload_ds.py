import argparse

import wandb


# site.addsitedir("src/")


def main(args: argparse.Namespace):
    project: str = args.project
    if args.root is None:
        raise ValueError("Invalid root")

    run = wandb.init(
        id="4ar2g5xo9r6jargblyqnyefiwicbjn34",
        resume="must",
        job_type="db_upload",
        project=project,
        config={},
        mode="online",
        force=True,
        save_code=False,
        settings=wandb.Settings(disable_code=True, disable_git=True),
    )

    artifact_name = f"data-{args.dataset}"
    artifact = wandb.Artifact(name=artifact_name, type="dataset")
    artifact.add_dir(args.root)
    run.log_artifact(artifact)
    run.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make Experiment Bundle")

    parser.add_argument(
        "-d", "--dataset", metavar="DATASET", type=str, help="Dataset name's bundle"
    )

    parser.add_argument(
        "-p",
        "--project",
        metavar="project",
        type=str,
        default="comp_gen_v2",
        help="Wandb project",
    )

    parser.add_argument(
        "-r",
        "--root",
        metavar="ROOT_DIR",
        type=str,
        help="Root directory",
    )

    args = parser.parse_args()

    main(args)
