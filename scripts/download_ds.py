import argparse
import site
from pathlib import Path

site.addsitedir("src/")


def main(args: argparse.Namespace):
    from comet_ml import ExistingExperiment

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exp = ExistingExperiment(
        previous_experiment="71aca67973224fce952e5682bc733595",
        auto_output_logging=False,
        auto_metric_logging=False,
        auto_metric_step_rate=False,
        auto_log_co2=False,
        auto_param_logging=False,
        display_summary_level=0,
        workspace=args.workspace,
    )

    artifact_name = f"data-{args.dataset}"
    artifact = exp.get_artifact(artifact_name)
    artifact.download(str(output_dir), overwrite_strategy=True)

    exp.end()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make Experiment Bundle")

    parser.add_argument(
        "dataset",
        metavar="DATASET",
        type=str,
        help="Dataset name's bundle",
    )

    parser.add_argument(
        "-p",
        "--project",
        metavar="project",
        type=str,
        default="prompt-tune-comp-gen",
        help="CometML project",
    )

    parser.add_argument(
        "-o",
        "--output_dir",
        metavar="ROOT_DIR",
        type=str,
        default="data",
        help="Root directory",
    )

    parser.add_argument(
        "-w",
        "--workspace",
        metavar="KEY=VAL[,KEY=VAL]",
        type=str,
        default="kazemnejad",
        help="CometML workspace",
    )

    args = parser.parse_args()

    main(args)
