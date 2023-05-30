import argparse
import os
import shlex
import site
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any

site.addsitedir("src/")

import _jsonnet
import yaml

from common.nest import flatten


def main(args: argparse.Namespace):
    project: str = args.project
    config_files = args.config_files
    config_files = [f.strip() for f in config_files.split(",")]
    print(config_files)

    jsonnet_str = "+".join([f'(import "{f}")' for f in config_files])
    json_str = _jsonnet.evaluate_snippet("snippet", jsonnet_str)
    config: Dict[str, Any] = json.loads(json_str)

    parameters = config.get("parameters", {})
    parameters = flatten(parameters, separator=".")
    for k, v in parameters.items():
        print(v)
        parameters[k] = json.loads(v)

    config["parameters"] = parameters

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = tmp_dir / "sweep.yaml"
    with yaml_path.open("w") as f:
        yaml.dump(config, f)

    print(yaml_path.open().read())
    print()

    if args.name is not None:
        sweep_name = args.name
    else:
        sweep_name = "_".join(
            [os.path.splitext(os.path.basename(p))[0] for p in config_files]
        )

    subprocess.check_call(
        shlex.split(f"wandb sweep -e {args.entity} -p {project} --name {sweep_name} {yaml_path}")
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make Experiment Bundle")

    if os.path.exists("configs/project_name.json"):
        with open("configs/project_name.json") as f:
            import json

            default_proj_name = json.load(f)["project_name"]
    else:
        default_proj_name = None

    if os.path.exists("configs/entity_name.json"):
        with open("configs/entity_name.json") as f:
            import json

            default_entity_name = json.load(f)["entity_name"]
    else:
        default_entity_name = None

    parser.add_argument(
        "config_files",
        metavar="CONFIG_YAML",
        nargs="?",
        type=str,
        help="Sweep config file",
    )

    parser.add_argument(
        "-p",
        "--project",
        metavar="project",
        type=str,
        default=default_proj_name,
        help="Wandb project",
    )

    parser.add_argument(
        "--entity",
        metavar="entity",
        type=str,
        default=default_entity_name,
        help="Wandb entity",
    )

    parser.add_argument(
        "--name",
        metavar="name",
        type=str,
        help="Wandb sweep name",
    )

    args = parser.parse_args()

    main(args)
