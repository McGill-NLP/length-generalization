import argparse
import copy
import json
import os
import site
import tempfile
from pathlib import Path

from wandb.sdk.lib import filenames

site.addsitedir("src/")

from wandb import env as wandb_env


os.environ[wandb_env.SILENT] = "true"
os.environ[wandb_env.DISABLE_CODE] = "true"


LOAD_GPU_COUNTS_TO_VAR = """
source scripts/set_num_gpus.sh
"""

FAIL_IF_SWEEP_NOT_COMPLETE = """
python scripts/fail_if_sweep_not_complete.py
if [ $? -ne 0 ]; then
  echo "Python script failed with exit code $?"
  exit 1
fi
"""


def maybe_add_post_script(args) -> str:
    if args is not None and args.post_script is not None:
        post_script = args.post_script
        script = f"\nchmod a+x {post_script}\n"
        script += f"{post_script}\n"
        return script
    return ""


def use_torch_distributed(args: argparse.Namespace = None) -> bool:
    return (
        args is not None
        and "use_torch_distributed" in args
        and args.use_torch_distributed
    )


def maybe_set_master_ip_and_address(args: argparse.Namespace = None) -> str:
    if use_torch_distributed(args):
        return "\nsource scripts/set_master_ip_and_addr.sh\n"
    return ""


def command_to_bash_str(
    cmd: str, configs_str: str, prefix: str = "", args: argparse.Namespace = None
) -> str:
    cmd = cmd.strip()
    # Currently torch distributed is only supported for train and hp_step
    if use_torch_distributed(args) and any(
        [c in cmd for c in ["train", "hp_step", "predict"]]
    ):
        script = (
            f"{prefix}torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \\\n"
        )
        script += f'{prefix}\tsrc/main.py --configs "{configs_str}" \\\n'
        script += f"{prefix}\t\t{cmd}\n\n"
    else:
        script = f'{prefix}python src/main.py --configs "{configs_str}" \\\n'
        script += f"{prefix}\t{cmd}\n\n"

    return script


def make_run_script(
    configs: str,
    commands: str,
    env_vars: str,
    exp_key: str,
    exp_name: str,
    args=None,
) -> Path:
    script = "#!/bin/bash\n\n\n"

    if env_vars:
        for ev in env_vars.split(","):
            ev = ev.strip()
            script += f"export {ev}\n"

    script += f"export WANDB_RUN_ID={exp_key}\n"
    script += f"export APP_EXPERIMENT_NAME={exp_name}\n"
    script += f"export WANDB_TAGS=launched_by_{exp_key}\n"

    if use_torch_distributed(args):
        script += LOAD_GPU_COUNTS_TO_VAR

    script += maybe_set_master_ip_and_address(args)

    script = add_python_paths(script)

    configs_str = configs
    script += "\n\n"
    for c in commands.split(","):
        script += command_to_bash_str(c, configs_str, prefix="", args=args)

    script += maybe_add_post_script(args)

    script += 'echo "Experiment finished!"\n'

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "run.sh"
    with open(script_path, "w") as f:
        f.write(script)

    # subprocess.check_call(shlex.split(f"vim {script_path}"))

    return script_path


def add_python_paths(script):
    script += (
        "\n\nexport PYTHONPATH=$HOME/.local/lib/python3.6/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.7/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.8/site-packages/:$PYTHONPATH\n"
    )
    script += (
        "export PYTHONPATH=$HOME/.local/lib/python3.9/site-packages/:$PYTHONPATH\n"
    )
    script += "\n\n#pip install --user -r src/requirements.txt\n"
    return script


def make_run_script_seeds(
    configs: str,
    commands: str,
    env_vars: str,
    exp_key: str,
    exp_name: str,
    seeds: int,
    args=None,
) -> Path:
    script = "#!/bin/bash\n\n\n"

    if env_vars:
        for ev in env_vars.split(","):
            ev = ev.strip()
            script += f"export {ev}\n"

    # script += f"export WANDB_RUN_ID={exp_key}\n"
    script += f"export WANDB_RUN_GROUP={args.group}\n"
    script += f"export ORIG_APP_EXPERIMENT_NAME={exp_name}\n"
    script += f"export ORIG_WANDB_RUN_ID={exp_key}\n"
    script += f"export WANDB_TAGS=launched_by_{exp_key}\n"

    if use_torch_distributed(args):
        script += LOAD_GPU_COUNTS_TO_VAR

    script += maybe_set_master_ip_and_address(args)

    script = add_python_paths(script)
    script += "\n\n"

    configs_str = configs
    script += f"for SEED in `seq 1 {seeds}`; do\n"
    script += f"\texport APP_DIRECTORY=experiments/{exp_name}\n"
    script += f"\texport APP_EXPERIMENT_NAME=seed_$SEED\n"
    script += f"\texport APP_SEED=$SEED\n"
    script += f"\texport WANDB_JOB_TYPE=exp\n"
    script += f"\texport WANDB_RUN_ID={exp_key}_seed_$SEED\n\n"

    for c in commands.split(","):
        script += command_to_bash_str(c, configs_str, prefix="\t", args=args)

    script += "done\n"

    script += maybe_add_post_script(args)
    script += '\necho "Experiment finished!"\n'

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "run.sh"
    with open(script_path, "w") as f:
        f.write(script)

    # subprocess.check_call(shlex.split(f"vim {script_path}"))

    return script_path


def make_run_script_sweep_job(
    configs: str,
    commands: str,
    env_vars: str,
    exp_key: str,
    exp_name: str,
    seeds: int,
    args=None,
) -> Path:
    script = "#!/bin/bash\n\n\n"
    script += "\nexport WANDB_JOB_TYPE=hp_exp\n"
    script += f"export WANDB_TAGS=launched_by_{exp_key}\n\n\n"

    if use_torch_distributed(args):
        script += LOAD_GPU_COUNTS_TO_VAR

    script += maybe_set_master_ip_and_address(args)

    configs_str = configs
    for c in commands.split(","):
        script += command_to_bash_str(c, configs_str, prefix="", args=args)

    script += '\necho "Experiment finished!"\n'

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "job.sh"
    with open(script_path, "w") as f:
        f.write(script)

    # subprocess.check_call(shlex.split(f"vim {script_path}"))

    return script_path


def make_run_script_sweep_agent(
    configs: str,
    commands: str,
    env_vars: str,
    exp_key: str,
    exp_name: str,
    sweep_id: str,
    args=None,
) -> Path:
    script = "#!/bin/bash\n\n\n"

    if env_vars:
        for ev in env_vars.split(","):
            ev = ev.strip()
            script += f"export {ev}\n"

    script = add_python_paths(script)
    script += "\n\n"

    sweep_key = os.path.basename(sweep_id)

    script += f"\nexport WANDB_RUN_GROUP={args.group}\n"
    script += f"export WANDB_DIR=experiments/wandb_sweep_{sweep_key}\n"
    script += f"export WANDB_TAGS=launched_by_{exp_key}\n"
    script += f"export SWEEP_ID={sweep_id}\n"
    script += f"mkdir -p $WANDB_DIR\n"

    script += f"ln -srnf experiments/wandb_sweep_{sweep_key} experiments/{exp_name}/wandb_sweep_{sweep_key}\n"

    script += f"\nchmod a+x ./job.sh\n"
    script += f"wandb agent {sweep_id}\n"

    post_script = maybe_add_post_script(args)
    if post_script != "":
        script += FAIL_IF_SWEEP_NOT_COMPLETE
        script += post_script

    script += '\necho "Experiment finished!"\n'

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "run.sh"
    with open(script_path, "w") as f:
        f.write(script)

    return script_path


def make_run_script_sweep_manual_job(
    configs: str,
    commands: str,
    env_vars: str,
    exp_key: str,
    exp_name: str,
    seeds: int,
    args=None,
) -> Path:
    script = "#!/bin/bash\n\n\n"
    script += "\nexport WANDB_JOB_TYPE=hp_exp\n"
    script += "export WANDB_RUN_ID=$RUN_ID\n\n\n"
    script += f"CONFIGSTR='{configs}'\n\n"
    configs_str = "$CONFIGSTR"

    script += LOAD_GPU_COUNTS_TO_VAR

    script += 'if [ "$NUM_GPUS" -gt 1 ]; then\n'

    script += "\tsource scripts/set_master_ip_and_addr.sh\n"
    args_cp = copy.deepcopy(args)
    args_cp.use_torch_distributed = True
    for c in commands.split(","):
        script += command_to_bash_str(c, configs_str, prefix="\t", args=args_cp)

    script += f"else\n"

    args_cp = copy.deepcopy(args)
    args_cp.use_torch_distributed = False
    for c in commands.split(","):
        script += command_to_bash_str(c, configs_str, prefix="\t", args=args_cp)

    script += "fi\n"

    script += '\necho "Experiment finished!"\n'

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "job.sh"
    with open(script_path, "w") as f:
        f.write(script)

    # subprocess.check_call(shlex.split(f"vim {script_path}"))

    return script_path


def make_run_script_sweep_manual_agent(
    configs: str,
    commands: str,
    env_vars: str,
    exp_key: str,
    exp_name: str,
    sweep_id: str,
    args=None,
) -> Path:
    sweep_id = None

    script = "#!/bin/bash\n\n\n"

    if env_vars:
        for ev in env_vars.split(","):
            ev = ev.strip()
            script += f"export {ev}\n"

    script = add_python_paths(script)
    script += "\n\n"

    sweep_configs = args.sweep_configs
    assert sweep_configs is not None

    sweep_name = args.sweep_name
    if sweep_name is None:
        # Make Auto Sweep Name
        sweep_name = unique_experiment_name_from_filenames(sweep_configs)

    script += f"\nexport SWEEP_NAME={sweep_name}\n"
    script += f"export SWEEP_CONFIGS='{sweep_configs}'\n"
    script += f"export CAPTURE_LOG=1\n"
    script += f"export SWEEP_ROOT_DIR=experiments/$SWEEP_NAME\n"
    script += f"export HP_EXP_CONFIG='{configs}'\n"
    script += f"mkdir -p $SWEEP_ROOT_DIR\n"

    script += f"\nexport WANDB_RUN_GROUP={args.group}\n"

    if args.no_pe_run_ids is not None:
        # Load json file
        with open(args.no_pe_run_ids, "r") as f:
            no_pe_run_ids = json.load(f)

        # Convert the json to string and save it into a variable in the script
        script += f"export APP_NO_PE_RUN_IDS='{json.dumps(no_pe_run_ids)}'\n"

    tags = args.tags
    if tags is None:
        tags = ""
    else:
        tags = f",{tags}"
    script += f"export WANDB_TAGS=sweep,manual_sweep,launched_by_{exp_key}{tags}\n"

    script += f"\nchmod a+x scripts/manual_sweep_agent.sh\n"
    script += f"./scripts/manual_sweep_agent.sh\n\n"

    if args.post_script is not None:
        post_script = args.post_script
    else:
        post_script = "scripts/autoDist_manual_sweep_launch_best_run.sh"
    script += f"\nchmod a+x {post_script}\n"
    script += f"./{post_script}\n\n"

    script += '\necho "Experiment finished!"\n'

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "run.sh"
    with open(script_path, "w") as f:
        f.write(script)

    return script_path


def make_download_nope_scripts(
    configs: str,
    commands: str,
    env_vars: str,
    exp_key: str,
    exp_name: str,
    sweep_id: str,
    args=None,
) -> Path:
    script = "#!/bin/bash\n\n\n"

    if args.no_pe_run_ids is not None:
        # Load json file
        with open(args.no_pe_run_ids, "r") as f:
            no_pe_run_ids = json.load(f)

        run_ids = sorted(list(no_pe_run_ids.values()))
        run_ids = ",".join(run_ids)
        script += f"python scripts/download_nope_runs_to_local.py {run_ids}\n\n"

    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    script_path = tmp_dir / "pre_submit_script.sh"
    with open(script_path, "w") as f:
        f.write(script)

    return script_path


def make_metadata(exp_name, exp_key):
    tmp_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    tmp_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = tmp_dir / "metadata.json"

    metadata = {"exp_name": exp_name, "exp_key": exp_key}

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    return metadata_path


def unique_experiment_name_from_filenames(config_filenames):
    for p in config_filenames:
        assert os.path.exists(p), p

    configs = "_".join(
        [os.path.splitext(os.path.basename(p))[0] for p in config_filenames]
    )

    unique_name = f"{configs}"

    return unique_name


def get_exp_name(configs):
    filenames = list(map(lambda x: x.strip(), configs.split(",")))
    exp_name = unique_experiment_name_from_filenames(filenames)
    return exp_name


def add_source_code(art):
    root = "src/"
    root = os.path.abspath(root)
    exclude_fn = lambda path: path.endswith(".pyc") or path.endswith("__pycache__")
    for file_path in filenames.filtered_dir(root, lambda p: True, exclude_fn):
        save_name = os.path.relpath(file_path, root)
        art.add_file(file_path, name=f"src/{save_name}")


def main(args: argparse.Namespace):
    project: str = args.project
    entity: str = args.entity
    configs: str = args.configs

    exp_name = get_exp_name(configs)

    if args.dataset is not None:
        if not args.dataset.startswith("data-"):
            args.dataset = f"data-{args.dataset}"

        ds_name = args.dataset.replace("/", "_")
        exp_name += f"___{ds_name}"

    if args.name is not None:
        exp_name += args.name

    if args.output_only_name:
        print(exp_name)
        return

    if args.post_script:
        post_script_path = Path(args.post_script)
        if use_torch_distributed(args):
            distributed_post_script_path = (
                post_script_path.parent / f"distributed_{post_script_path.name}"
            )
            if distributed_post_script_path.exists():
                post_script_path = distributed_post_script_path
        args.post_script = str(post_script_path)

    print("# ----> 1. Generating a unique experiment name...")

    import wandb

    group = "general"
    job_type = "exp"
    if args.seeds is not None:
        job_type = "seed_launcher"
        group = f"SE-{exp_name}"
    elif args.sweep_id is not None:
        job_type = "agent"
        group = f"sweep-{os.path.basename(args.sweep_id)}"
    elif args.sweep_configs is not None:
        job_type = "agent"
        group = f"sweep-{exp_name}"

    if args.group is not None:
        group = args.group
    else:
        args.group = group

    if args.tags is not None:
        tags = args.tags.split(",")
    else:
        tags = None

    dir_dir = Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())
    dir_dir.mkdir(parents=True, exist_ok=True)
    settings = wandb.Settings()
    settings.update(
        disable_code=True,
        disable_git=True,
        silent=True,
        _save_requirements=False,
        _disable_meta=True,
    )
    run = wandb.init(
        project=project,
        entity=entity,
        dir=dir_dir,
        group=group,
        name=exp_name,
        config={},
        mode="online",
        force=True,
        save_code=False,
        settings=settings,
        job_type=job_type,
        id=args.idx,
        resume="allow",
        tags=tags,
    )

    run_id = run.id

    job_script_path = None
    pre_submit_script_path = None
    if args.seeds is not None:
        run_script_path = make_run_script_seeds(
            configs,
            args.commands,
            args.env_vars,
            run_id,
            exp_name,
            args.seeds,
            args=args,
        )
    elif args.sweep_id is not None:
        run_script_path = make_run_script_sweep_agent(
            configs,
            args.commands,
            args.env_vars,
            run_id,
            exp_name,
            args.sweep_id,
            args=args,
        )
        job_script_path = make_run_script_sweep_job(
            configs,
            args.commands,
            args.env_vars,
            run_id,
            exp_name,
            args.sweep_id,
            args=args,
        )
    elif args.sweep_configs is not None:
        exp_name = args.sweep_name
        run_script_path = make_run_script_sweep_manual_agent(
            configs,
            args.commands,
            args.env_vars,
            run_id,
            exp_name,
            args.sweep_id,
            args=args,
        )
        job_script_path = make_run_script_sweep_manual_job(
            configs,
            args.commands,
            args.env_vars,
            run_id,
            exp_name,
            args.sweep_id,
            args=args,
        )
        pre_submit_script_path = make_download_nope_scripts(
            configs,
            args.commands,
            args.env_vars,
            run_id,
            exp_name,
            args.sweep_id,
            args=args,
        )
    else:
        run_script_path = make_run_script(
            configs, args.commands, args.env_vars, run_id, exp_name, args=args
        )

    metadata_path = make_metadata(exp_name, run_id)

    artifact_name = f"bundle-{run_id}"
    artifact = wandb.Artifact(name=artifact_name, type="code")
    artifact.add_dir("configs", "configs/")
    add_source_code(artifact)
    artifact.add_dir("scripts", "scripts/")

    if os.path.exists(".run"):
        artifact.add_dir(".run", ".run/")
    if os.path.exists(".idea"):
        artifact.add_dir(".idea", ".idea/")
    if os.path.exists(".vscode"):
        artifact.add_dir(".vscode", ".vscode/")

    artifact.add_file(str(run_script_path), "run.sh")
    artifact.add_file(str(metadata_path), "metadata.json")
    if job_script_path is not None:
        artifact.add_file(str(job_script_path), "job.sh")
    if pre_submit_script_path is not None:
        artifact.add_file(str(pre_submit_script_path), "pre_submit_script.sh")

    if args.dataset is not None:
        artifact.metadata["data"] = args.dataset

    run.log_artifact(artifact)

    if args.dataset is not None:
        data_art_name = args.dataset
        if ":" not in data_art_name:
            data_art_name += ":latest"
        run.use_artifact(data_art_name)

    run.finish()

    print(f"\n\nExp name: {exp_name}")
    print(f"\n\nExp Key: {run_id}")
    print(f"Exp URL: {run.url}\n")


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
        "-s",
        "--configs",
        metavar="CONFIGS[,CONFIGS,CONFIGS]",
        type=str,
        help="Config file names",
    )

    parser.add_argument(
        "-c",
        "--commands",
        metavar="cmd -a -b[,cmd -c -d]",
        type=str,
        help="Experiment commands",
    )

    parser.add_argument(
        "-d", "--dataset", metavar="DATASET", type=str, help="Dataset name's bundle"
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
        "-e",
        "--env-vars",
        metavar="KEY=VAL[,KEY=VAL]",
        type=str,
        help="Experiment environment variables",
    )

    parser.add_argument(
        "--tags",
        metavar="VAL[,VAL]",
        type=str,
        help="Experiment tags",
    )

    parser.add_argument(
        "--seeds",
        metavar="NUM_SEEDS",
        type=int,
        help="Num of seeds",
    )

    parser.add_argument(
        "--sweep_id",
        metavar="SWEEP_ID",
        type=str,
        help="Wandb sweep id",
    )

    parser.add_argument(
        "-n",
        "--name",
        metavar="NAME",
        type=str,
        help="Name postfix",
    )

    parser.add_argument(
        "-i",
        "--idx",
        metavar="IDX",
        type=str,
        help="Experiment Idx",
    )

    parser.add_argument(
        "--group",
        metavar="GROUP",
        type=str,
        help="Wandb run group name",
    )

    parser.add_argument(
        "--post_script",
        metavar="POST_SCRIPT_PATH",
        type=str,
        help="Path to post script",
    )

    parser.add_argument(
        "--output_only_name",
        action="store_true",
        help="Print only the name of the experiment",
        default=False,
    )

    parser.add_argument(
        "--use_torch_distributed",
        action="store_true",
        help="Use torch.distributed.launch",
        default=False,
    )

    parser.add_argument(
        "--sweep_name", metavar="SWEEP_NAME", type=str, help="Sweep name"
    )

    parser.add_argument(
        "--sweep_configs", metavar="SWEEP_CONFIGS", type=str, help="Sweep configs"
    )

    parser.add_argument(
        "--no_pe_run_ids",
        metavar="JSON_PATH",
        type=str,
        help="Path to json file with no pe run ids",
    )

    args = parser.parse_args()

    main(args)
