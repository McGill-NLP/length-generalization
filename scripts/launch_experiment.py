#!/usr/bin/env python3

import argparse
import json
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path
from shutil import which
from typing import List, Tuple

VENV_PATH = "~/RUN_EXP_PY_ENV"
SLURM_PYTHON_MODULE_NAME = "python/3"
SLURM_SINGULARITY_MODULE_NAME = "singularity"

PARTIAL_CREATE_AND_ACTIVATE_VENV = """
# command -v "module" >/dev/null && module load {python_module}
"""

REQUIREMENT_SCRIPT = """#!/bin/bash
{extra}
# python -c "import sys, pkgutil; sys.exit(0 if pkgutil.find_loader('codalab') else 1)" || pip3 install --user --quiet codalab
# python -c "import sys, pkgutil; sys.exit(0 if pkgutil.find_loader('tensorboard') else 1)" || pip3 install --user --quiet tensorboard
"""

NOTIFY_SCRIPT = """#!/bin/bash
arg=$(echo $1 | sed 's/ /%20/g')
curl -s -o /dev/null "https://maker.ifttt.com/trigger/{event_name}/with/key/{webhook_key}?value1=$arg"
"""

PARTIAL_HEADER = """#!/bin/bash
#SBATCH -o {log_dir}/{bundle_id}/compute.txt
{sbatch_account} 
"""

PARTIAL_DOWNLOAD_EXP_BUNDLE = """
cl work {codalab_worksheet}

echo "---> 2. Downloading the experiment bundle..."
mkdir -p {node_storage}/
cl download -o {node_storage}/home/ {bundle_id}
chmod a+x {node_storage}/home/run.sh
"""

PARTIAL_EXPORT_EXP_NAME = """
export EXP_NAME=$(python3 -c "import json; print(json.load(open('{node_storage}/home/metadata.json'))['exp_name'])")
"""

PARTIAL_PREPARE_EXP_DIR = """
mkdir -p {long_term_storage}/experiments/$EXP_NAME/
export EXP_DIR="{long_term_storage}/experiments/$EXP_NAME/"
ln -sfn {long_term_storage}/experiments/$EXP_NAME/ {long_term_storage}/experiments/{bundle_id} 
ln -sfn {long_term_storage}/experiments/$EXP_NAME/ {log_dir}/{bundle_id}/exp_dir
python3 -c "import json; json.dump({{'exp_bundle':'{bundle_id}'}}, open('{long_term_storage}/experiments/$EXP_NAME/cm_metadata.json', 'w'));"
sleep 5
"""

PARTIAL_COPY_ASSETS = """
mkdir -p {long_term_storage}/assets
rsync -avz {long_term_storage}/assets/{assets_name} {node_storage}/home/assets/
"""

PARTIAL_COPY_CREDENTIALS = """
echo "---> 3. Uploading credentials to container..."
rsync -avz $HOME/.config {node_storage}/home/
rsync -avz $HOME/.codalab {node_storage}/home/
rsync -avz $HOME/.comet.config {node_storage}/home/
rsync -avz $HOME/.netrc {node_storage}/home/
rsync -avz $HOME/.ssh {node_storage}/home/
"""

PARTIAL_RUN_TENSORBOARD_DEV = """
"""

PARTIAL_COPY_TMP_TO_NODE_STORAGE = """
echo "---> 4. Uploading contents to compute node..."
rsync -azP {tmp_dir}/* {node_storage}
"""

PARTIAL_CNTR_INIT_SINGULARITY = """
echo "---> 5. Copying container {image_path} to compute node..."
rsync -avzP {image_path} {node_storage}/
"""

PARTIAL_CNTR_EXEC_SINGULARITY = """
echo "---> 6. Running the computation..."
touch {long_term_storage}/experiments/{bundle_id}/stdout.txt
touch {long_term_storage}/experiments/{bundle_id}/stderr.txt

cd $HOME
command -v "module" >/dev/null && module load {singularity_module_name}
singularity exec --nv \\
        -H {node_storage}/home:$HOME \\
        -B {long_term_storage}/experiments:$HOME/experiments \\
        {node_storage}/{image} \\
        ./run.sh > {long_term_storage}/experiments/{bundle_id}/stdout.txt 2> {long_term_storage}/experiments/{bundle_id}/stderr.txt
"""

PARTIAL_CNTR_SHELL_SINGULARITY = """
echo "---> 6. Running the computation..."
cd $HOME
module load {singularity_module_name}
singularity shell --nv \\
        -H {node_storage}/home:$HOME \\
        -B {long_term_storage}/experiments:$HOME/experiments \\
        {node_storage}/{image}
"""

PARTIAL_ADD_TB_URL_TO_METADATA = """
"""

PARTIAL_UPLOAD_RESULT_TO_CL = """
"""

PARTIAL_CLEAN_UP = """
echo "---> 8. Cleaning up!"
rm -r {tmp_dir}
"""


def make_executable(script_path):
    mode = os.stat(str(script_path)).st_mode
    mode |= (mode & 0o444) >> 2
    os.chmod(str(script_path), mode)


def get_tempfile_path():
    return Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None


def install_requirements(args: argparse.Namespace):
    if args.install_notify:
        if is_tool("notify"):
            return
        if args.notify_webhook_key is None or args.notify_event_name is None:
            raise ValueError("notify-webhook-key or notify-event-name is not provided]")

        script_path = Path("notify")
        with open(script_path, "w") as f:
            f.write(
                NOTIFY_SCRIPT.format(
                    webhook_key=args.notify_webhook_key,
                    event_name=args.notify_event_name,
                )
            )

        make_executable(script_path)

        (Path.home() / ".local" / "bin").mkdir(parents=True, exist_ok=True)
        script_path.replace(Path.home() / ".local" / "bin" / "notify")
        print('Installation completed. You can use: $ notify "My Message"')
        exit(0)

    print("---> 1. Checking the requirements...")

    script = REQUIREMENT_SCRIPT.format(
        extra=PARTIAL_CREATE_AND_ACTIVATE_VENV.format(
            python_module=SLURM_PYTHON_MODULE_NAME
        )
    )
    script_path = f"{get_tempfile_path()}.sh"
    with open(script_path, "w") as f:
        f.write(script)

    make_executable(script_path)
    subprocess.check_call([script_path], shell=True)
    os.remove(script_path)


def run_on_slurm(args: argparse.Namespace):
    install_requirements(args)

    args.long_term_storage = replace_env_vars(
        str(Path(args.long_term_storage).expanduser())
    )
    args.log_dir = replace_env_vars(str(Path(args.log_dir).expanduser()))
    args.script_dir = replace_env_vars(str(Path(args.script_dir).expanduser()))

    (Path(args.log_dir) / args.bundle).mkdir(parents=True, exist_ok=True)
    Path(args.script_dir).mkdir(parents=True, exist_ok=True)

    tmp_node_storage = Path(args.long_term_storage) / next(
        tempfile._get_candidate_names()
    )
    tmp_node_storage.mkdir(parents=True, exist_ok=True)

    wrapper_script = "#!/bin/bash \n\n"
    wrapper_script += PARTIAL_CREATE_AND_ACTIVATE_VENV.format(
        python_module=SLURM_PYTHON_MODULE_NAME
    )

    download_bundle(args.bundle, output_dir=tmp_node_storage / "home")
    make_executable(tmp_node_storage / "home" / "run.sh")

    # wrapper_script += PARTIAL_DOWNLOAD_EXP_BUNDLE.format(
    #     codalab_worksheet=args.codalab_worksheet,
    #     bundle_id=args.bundle,
    #     node_storage=tmp_node_storage
    # )

    wrapper_script += PARTIAL_EXPORT_EXP_NAME.format(node_storage=tmp_node_storage)

    wrapper_script += PARTIAL_PREPARE_EXP_DIR.format(
        long_term_storage=args.long_term_storage,
        bundle_id=args.bundle,
        node_storage=tmp_node_storage,
        log_dir=str(Path(args.log_dir).expanduser()),
    )

    wrapper_script += PARTIAL_COPY_CREDENTIALS.format(node_storage=tmp_node_storage)

    if args.assets is not None and len(args.assets) > 0:
        wrapper_script += PARTIAL_COPY_ASSETS.format(
            long_term_storage=args.long_term_storage,
            node_storage=tmp_node_storage,
            assets_name=args.assets,
        )

    # if (not args.interactive or args.tb_on_interactive) and args.platform != "cc":
    #     wrapper_script += PARTIAL_RUN_TENSORBOARD_DEV

    compute_script = PARTIAL_HEADER.format(
        log_dir=str(Path(args.log_dir).expanduser()),
        sbatch_account=args.sbatch_account,
        bundle_id=args.bundle,
    )

    if args.env and args.env != "":
        env_vars = list(map(lambda x: x.strip(), args.env.split(",")))
        for ev in env_vars:
            if len(ev) == 0:
                continue
            compute_script += f"export {ev}\n"

    compute_script += "\nsleep 5 \n"

    compute_script += PARTIAL_COPY_TMP_TO_NODE_STORAGE.format(
        node_storage=args.node_storage, tmp_dir=tmp_node_storage
    )

    if args.image is None:
        args.image = "deepl-tf_v0.1.sif"
    compute_script += PARTIAL_CNTR_INIT_SINGULARITY.format(
        image_path=f"{args.images_dir}/{args.image}", node_storage=args.node_storage
    )

    wrapper_path = Path(args.script_dir).expanduser() / f"{args.bundle}_wrapper.sh"
    compute_path = Path(args.script_dir).expanduser() / f"{args.bundle}_compute.sh"

    if args.interactive:
        compute_script += PARTIAL_CNTR_SHELL_SINGULARITY.format(
            singularity_module_name=SLURM_SINGULARITY_MODULE_NAME,
            node_storage=args.node_storage,
            long_term_storage=args.long_term_storage,
            image=args.image,
        )

        wrapper_script += (
            'printf "\\n\\n--------------------------------------------------\\n"; \n'
        )
        wrapper_script += (
            'printf "Run the following command once the job is granted:\\n"; \n'
        )
        wrapper_script += f'echo "$ {compute_path}";\n'
        wrapper_script += (
            'echo "--------------------------------------------------"; \n'
        )

        account_str = f"--account={args.account}" if args.account else ""
        if args.platform == "ava":
            wrapper_script += f"srun --pty {args.slurm_args} {account_str} zsh\n"
        else:
            wrapper_script += f"salloc {args.slurm_args} {account_str} \n"
        wrapper_script += PARTIAL_CLEAN_UP.format(tmp_dir=tmp_node_storage)

        save_and_make_executable(wrapper_path, wrapper_script)
        save_and_make_executable(compute_path, compute_script)

        subprocess.check_call([wrapper_path])
    else:
        compute_script += PARTIAL_CNTR_EXEC_SINGULARITY.format(
            singularity_module_name=SLURM_SINGULARITY_MODULE_NAME,
            node_storage=args.node_storage,
            long_term_storage=args.long_term_storage,
            image=args.image,
            bundle_id=args.bundle,
        )

        wrapper_script += f"sbatch -W {args.slurm_args} {compute_path} \n"
        # wrapper_script += PARTIAL_ADD_TB_URL_TO_METADATA.format(
        #     tmp_dir=tmp_node_storage
        # )
        # wrapper_script += PARTIAL_UPLOAD_RESULT_TO_CL.format(
        #     long_term_storage=args.long_term_storage,
        #     bundle_id=args.bundle,
        # )
        wrapper_script += PARTIAL_CLEAN_UP.format(tmp_dir=tmp_node_storage)

        save_and_make_executable(wrapper_path, wrapper_script)
        save_and_make_executable(compute_path, compute_script)

        print("Started executing...")
        print("To check all logs, visit this directory:")
        print(f"$ cd {args.log_dir}/{args.bundle} && ls -lh")

        log_path = Path(args.log_dir).expanduser() / args.bundle / "runner.txt"
        log_file = open(log_path, "w")
        subprocess.Popen(
            [wrapper_path], start_new_session=True, stdout=log_file, stderr=log_file
        )


def save_and_make_executable(job_path, script):
    with open(job_path, "w") as f:
        f.write(script)
    make_executable(job_path)


def run_on_mila(args: argparse.Namespace):
    args.node_storage = "$SLURM_TMPDIR"
    # args.long_term_storage = "/network/projects/a/$USER"
    args.long_term_storage = "~/scratch"
    args.sbatch_account = ""
    run_on_slurm(args)


def run_on_cc(args: argparse.Namespace):
    args.node_storage = "$SLURM_TMPDIR"
    args.long_term_storage = "~/scratch"
    args.no_internet_on_compute = True
    if args.account is None:
        args.account = "rrg-bengioy-ad"
    args.sbatch_account = f"#SBATCH --account={args.account}"
    if "mem" not in args.slurm_args:
        args.slurm_args += " --mem=8G"
    run_on_slurm(args)


def run_on_ava(args: argparse.Namespace):
    args.node_storage = "/extra/ucinlp0/shivag5/comp-gen/comp-gen-nlp"
    args.long_term_storage = "/extra/ucinlp0/shivag5/comp-gen/comp-gen-nlp"
    # args.node_storage = "/home/shivag5/Documents/research/comp-gen/comp-gen-nlp/slurm_experiments"
    # args.long_term_storage = "/home/shivag5/Documents/research/comp-gen/comp-gen-nlp/slurm_experiments"
    args.sbatch_account = ""
    args.slurm_args = "-p ava_s.p " + args.slurm_args
    run_on_slurm(args)


def run_on_cl(args):
    exp_bundle = args.bundle
    dependencies = (
        f"data:{exp_bundle}/data "
        + f"configs:{exp_bundle}/configs "
        + f"scripts:{exp_bundle}/scripts "
        + f"src:{exp_bundle}/src run.sh:{exp_bundle}/run.sh "
        + f"metadata.json:{exp_bundle}/metadata.json "
        + f"cl_run.sh:cl_run"
    )

    docker_image = f"--request-docker-image {args.image}" if args.image else ""
    cmd = (
        f"cl run --tags result --allow-failed-dependencies --exclude-patterns experiments "
        + f'{docker_image} {args.cl_args} {dependencies} "sh cl_run.sh"'
    )

    out = subprocess.check_output(shlex.split(cmd))
    out = out.decode("utf8")
    print("Done!")
    print(f"Check here https://worksheets.codalab.org/bundles/{out}")


def run_on_google_colab(args):
    pass


def get_exp_metadata(exp: Path):
    metadata = {}
    file_path = exp / "metadata.json"
    if file_path.exists():
        with open(file_path) as f:
            metadata = json.load(f)

    file_path = exp / "cl_metadata.json"
    if file_path.exists():
        with open(file_path) as f:
            try:
                metadata["exp_bundle"] = json.load(f)["exp_bundle"]
            except:
                pass

    return metadata


def list_experiments(args):
    log_dir = Path(args.log_dir).expanduser()
    log_dir.mkdir(parents=True, exist_ok=True)

    experiments = filter(lambda x: x.name.startswith("0x"), Path(log_dir).iterdir())
    experiments = sorted(experiments, key=os.path.getmtime, reverse=True)

    for i, exp in enumerate(experiments[: args.n]):
        dir_name = exp.name

        output_str = f"### {i + 1}. {dir_name} "
        metadata = get_exp_metadata(exp / "exp_dir")
        if "exp_bundle" in metadata:
            if metadata["exp_bundle"] == dir_name:
                output_str += "(verified)"

        output_str += "\n"

        if "exp_name" in metadata:
            output_str += f"    name: `{metadata['exp_name']}`\n"

        if "tb_dev_url" in metadata:
            output_str += f"    tb_dev_url: `{metadata['tb_dev_url']}`\n"

        output_str += f"    see logs: $ cd {exp} && ls -lha\n"

        output_str += "\n"

        print(output_str)


def upload(args):
    install_requirements(args)

    script = "#!/bin/bash \n\n"
    script += PARTIAL_CREATE_AND_ACTIVATE_VENV.format(
        python_module=SLURM_PYTHON_MODULE_NAME
    )

    script += f"cl work {args.codalab_worksheet}\n"

    exps_dir = str(Path(args.long_term_storage).expanduser() / "experiments")

    import os

    for key, value in os.environ.items():
        exps_dir = exps_dir.replace(f"${key}", value)

    exps_dir = Path(exps_dir)
    assert exps_dir.exists()

    metadata = get_exp_metadata(exps_dir / args.bundle)
    assert "exp_name" in metadata

    print(f"Uploading {exps_dir / args.bundle}...\n")
    script += f"export EXP_NAME={metadata['exp_name']}\n"

    script += PARTIAL_UPLOAD_RESULT_TO_CL.format(
        long_term_storage=args.long_term_storage,
        bundle_id=args.bundle,
    )

    script_path = f"{get_tempfile_path()}.sh"
    save_and_make_executable(script_path, script)
    subprocess.check_call([script_path], shell=True)
    os.remove(script_path)


def upload_on_mila(args: argparse.Namespace):
    args.node_storage = "$SLURM_TMPDIR"
    # args.long_term_storage = "/network/projects/a/$USER"
    args.long_term_storage = "~/scratch/$USER"
    args.sbatch_account = ""
    upload(args)


def upload_on_cc(args: argparse.Namespace):
    args.node_storage = "$SLURM_TMPDIR"
    args.long_term_storage = "~/scratch"
    args.no_internet_on_compute = True
    if args.account is None:
        args.account = "rrg-bengioy-ad"
    args.sbatch_account = f"#SBATCH --account={args.account}"
    if "mem" not in args.slurm_args:
        args.slurm_args += " --mem=8G"
    upload(args)


def upload_on_ava(args: argparse.Namespace):
    # args.node_storage = "/srv/disk00/ucinlp/shivag5"
    # args.long_term_storage = "/srv/disk00/ucinlp/shivag5"
    args.node_storage = "/extra/ucinlp0/shivag5/comp-gen/comp-gen-nlp"
    args.long_term_storage = "/extra/ucinlp0/shivag5/comp-gen/comp-gen-nlp"
    args.sbatch_account = ""
    args.slurm_args = "-p ava_s.p " + args.slurm_args
    upload(args)


def replace_env_vars(target_str: str):
    for key, value in os.environ.items():
        target_str = target_str.replace(f"${key}", value)

    return target_str


def download_trained_model(args: argparse.Namespace):
    download_path = get_tempfile_path()

    script = "#!/bin/bash \n\n"
    script += PARTIAL_CREATE_AND_ACTIVATE_VENV.format(
        python_module=SLURM_PYTHON_MODULE_NAME
    )

    script += f"cl work {args.codalab_worksheet}\n"
    script += f"cl download -o {download_path} {args.bundle}\n"

    script += f'[ -d "{download_path}/selected_checkpoints" ] && cp -r {download_path}/selected_checkpoints {download_path}/checkpoints\n'

    script_path = f"{get_tempfile_path()}.sh"
    save_and_make_executable(script_path, script)
    subprocess.check_call([script_path], shell=True)
    os.remove(script_path)

    script = "#!/bin/bash \n\n"
    metadata = get_exp_metadata(download_path)

    target_path = str(
        Path(args.long_term_storage) / "experiments" / metadata["exp_name"]
    )
    for key, value in os.environ.items():
        target_path = target_path.replace(f"${key}", value)

    Path(target_path).expanduser().mkdir(exist_ok=True, parents=True)

    script += f"rsync -avz {download_path}/* {target_path}\n"

    script += (
        "ln -sfn {target_path} {long_term_storage}/experiments/{bundle_id}\n".format(
            target_path=target_path,
            long_term_storage=args.long_term_storage,
            bundle_id=metadata["exp_bundle"],
        )
    )

    script_path = f"{get_tempfile_path()}.sh"
    save_and_make_executable(script_path, script)
    subprocess.check_call([script_path], shell=True)
    os.remove(script_path)
    shutil.rmtree(download_path)

    print(f"Path to experiment dir: {target_path}")
    print(f"This result was created from experiment bundle `{metadata['exp_bundle']}`")


def download_trained_model_on_mila(args: argparse.Namespace):
    args.node_storage = "$SLURM_TMPDIR"
    args.long_term_storage = "/network/projects/a/$USER"
    args.sbatch_account = ""
    download_trained_model(args)


def download_trained_model_on_cc(args: argparse.Namespace):
    args.node_storage = "$SLURM_TMPDIR"
    args.long_term_storage = "~/scratch"
    args.no_internet_on_compute = True
    if args.account is None:
        args.account = "rrg-bengioy-ad"
    args.sbatch_account = f"#SBATCH --account={args.account}"
    if "mem" not in args.slurm_args:
        args.slurm_args += " --mem=8G"
    download_trained_model(args)


def download_trained_model_on_ava(args: argparse.Namespace):
    # args.node_storage = "/srv/disk00/ucinlp/shivag5"
    # args.long_term_storage = "/srv/disk00/ucinlp/shivag5"
    args.node_storage = "/extra/ucinlp0/shivag5/comp-gen/comp-gen-nlp"
    args.long_term_storage = "/extra/ucinlp0/shivag5/comp-gen/comp-gen-nlp"
    args.sbatch_account = ""
    args.slurm_args = "-p ava_s.p " + args.slurm_args
    download_trained_model(args)


def download_bundle2(exp_key: str, output_dir: Path):
    from comet_ml import ExistingExperiment

    output_dir.mkdir(parents=True, exist_ok=True)

    exp = ExistingExperiment(
        previous_experiment=args.bundle,
        auto_output_logging=False,
        auto_metric_logging=False,
        auto_metric_step_rate=False,
        auto_log_co2=False,
        auto_param_logging=False,
        display_summary_level=0,
    )

    artifact_name = f"artf-{exp_key}"
    artifact = exp.get_artifact(artifact_name)
    artifact.download(str(output_dir), overwrite_strategy=True)

    if len(artifact.remote_assets) > 0:
        from comet_ml.artifacts import LoggedArtifactAsset

        data_asset: LoggedArtifactAsset = [
            asset for asset in artifact.remote_assets if asset.logical_path == "data"
        ][0]
        data_artifact = exp.get_artifact(data_asset.link)
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_artifact.download(str(data_dir), overwrite_strategy=True)

    exp.end()


def download_bundle(exp_key: str, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    import wandb

    project = os.environ.get("WANDB_PROJECT", "comp-gen_v2")
    user = os.environ.get("WANDB_USER", None)
    api = wandb.Api(overrides={"project": project})
    if user is not None:
        artifact_name = f"{user}/{project}/"
    else:
        artifact_name = ""

    artifact_name += f"bundle-{exp_key}:latest"
    artifact = api.artifact(artifact_name)
    artifact.download(str(output_dir))

    if "data" in artifact.metadata:
        data_art_name = artifact.metadata["data"]
        if ":" not in data_art_name:
            data_art_name += ":latest"

        data_artifact = api.artifact(data_art_name)
        data_dir = output_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_artifact.download(str(data_dir))


def get_queued_jobs() -> List[Tuple[str, str, str]]:
    user = os.environ.get("USER")
    cmd = f"squeue -u {user} -o %A,%j,%T --noheader"
    output = subprocess.check_output(shlex.split(cmd)).decode("utf-8")
    jobs = []
    for line in output.splitlines():
        job_id, job_name, state = line.split(",")
        launcher_id = job_name.split("_compute.sh")[0]
        jobs.append((job_id, launcher_id, state))
    return jobs


def print_info(args: argparse.Namespace):
    jobs = get_queued_jobs()
    if len(jobs) == 0:
        print("No jobs in queue")
        return

    jobs.sort(key=lambda x: x[1])

    import wandb

    project = os.environ.get("WANDB_PROJECT", "comp-gen_v2")
    api = wandb.Api(overrides={"project": project})

    for i, (job_id, launcher_id, state) in enumerate(jobs):
        print("\n\n----------------------------------------")
        print(f"{i + 1}: Job {job_id}, LauncherID {launcher_id} is in state {state}")
        print("\tcd ~/sbatch_logs/{launcher_ids} && ls -lha")
        try:
            launcher_run = api.run(f"{project}/{launcher_id}")
        except Exception as e:
            print(f"Error: {e}")
            continue

        group = launcher_run.group
        if group is None:
            continue

        print("\tGroup: ", group)
        print("\tLink: ", launcher_run.url)
        is_sweep = launcher_run.job_type == "agent"

        runs = api.runs(
            f"{project}",
            {
                "$and": [
                    {"group": group},
                ],
            },
        )
        runs = list(runs)
        print(f"\tis_sweep: {is_sweep}, #runs: {len(runs)}")

        if is_sweep:
            sweep_runs = [run for run in runs if run.job_type == "hp_exp"]
            if len(sweep_runs) == 0:
                print("\tNo sweep runs")
                continue

            sweep = sweep_runs[0].sweep
            print("\tSweep URL: ", sweep.url)


def main(args):
    if args.info:
        print_info(args)
        return

    if args.bundle is not None:
        if "," not in args.bundle:
            bundles = [args.bundle]
        else:
            bundles = args.bundle.split(",")
            bundles = [b.strip() for b in bundles if b != ""]
    else:
        bundles = [None]

    queued_jobs = []
    if args.nodup:
        try:
            queued_jobs = get_queued_jobs()
            print(f"Queued jobs:")
            from pprint import pprint

            pprint(queued_jobs)
        except subprocess.CalledProcessError as e:
            print("Could not get queued jobs")
            print(e)

    already_launched = set([job[1] for job in queued_jobs])

    print("Bundles:", bundles)
    for bundle in bundles:
        if args.nodup and bundle in already_launched:
            print(f"Skipping {bundle} because it is already queued")
            continue

        args.bundle = bundle

        if args.list:
            list_experiments(args)
            exit(0)

        if args.download:
            if args.platform == "mila":
                download_trained_model_on_mila(args)
            elif args.platform == "cc":
                download_trained_model_on_cc(args)
            if args.platform == "ava":
                download_trained_model_on_ava(args)
            exit(0)

        if args.upload:
            if args.platform == "mila":
                upload_on_mila(args)
            elif args.platform == "cc":
                upload_on_cc(args)
            if args.platform == "ava":
                upload_on_ava(args)
            continue

        if args.platform == "mila":
            run_on_mila(args)
        elif args.platform == "cc":
            run_on_cc(args)
        if args.platform == "ava":
            run_on_ava(args)
        elif args.platform == "slurm":
            run_on_slurm(args)
        elif args.platform == "cl":
            run_on_cl(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment runner")

    parser.add_argument(
        "bundle", metavar="EXP_KEY", nargs="?", type=str, help="Experiment Key"
    )

    parser.add_argument(
        "-p",
        "--platform",
        metavar="PLATFORM",
        type=str,
        choices=["mila", "cc", "ava", "cl"],
        default="mila",
        help="The computation platform we're running the experiment",
    )

    parser.add_argument(
        "-s",
        "--slurm-args",
        metavar="ARGS",
        type=str,
        default="--gres=gpu:1",
        help="Slurm args",
    )

    parser.add_argument(
        "-c",
        "--cl-args",
        metavar="ARGS",
        type=str,
        default="--request-gpus 1",
        help="Codalab `run` args",
    )

    parser.add_argument(
        "-a",
        "--assets",
        metavar="ASSETS",
        type=str,
        help="Experiment assets that should be copied to container",
    )

    parser.add_argument(
        "-w",
        "--codalab-worksheet",
        metavar="WORKSHEET",
        type=str,
        default="kazemnejad-comp-gen",
        help="Which codalab worksheet to use",
    )

    parser.add_argument(
        "-i", "--image", metavar="IMAGE", type=str, help="Container Image"
    )

    parser.add_argument(
        "--images-dir",
        metavar="DIR",
        type=str,
        default="$HOME/scratch/containers",
        help="Container Images Directory (only needed for singularity)",
    )

    parser.add_argument(
        "--lt-storage",
        metavar="DIR",
        type=str,
        help="Path to platform's long-term storage",
    )

    parser.add_argument(
        "--node-storage",
        metavar="DIR",
        type=str,
        help="Platform's node storage (short-term)",
    )

    parser.add_argument(
        "--account",
        metavar="ACCOUNT",
        type=str,
        default=None,
        help="Slurm account (only needed for CC)",
    )

    parser.add_argument(
        "--script-dir",
        metavar="DIR",
        type=str,
        default="~/sbatch_jobs",
        help="Directory to output generated job scripts",
    )

    parser.add_argument(
        "--log-dir",
        metavar="DIR",
        type=str,
        default="~/sbatch_logs",
        help="Directory to store jobs' log",
    )

    parser.add_argument(
        "--env",
        metavar="ENVS",
        type=str,
        help="Environment variables passed to the container, e.g. X1=V1,x2=V2",
    )

    parser.add_argument(
        "--interactive", action="store_true", help="Run in the interactive mode"
    )

    parser.add_argument(
        "--tb-on-interactive",
        action="store_true",
        help="Launch TensorBoard in the interactive mode",
    )

    parser.add_argument(
        "--install-notify",
        action="store_true",
        help="Install notification program (only to use once)",
    )

    parser.add_argument("--notify-webhook-key", type=str, help="IFTTT webhook key")

    parser.add_argument("--notify-event-name", type=str, help="IFTTT event name")

    parser.add_argument(
        "--list", action="store_true", help="List all experiments on this platform"
    )

    parser.add_argument("-n", type=int, default=10, help="Number of items in the list")

    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload the experiment's results to Codalab",
    )

    parser.add_argument(
        "--download",
        action="store_true",
        help="Upload the experiment's results to Codalab",
    )

    parser.add_argument(
        "--nodup",
        action="store_true",
        help="Do not run already queued the experiment",
        default=False,
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Print queued experiments' info",
        default=False,
    )

    args = parser.parse_args()

    main(args)
