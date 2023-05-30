#!/usr/bin/env python3

import argparse
import copy
import hashlib
import json
import os
import random
import shlex
import subprocess
import tempfile
from pathlib import Path
from shutil import which
from typing import Dict, Union, Any, List, Tuple


def create_md5_hash(inp: str):
    # Create MD5 hash object
    md5 = hashlib.md5()
    # Update the hash with the string
    md5.update(inp.encode("utf-8"))
    # Get the hexadecimal representation of the hash
    return md5.hexdigest()


def make_executable(script_path):
    mode = os.stat(str(script_path)).st_mode
    mode |= (mode & 0o444) >> 2
    os.chmod(str(script_path), mode)


def get_tempfile_path():
    return Path(tempfile.gettempdir()) / next(tempfile._get_candidate_names())


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""
    return which(name) is not None


def save_and_make_executable(job_path, script):
    with open(job_path, "w") as f:
        f.write(script)
    make_executable(job_path)


def replace_env_vars(target_str: str):
    for key, value in os.environ.items():
        target_str = target_str.replace(f"${key}", value)

    return target_str


class ComputingCluster:
    def __init__(self, **kwargs):
        pass

    def setup_cluster(self) -> None:
        pass

    def prepare_job(self, output_dir: Path) -> str:
        pass

    def create_launch_script(self, job_body) -> Path:
        pass

    def execute_job(self, job_body):
        pass


class SlurmComputingCluster(ComputingCluster):
    def __init__(
        self,
        launcher_id: str,
        slurm_args: str,
        project_name: str = "pt_hf_base",
        images_dir: str = "containers",
        image_name: str = "latest.sif",
        logs_dir: str = "~/sbatch_logs/",
        scripts_dir: str = "~/sbatch_jobs/",
        shared_storage_dir: str = "$SCRATCH",
        compute_storage_dir: str = "$SLURM_TMPDIR",
        github_token: str = None,
        interactive: bool = False,
        wait_for_login_script: bool = False,
        wandb_offline: bool = False,
        transformers_offline: bool = False,
        hf_datasets_offline: bool = False,
        account: str = None,
        config: Dict[str, str] = None,
        env_vars: List[str] = None,
        dry_run: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if config is None:
            config = {}

        self.launcher_id = launcher_id

        self.global_logs_dir = Path(replace_env_vars(logs_dir)).expanduser()
        self.global_scripts_dir = Path(replace_env_vars(scripts_dir)).expanduser()
        self.cluster_shared_storage_dir = Path(
            replace_env_vars(shared_storage_dir)
        ).expanduser()
        self.compute_node_storage_dir = compute_storage_dir

        self.singularity_image_library_path = (
            self.cluster_shared_storage_dir / images_dir
        )

        self.log_dir = Path(self.global_logs_dir) / f"lid_{launcher_id}"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.script_dir = Path(self.global_scripts_dir)
        self.script_dir.mkdir(parents=True, exist_ok=True)

        self.image_name = image_name
        self.slurm_args = slurm_args

        self.github_token = github_token
        self.wait_for_login_script = wait_for_login_script
        self.interactive = interactive

        self.project_name = config.get("wandb_project_name", project_name)

        self.run_script_name = "worker_job.sh"

        self.wandb_offline = wandb_offline
        self.transformers_offline = transformers_offline
        self.hf_datasets_offline = hf_datasets_offline
        self.account = account
        self.env_vars = env_vars

        self.experiments_dir = (
            self.cluster_shared_storage_dir / self.project_name / "experiments"
        )
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

        self.wandb_api_key = config.get("wandb_api_key", None)
        self.wandb_project_name = config.get("wandb_project_name", None)
        self.wandb_entity_name = config.get("wandb_entity_name", None)

        self.dry_run = dry_run

    def prepare_job(self, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)

        import wandb

        if self.wandb_project_name is None:
            project = os.environ.get("WANDB_PROJECT", self.project_name)
        else:
            project = self.wandb_project_name

        if self.wandb_entity_name is None:
            entity = os.environ.get("WANDB_ENTITY", None)
        else:
            entity = self.wandb_entity_name

        api = wandb.Api(
            overrides={"project": project, "entity": entity}, api_key=self.wandb_api_key
        )
        if entity is not None:
            artifact_name = f"{entity}/{project}/"
        else:
            artifact_name = ""

        artifact_name += f"bundle-{self.launcher_id}:latest"
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

        try:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            persistent_key = metadata["exp_name"]
        except Exception as e:
            print(
                "Unable to load metadata.json, computing "
                "persistent_dir based on launcher_id"
            )
            persistent_key = create_md5_hash(self.launcher_id)

        worker_script = f"#!/bin/bash\n\n"
        # worker_script += "if test -v WANDB_CACHE_DIR; then\n"
        # worker_script += f"\tln -sfn experiments/wandb_cache_dir $WANDB_CACHE_DIR\n"
        # worker_script += "fi\n\n"

        # worker_script += "export WANDB_DIR=wandb_dir\n"
        # worker_script += "mkdir -p $WANDB_DIR\n\n"

        worker_script += "chmod a+x run.sh\n"
        worker_script += "./run.sh\n\n"

        # worker_script += f"mkdir -p experiments/{persistent_key}/\n"
        # worker_script += f"cp -r wandb_dir/* experiments/{persistent_key}/\n\n"

        save_and_make_executable(output_dir / self.run_script_name, worker_script)

        return persistent_key

    def execute_job(self, job_body):
        login_script_path, compute_script_path = self.create_launch_script(job_body)

        if self.dry_run:
            print("Dry run, not executing job")
            print(f"Login script: {login_script_path}")
            print(f"Compute script: {compute_script_path}")
            return

        if self.interactive:
            try:
                subprocess.check_call([login_script_path])
            except subprocess.CalledProcessError as e:
                print(e)
                print("Exiting...")
        else:
            print("Started executing...")
            print("To check all logs, visit this directory:")
            print(f"$ cd {self.log_dir} && ls -lh")

            log_path = self.log_dir / "launcher.txt"
            log_file = open(log_path, "w")
            p = subprocess.Popen(
                [login_script_path],
                start_new_session=True,
                stdout=log_file,
                stderr=log_file,
            )

            if self.wait_for_login_script:
                p.wait()

    def create_launch_script(self, job_body) -> Tuple[Path, Path]:
        tmp_exp_dir = (
            self.cluster_shared_storage_dir
            / "job_launcher_files"
            / f"{self.launcher_id}"
        )
        tmp_exp_dir.mkdir(parents=True, exist_ok=True)

        persistent_key = self.prepare_job(tmp_exp_dir / "home")

        compute_script = self.create_compute_script(tmp_exp_dir, persistent_key)
        compute_script_path = self.script_dir / f"{self.launcher_id}_compute.sh"
        save_and_make_executable(compute_script_path, compute_script)

        login_script = self._create_pre_sbatch_launch_script(
            tmp_exp_dir, persistent_key
        )
        login_script += self._create_sbatch_launch_script(
            compute_script_path, persistent_key
        )
        login_script += self._create_post_sbatch_launch_script(
            tmp_exp_dir, persistent_key
        )
        login_script += self._create_notify_script(job_body, persistent_key)

        login_script_path = self.script_dir / f"{self.launcher_id}_login.sh"
        save_and_make_executable(login_script_path, login_script)

        return login_script_path, compute_script_path

    def _create_pre_sbatch_launch_script(
        self, tmp_exp_dir: Path, persistent_key: str
    ) -> str:
        script = "#!/bin/bash \n\n"

        job_persistent_dir = f"{self.experiments_dir}/{persistent_key}/"
        script += f"mkdir -p {job_persistent_dir}\n"
        script += f"ln -sfn {job_persistent_dir} {self.log_dir}/exp_dir\n"
        script += f"ln -sfn {job_persistent_dir} {self.experiments_dir}/lid_{self.launcher_id}\n"
        script += "sleep 5\n\n"

        script += f'echo "Copying credentials to container..."\n'
        script += f"rsync -avz $HOME/.config {tmp_exp_dir}/home/\n"
        script += f"rsync -avz $HOME/.aws {tmp_exp_dir}/home/\n"
        script += f"rsync -avz $HOME/.codalab {tmp_exp_dir}/home/\n"
        script += f"rsync -avz $HOME/.comet.config {tmp_exp_dir}/home/\n"
        script += f"rsync -avz $HOME/.netrc {tmp_exp_dir}/home/\n"
        script += f"rsync -avz $HOME/.ssh {tmp_exp_dir}/home/\n"

        # Check if the job has `pre_submit_script.sh`
        pre_submit_script_path = tmp_exp_dir / "home" / "pre_submit_script.sh"
        if pre_submit_script_path.exists():
            script += f'echo "Running pre_submit_script.sh..."\n'
            script += f"echo '{pre_submit_script_path}'\n"
            script += f"chmod a+x {pre_submit_script_path}\n"
            script += f"(cd {tmp_exp_dir}/home && ./{pre_submit_script_path.name})\n"

        return script

    def _create_sbatch_launch_script(
        self, compute_script_path: Path, persistent_key: str
    ) -> str:
        script = f"cd {self.cluster_shared_storage_dir}\n"
        if not self.interactive:
            script += 'echo "Submitting the job..." \n'
            script += f"sbatch --wait {self.slurm_args} {compute_script_path} \n"
        else:
            script += 'printf "\\n\\n--------------------------------------------------\\n"; \n'
            script += (
                'printf "Run the following command once the job is granted:\\n"; \n'
            )
            script += f'echo "$ {compute_script_path}";\n'
            script += 'echo "--------------------------------------------------"; \n'
            account_str = f"--account={self.account}" if self.account else ""
            script += f"salloc {self.slurm_args} {account_str}\n"
        return script

    def _create_post_sbatch_launch_script(
        self, tmp_exp_dir: Path, persistent_key: str
    ) -> str:
        # script = f'echo "Cleaning up..."\n'
        # script += f"rm -rf {tmp_exp_dir}\n"
        script = ""
        return script

    def _create_notify_script(
        self, job_body: Dict[str, Any], persistent_key: str
    ) -> str:
        return ""

    def create_compute_script(self, tmp_exp_dir: Path, persistent_key: str) -> str:
        script = "#!/bin/bash\n"
        script += f"#SBATCH -o {self.log_dir}/compute_log.txt\n"
        if self.account is not None:
            script += f"#SBATCH --account={self.account}\n"

        script += "\nsleep 5 \n"

        script += f'echo "Uploading contents to compute node..." \n'
        script += f"rsync -azP {tmp_exp_dir}/* {self.compute_node_storage_dir} \n\n"

        image_path = self.singularity_image_library_path / self.image_name
        script += f'echo "Copying container {image_path} to compute node..." \n'
        script += f"rsync -avzP {image_path} {self.compute_node_storage_dir}/ \n\n"

        stdout_path = f"{self.experiments_dir}/{persistent_key}/stdout.txt"
        stderr_path = f"{self.experiments_dir}/{persistent_key}/stderr.txt"
        script += f"touch {stdout_path} \n"
        script += f"touch {stderr_path} \n\n"

        script += "export TRANSFORMERS_CACHE=~/experiments/hf_cache\n"
        script += "export HF_DATASETS_CACHE=~/experiments/hf_ds_cache\n"
        script += "export HF_MODULES_CACHE=~/experiments/hf_modules_cache\n"
        script += f"export WANDB_CACHE_DIR={self.experiments_dir}/wandb_cache_dir\n"
        script += f"export WANDB_DIR={self.experiments_dir}/{persistent_key}\n\n"

        if self.wandb_offline:
            script += "export WANDB_MODE=offline\n"

        if self.transformers_offline:
            script += "export TRANSFORMERS_OFFLINE=1\n"

        if self.hf_datasets_offline:
            script += "export HF_DATASETS_OFFLINE=1\n"

        if self.env_vars is not None:
            for k_v in self.env_vars:
                script += f"export {k_v}\n"

        script += f'\necho "Running the computation..." \n'
        script += "cd $HOME\n"
        script += 'command -v "module" >/dev/null && module load singularity\n'

        if not self.interactive:
            script += "singularity exec --nv \\\n"
            script += f"\t-H {self.compute_node_storage_dir}/home:$HOME \\\n"
            script += f"\t-B {self.experiments_dir}:$HOME/experiments \\\n"
            script += f"\t-B {self.cluster_shared_storage_dir}:$HOME/{self.cluster_shared_storage_dir.name} \\\n"
            script += f"\t{self.compute_node_storage_dir}/{self.image_name} \\\n"
            script += (
                f"\t./{self.run_script_name} > {stdout_path} 2> {stderr_path} \n\n"
            )
        else:
            script += "singularity shell --nv \\\n"
            script += f"\t-H {self.compute_node_storage_dir}/home:$HOME \\\n"
            script += f"\t-B {self.experiments_dir}:$HOME/experiments \\\n"
            script += f"\t-B {self.cluster_shared_storage_dir}:$HOME/{self.cluster_shared_storage_dir.name} \\\n"
            script += f"\t{self.compute_node_storage_dir}/{self.image_name} \n\n"

        return script


class ComputeCanadaCluster(SlurmComputingCluster):
    def __init__(
        self,
        **kwargs,
    ):
        account = kwargs.pop("account", "rrg-bengioy-ad")
        wandb_offline = kwargs.pop("wandb_offline", True)
        transformers_offline = kwargs.pop("transformers_offline", True)
        hf_datasets_offline = kwargs.pop("hf_datasets_offline", True)
        super().__init__(
            **kwargs,
            shared_storage_dir="~/scratch",
            account=account,
            wandb_offline=wandb_offline,
            transformers_offline=transformers_offline,
            hf_datasets_offline=hf_datasets_offline,
        )

    def prepare_job(self, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)

        import wandb

        project = os.environ.get("WANDB_PROJECT", self.project_name)
        user = os.environ.get("WANDB_USER", None)
        api = wandb.Api(overrides={"project": project})
        if user is not None:
            artifact_name = f"{user}/{project}/"
        else:
            artifact_name = ""

        artifact_name += f"bundle-{self.launcher_id}:latest"
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

        try:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            persistent_key = metadata["exp_name"]
        except Exception as e:
            print(
                "Unable to load metadata.json, computing "
                "persistent_dir based on launcher_id"
            )
            persistent_key = create_md5_hash(self.launcher_id)

        worker_script = f"#!/bin/bash\n\n"
        # worker_script += "if test -v WANDB_CACHE_DIR; then\n"
        # worker_script += f"\tln -sfn experiments/wandb_cache_dir $WANDB_CACHE_DIR\n"
        # worker_script += "fi\n\n"

        # worker_script += "export WANDB_DIR=wandb_dir\n"
        # worker_script += "mkdir -p $WANDB_DIR\n\n"
        #
        # worker_script += "chmod a+x scripts/sync_wandb_logs.sh\n"
        # worker_script += (
        #     f"./scripts/sync_wandb_logs.sh wandb_dir experiments/{persistent_key} &\n"
        # )

        worker_script += "chmod a+x run.sh\n"
        worker_script += "./run.sh\n\n"

        # worker_script += f"mkdir -p experiments/{persistent_key}/\n"
        # worker_script += f"rm -r experiments/{persistent_key}/wandb || true\n"
        # worker_script += f"cp -r wandb_dir/* experiments/{persistent_key}/\n\n"

        save_and_make_executable(output_dir / self.run_script_name, worker_script)

        return persistent_key

    def _create_post_sbatch_launch_script(
        self, tmp_exp_dir: Path, persistent_key: str
    ) -> str:
        script = super()._create_post_sbatch_launch_script(tmp_exp_dir, persistent_key)
        if self.interactive:
            return script

        script += "\n\n"
        # script += 'if [[ -e "~/.wandb_cache_dir" ]] ; then\n'
        # script += "\tmoved_wandb_cache=1\n"
        # script += "\tmv ~/.wandb_cache_dir ~/.wandb_cache_dir.back\n"
        # script += "fi\n\n"

        # script += 'if [[ -e "~/experiments" ]] ; then\n'
        # script += "\tmoved_experiment=1\n"
        # script += "\tmv ~/experiments ~/experiments.back\n"
        # script += "fi\n\n"

        # script += f"ln -sfn {self.experiments_dir}/wandb_cache_dir ~/.wandb_cache_dir\n"
        script += f"ln -sfn {self.experiments_dir} ~/experiments || true\n"
        script += f"export WANDB_CACHE_DIR={self.experiments_dir}/wandb_cache_dir\n"
        script += (
            f"find {self.experiments_dir}/{persistent_key}/wandb/ "
            f'-maxdepth 1 -type d -name "offline*" '
            f"-exec wandb sync {{}} \; "
            f"-exec sleep 1s \;\n\n"
        )

        # script += (
        #     f"find {self.experiments_dir}/{persistent_key}/ "
        #     f'-name "*.bin" -type f -delete\n'
        # )
        #
        # script += (
        #     f"find {self.experiments_dir}/{persistent_key}/ "
        #     f'-name "*.pt" -type f -delete\n\n'
        # )
        # script += "rm ~/.wandb_cache_dir\n"
        # script += "rm ~/experiments\n\n"
        #
        # script += "if test -v moved_wandb_cache; then\n"
        # script += "\tmv ~/.wandb_cache_dir.back ~/.wandb_cache_dir\n"
        # script += "fi\n\n"
        #
        # script += "if test -v moved_experiment; then\n"
        # script += "\tmv ~/experiments.back ~/experiments\n"
        # script += "fi\n\n"

        return script


class MilaCluster(SlurmComputingCluster):
    def __init__(self, **kwargs):
        wandb_offline = kwargs.pop("wandb_offline", False)
        transformers_offline = kwargs.pop("transformers_offline", False)
        hf_datasets_offline = kwargs.pop("hf_datasets_offline", False)
        super().__init__(
            **kwargs,
            shared_storage_dir="~/scratch",
            wandb_offline=wandb_offline,
            transformers_offline=transformers_offline,
            hf_datasets_offline=hf_datasets_offline,
        )


class IBMCluster(SlurmComputingCluster):
    def __init__(self, num_submission_to_queue: int = 1, **kwargs):
        wandb_offline = kwargs.pop("wandb_offline", False)
        transformers_offline = kwargs.pop("transformers_offline", False)
        hf_datasets_offline = kwargs.pop("hf_datasets_offline", False)
        shared_storage_dir = kwargs.pop("shared_storage_dir", "~/scratch")
        logs_dir = kwargs.pop("logs_dir", "~/scratch/logs")
        scripts_dir = kwargs.pop("scripts_dir", "~/scratch/scripts")
        super().__init__(
            **kwargs,
            shared_storage_dir=shared_storage_dir,
            logs_dir=logs_dir,
            scripts_dir=scripts_dir,
            wandb_offline=wandb_offline,
            transformers_offline=transformers_offline,
            hf_datasets_offline=hf_datasets_offline,
        )

        self.conda_env_name = kwargs["config"]["conda_env_name"]
        self.num_submission_to_queue = num_submission_to_queue

    def prepare_job(self, output_dir: Path) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)

        import wandb

        if self.wandb_project_name is None:
            project = os.environ.get("WANDB_PROJECT", self.project_name)
        else:
            project = self.wandb_project_name

        if self.wandb_entity_name is None:
            entity = os.environ.get("WANDB_ENTITY", None)
        else:
            entity = self.wandb_entity_name

        api = wandb.Api(
            overrides={"project": project, "entity": entity}, api_key=self.wandb_api_key
        )
        if entity is not None:
            artifact_name = f"{entity}/{project}/"
        else:
            artifact_name = ""

        os.environ["WANDB_API_KEY"] = self.wandb_api_key

        artifact_name += f"bundle-{self.launcher_id}:latest"
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

        try:
            metadata_path = output_dir / "metadata.json"
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            persistent_key = metadata["exp_name"]
        except Exception as e:
            print(
                "Unable to load metadata.json, computing "
                "persistent_dir based on launcher_id"
            )
            persistent_key = create_md5_hash(self.launcher_id)

        worker_script = f"#!/bin/bash\n\n"

        worker_script += "chmod a+x run.sh\n"
        worker_script += "./run.sh\n\n"

        save_and_make_executable(output_dir / self.run_script_name, worker_script)

        return persistent_key

    def create_compute_script(self, tmp_exp_dir: Path, persistent_key: str) -> str:
        script = "#!/bin/bash\n\n"

        random_wait_time = random.randint(0, 11)
        script += f"sleep {random_wait_time} \n"

        stdout_path = f"{self.experiments_dir}/{persistent_key}/stdout.txt"
        stderr_path = f"{self.experiments_dir}/{persistent_key}/stderr.txt"
        script += f"touch {stdout_path} \n"
        script += f"touch {stderr_path} \n\n"

        tmp_dir_home = tmp_exp_dir / "home"
        script += f"export TRANSFORMERS_CACHE={tmp_dir_home}/experiments/hf_cache\n"
        script += f"export HF_DATASETS_CACHE={tmp_dir_home}/experiments/hf_ds_cache\n"
        script += (
            f"export HF_MODULES_CACHE={tmp_dir_home}/experiments/hf_modules_cache\n"
        )
        script += f"export WANDB_CACHE_DIR={self.experiments_dir}/wandb_cache_dir\n"
        script += f"export WANDB_DIR={self.experiments_dir}/{persistent_key}\n\n"

        assert self.wandb_api_key is not None
        script += f"export WANDB_API_KEY={self.wandb_api_key}\n"

        assert self.wandb_project_name is not None
        script += f"export WANDB_PROJECT={self.wandb_project_name}\n"

        if self.wandb_entity_name is not None:
            script += f"export WANDB_ENTITY={self.wandb_entity_name}\n"

        if self.wandb_offline:
            script += "export WANDB_MODE=offline\n"

        if self.transformers_offline:
            script += "export TRANSFORMERS_OFFLINE=1\n"

        if self.hf_datasets_offline:
            script += "export HF_DATASETS_OFFLINE=1\n"

        if self.env_vars is not None:
            for k_v in self.env_vars:
                script += f"export {k_v}\n"

        script += f"cd {tmp_exp_dir}/home\n"
        script += f"conda activate {self.conda_env_name}\n\n"

        if not self.interactive:
            script += f'\necho "Running the computation..." \n'
            script += f"./{self.run_script_name} > {stdout_path} 2> {stderr_path} \n\n"
        else:
            script += "echo '---------------------'\n"
            script += "echo 'Interactive mode, not running the job.'\n"
            script += f"echo 'You can run ./{self.run_script_name} to execute the job manually.'\n"
            script += "echo '---------------------'\n"
            script += f"ls -lh\n"

        return script

    def _create_pre_sbatch_launch_script(
        self, tmp_exp_dir: Path, persistent_key: str
    ) -> str:
        script = "#!/bin/bash \n\n"

        job_persistent_dir = f"{self.experiments_dir}/{persistent_key}"
        script += f"mkdir -p {job_persistent_dir}\n"
        script += f"ln -sfn {job_persistent_dir} {self.log_dir}/exp_dir\n"
        script += f"ln -sfn {job_persistent_dir} {self.experiments_dir}/lid_{self.launcher_id}\n"
        script += "sleep 5\n\n"

        script += f"ln -sfn {self.experiments_dir} {tmp_exp_dir}/home/experiments\n"

        return script

    def _create_sbatch_launch_script(
        self, compute_script_path: Path, persistent_key: str
    ) -> str:
        script = f"cd {self.cluster_shared_storage_dir}\n"
        if not self.interactive:
            script += 'echo "Submitting the job..." \n\n'

            # Command example: jbsub -mem 30g -cores 1+1 -q x86_6h -require v100 -depend n1 -name n2 ...

            last_job_id = None
            for i in range(self.num_submission_to_queue):
                job_name = f"j{self.launcher_id}_{i}"

                cmd = "jbsub"
                cmd += f" -name {job_name}"

                if last_job_id is not None:
                    cmd += f" -depend {last_job_id}"

                cmd += f" -out {self.log_dir}/compute_{i}.txt"
                cmd += f" {self.slurm_args}"
                cmd += f" {compute_script_path}"

                last_job_id = job_name

                script += f"{cmd} \n"
        else:
            script += 'printf "\\n\\n\\n\\n--------------------------------------------------\\n"; \n'
            script += 'printf "Interactive mode\\n"; \n'
            script += 'printf "Run the following command when the job is granted:\\n"; \n'
            script += f'printf "$ source {compute_script_path}\\n"; \n'
            script += 'printf "--------------------------------------------------\\n\\n\\n\\n"; \n'

            cmd = (
                "jbsub"
                f" -interactive"
                f" -name {self.launcher_id}_interactive"
                f" {self.slurm_args}"
                f" bash"
            )

            script += f"{cmd} \n"
        return script


def get_config(required_keys: List[str]) -> Dict[str, Union[str, bool]]:
    config_path = Path(__file__).parent / ".launcher_config.json"
    if config_path.exists():
        config_ob = json.load(config_path.open())
    else:
        config_ob = {}

    key_to_message = {
        "wandb_api_key": "Enter your wandb api key",
        "conda_env_name": "Enter the name of the conda environment",
        "wandb_project_name": "Enter the name of the wandb project",
        "wandb_entity_name": "Enter the name of the wandb entity",
    }

    for key in required_keys:
        if key in config_ob:
            continue

        from InquirerPy import inquirer

        new_config_ob = {
            k: inquirer.text(
                message=key_to_message.get(k, f"Enter {k}"),
            ).execute()
            for k in [key]
        }
        config_ob.update(new_config_ob)

    with config_path.open("w") as f:
        json.dump(config_ob, f, indent=4)

    return config_ob


def launch_job(args: argparse.Namespace) -> None:
    if args.platform == "mila":
        cluster_class = MilaCluster
    elif args.platform == "cc":
        cluster_class = ComputeCanadaCluster
    elif args.platform == "ibm":
        cluster_class = IBMCluster
    else:
        raise ValueError()

    cluster_kwargs = vars(args)
    for k in list(cluster_kwargs.keys()):
        if cluster_kwargs[k] is None:
            del cluster_kwargs[k]

    required_keys = []
    if args.platform == "ibm":
        required_keys += [
            "wandb_api_key",
            "conda_env_name",
            "wandb_project_name",
            "wandb_entity_name",
        ]

    config = get_config(required_keys)

    clstr_args = copy.deepcopy(cluster_kwargs)
    clstr_args.update({"launcher_id": args.bundle, "config": config})
    if "project_name" not in clstr_args:
        clstr_args["project_name"] = os.environ.get("WANDB_PROJECT", "pt_hf_base")

    clstr = cluster_class(**clstr_args)

    clstr.execute_job(None)


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

    project = os.environ.get("WANDB_PROJECT", "pt_hf_base")
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

        launch_job(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment runner")

    parser.add_argument(
        "bundle", metavar="EXP_KEY", nargs="?", type=str, help="Wandb ID"
    )

    parser.add_argument(
        "-p",
        "--platform",
        metavar="PLATFORM",
        type=str,
        choices=["mila", "cc", "ibm"],
        default="mila",
        help="The computation platform we're running the experiment",
    )

    parser.add_argument(
        "-s",
        "--slurm_args",
        metavar="ARGS",
        type=str,
        default="--gres=gpu:1",
        help="Slurm args",
    )

    parser.add_argument(
        "-i", "--image_name", metavar="IMAGE", type=str, help="Container Image"
    )

    parser.add_argument(
        "--images_dir",
        metavar="DIR",
        type=str,
        help="Container Images Directory (only needed for singularity)",
    )

    parser.add_argument(
        "--shared_storage_dir",
        metavar="DIR",
        type=str,
        help="Path to cluster's shared storage between compute nodes and login node",
    )

    parser.add_argument(
        "--compute_storage_dir",
        metavar="DIR",
        type=str,
        help="Path to on-device storage on compute nodes",
    )

    parser.add_argument(
        "--account",
        metavar="ACCOUNT",
        type=str,
        help="Slurm account (only needed for CC)",
    )

    parser.add_argument(
        "--scripts-dir",
        metavar="DIR",
        type=str,
        help="Directory to output generated job scripts",
    )

    parser.add_argument(
        "--logs-dir",
        metavar="DIR",
        type=str,
        help="Directory to store jobs' log",
    )

    parser.add_argument(
        "--env_vars",
        metavar="ENVS",
        type=str,
        help="Environment variables passed to the container, e.g. X1=V1,x2=V2",
    )

    parser.add_argument(
        "--nodup",
        action="store_true",
        help="Do not run already queued experiments",
        default=False,
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="Print queued experiments' info",
        default=False,
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run the job interactively",
        default=False,
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Don't execute the jobs (Useful for testing and debugging).",
        default=False,
    )

    parser.add_argument(
        "-n",
        "--num_submission_to_queue",
        metavar="NUM",
        type=int,
        default=1,
        help="Number of submissions to queue (Only for IBM Cluster)",
    )

    args = parser.parse_args()

    main(args)
