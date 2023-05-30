import itertools
import json
import os
import site
from pathlib import Path
from typing import Dict, Any, List, Optional

import fire

site.addsitedir("src/")
from common.py_utils import (
    get_run_name_from_config_obj,
    generate_deterministic_hp_run_id,
)


import _jsonnet

from common.nest import flatten


def create_sweep_config_obj(config_filenames: List[str]) -> Dict[str, Any]:
    jsonnet_str = "+".join([f'(import "{f}")' for f in config_filenames])
    json_str = _jsonnet.evaluate_snippet("snippet", jsonnet_str)
    config: Dict[str, Any] = json.loads(json_str)

    parameters = config.get("parameters", {})
    parameters = flatten(parameters, separator=".")
    for k, v in parameters.items():
        parameters[k] = json.loads(v)

    config["parameters"] = parameters

    return config


def is_training_complete(exp_dir: Path):
    try:
        config = json.load((exp_dir / "config.json").open())
        max_steps = config["trainer"]["max_steps"]
    except Exception as e:
        return False

    try:
        trainer_state = json.load(
            (exp_dir / "checkpoints" / "trainer_state.json").open()
        )
        last_step = trainer_state["global_step"]
    except Exception as e:
        return False

    if abs(last_step - max_steps) > 0.01 * max_steps:
        return False

    return True


def is_eval_complete(exp_dir: Path, metric_name: str) -> bool:
    analysis_root = exp_dir / "analysis"
    if not analysis_root.exists():
        return False

    is_complete = False
    for analyzer_dir in analysis_root.iterdir():
        if not analyzer_dir.is_dir():
            continue

        log_file = analyzer_dir / "log.json"
        if not log_file.exists():
            continue

        try:
            log = json.load(log_file.open())
            if metric_name in log:
                is_complete = True
                break
        except Exception as e:
            continue

    return is_complete


def read_eval_metric(exp_dir: Path, metric_name: str) -> Optional[Any]:
    analysis_root = exp_dir / "analysis"
    if not analysis_root.exists():
        return None

    for analyzer_dir in analysis_root.iterdir():
        if not analyzer_dir.is_dir():
            continue

        log_file = analyzer_dir / "log.json"
        if not log_file.exists():
            continue

        try:
            log = json.load(log_file.open())
            if metric_name in log:
                return log[metric_name]
        except Exception as e:
            continue

    return None


def is_hp_run_complete(exp_dir: Path, metric_name) -> bool:
    return is_training_complete(exp_dir) and is_eval_complete(exp_dir, metric_name)


def iter_sweep_params(sweep_config):
    sweep_params_dict: Dict[str, List[Any]] = {
        p: v["values"] for p, v in sweep_config["parameters"].items()
    }
    items = sorted(sweep_params_dict.items(), key=lambda x: x[0])
    if len(items) == 0:
        return
    keys, values = zip(*items)
    sweep_params: List[Dict[str, Any]] = [
        dict(zip(keys, v)) for v in itertools.product(*values)
    ]

    for sweep_param in sweep_params:
        yield sweep_param


class Main:
    def __init__(
        self,
        sweep_name: str = None,
        sweep_root_dir: str = None,
        sweep_configs: str = None,
    ):
        self.sweep_name = os.environ["SWEEP_NAME"] if sweep_name is None else sweep_name
        self.sweep_configs = (
            os.environ["SWEEP_CONFIGS"] if sweep_configs is None else sweep_configs
        )
        self.sweep_root_dir = sweep_root_dir
        assert self.sweep_root_dir is not None
        self.sweep_root_dir = Path(self.sweep_root_dir)
        self.sweep_root_dir.mkdir(parents=True, exist_ok=True)

        self.sweep_config = create_sweep_config_obj(
            [f.strip() for f in sweep_configs.split(",")]
        )

    def dump_sweep_config_obj(self):
        with open(f"sweep_cfg.json", "w") as f:
            json.dump(self.sweep_config, f)

    def create_exp_configs(self):
        sweep_config_dir = self.sweep_root_dir / "hyperparameters"
        sweep_config_dir.mkdir(parents=True, exist_ok=True)

        for i, hp_dict in enumerate(iter_sweep_params(self.sweep_config)):
            run_name = get_run_name_from_config_obj(hp_dict)
            hp_dict["__exp_name__"] = run_name
            config_file = sweep_config_dir / f"{run_name}.json"
            config_file.write_text(json.dumps(hp_dict, indent=4, sort_keys=True))

    def is_hp_run_complete(self, run_name: str):
        experiment_dir = self.sweep_root_dir / "exps" / run_name
        metric_name = self.sweep_config["metric"]["name"]

        if is_hp_run_complete(experiment_dir, metric_name):
            print("True", end="")
        else:
            print("False", end="")

    def fail_if_sweep_not_complete(self):
        exps_root_dir = self.sweep_root_dir / "exps"
        experiments = {
            exp_dir.name: exp_dir
            for exp_dir in exps_root_dir.iterdir()
            if exp_dir.is_dir()
        }

        metric_name = self.sweep_config["metric"]["name"]

        for i, hp_dict in enumerate(iter_sweep_params(self.sweep_config)):
            run_name = get_run_name_from_config_obj(hp_dict)
            if run_name not in experiments:
                print(f"Experiment {run_name} not found")
                exit(1)

            experiment_dir = self.sweep_root_dir / "exps" / run_name
            if not is_hp_run_complete(experiment_dir, metric_name):
                print(f"Experiment {run_name} not complete")
                exit(1)

    def save_best_config(self, output_path: str = None):
        exps_root_dir = self.sweep_root_dir / "exps"

        experiments = {
            exp_dir.name: exp_dir
            for exp_dir in exps_root_dir.iterdir()
            if exp_dir.is_dir()
        }

        metric_name = self.sweep_config["metric"]["name"]
        metric_goal = self.sweep_config["metric"]["goal"]  # minimize / maximize

        best_run_name = None
        best_metric_value = None
        for i, hp_dict in enumerate(iter_sweep_params(self.sweep_config)):
            run_name = get_run_name_from_config_obj(hp_dict)
            if run_name not in experiments:
                print(f"Experiment {run_name} not found")
                exit(1)

            experiment_dir = self.sweep_root_dir / "exps" / run_name
            if not is_hp_run_complete(experiment_dir, metric_name):
                print(f"Experiment {run_name} not complete")
                exit(1)

            metric_value = read_eval_metric(experiment_dir, metric_name)
            if metric_value is None:
                print(f"Experiment {run_name} metric {metric_name} not found")
                exit(1)

            if metric_goal == "minimize":
                if best_metric_value is None or metric_value < best_metric_value:
                    best_metric_value = metric_value
                    best_run_name = run_name
            elif metric_goal == "maximize":
                if best_metric_value is None or metric_value > best_metric_value:
                    best_metric_value = metric_value
                    best_run_name = run_name
            else:
                raise ValueError(f"Unknown metric goal {metric_goal}")

        print(f"Best run name: {best_run_name}")
        print(f"Best metric value: {best_metric_value}")

        # Load best run config
        with (exps_root_dir / best_run_name / "config.json").open() as f:
            best_config = json.load(f)

        # Save best run config
        if output_path is None:
            output_path = "best_run.json"
        with open(output_path, "w") as f:
            json.dump(best_config, f, indent=4, sort_keys=True)

        with open("best_run_info.json", "w") as f:
            json.dump(
                {
                    "name": best_run_name,
                    "id": best_run_name,
                    "sweep_name": self.sweep_name,
                    "metric_name": metric_name,
                    "metric_value": best_metric_value,
                },
                f,
            )

    def generate_deterministic_run_id(self, run_name: str):
        run_id = generate_deterministic_hp_run_id(self.sweep_name, run_name)
        print(run_id, end="")


if __name__ == "__main__":
    fire.Fire(Main)
