import random
import time

from common.torch_utils import is_world_process_zero, get_rank

INITIAL_SEED = 12345
random.seed(INITIAL_SEED)
import numpy as np

np.random.seed(INITIAL_SEED)
import torch

torch.manual_seed(INITIAL_SEED)
torch.cuda.manual_seed_all(INITIAL_SEED)

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import _jsonnet
import fire

from common import py_utils, Params, JsonDict
from common.nest import unflatten
from common.py_utils import unique_experiment_name
from runtime import Runtime

logger = logging.getLogger("app")
LOG_FORMAT = "%(levelname)s:%(name)-5s %(message)s"
logger.setLevel(logging.DEBUG)

handler = logging.StreamHandler(sys.stderr)
handler.setFormatter(py_utils.NewLineFormatter(LOG_FORMAT))
logger.addHandler(handler)

DEFAULT_SEED = str(INITIAL_SEED)


class EntryPoint(object):
    _exp = None
    _config = None

    def __init__(self, configs: str, debug_mode: bool = None):
        filenames = [f.strip() for f in configs.split(",")]

        config = self._load_config_obj(filenames)

        if debug_mode is not None:
            if "global_vars" not in config:
                config["global_vars"] = dict()
            config["global_vars"]["debug_mode"] = debug_mode

        if config.get("sweep_run", False):
            is_manual_sweep = os.environ.get("APP_MANUAL_SWEEP", False)
            if not is_manual_sweep:
                config = self._get_config_for_sweep(config)
            else:
                logger.info("Activating manual sweep mode")
                config = self._get_config_for_manual_sweep(config)
        else:
            config["exp_name"] = os.environ.get(
                "APP_EXPERIMENT_NAME", unique_experiment_name(config)
            )

        if is_world_process_zero():
            config_str = json.dumps(config, indent=4, sort_keys=True)
            logger.info(f"# configs: {filenames}")
            logger.info(f"----Config----\n{config_str}\n--------------")

        self._dump_config_obj(config)

        config = self._patch_config_obj_for_di(config)

        try:
            DEBUG_MODE = config["global_vars"]["debug_mode"]
        except:
            pass

        self._config = config
        self._exp = Runtime.from_params(Params({"config_dict": config, **config}))

    def _get_config_for_sweep(self, config: JsonDict) -> JsonDict:
        rank = get_rank()
        if rank == -1:
            return self._patch_config_obj_for_sweep(config)

        run_id, exps_dir = self._get_sweep_run_id_and_dir(config)
        cfg_dump_path = Path(exps_dir) / f"cfg_{run_id}.json"
        if rank == 0:
            cfg = self._patch_config_obj_for_sweep(config)
            with cfg_dump_path.open("w") as f:
                json.dump(cfg, f, indent=4, sort_keys=True)
            time.sleep(0.1)
        else:
            # Poll for the config file to be created
            while not cfg_dump_path.exists():
                time.sleep(0.1)
            logger.info(f"Rank {rank}: Fetched config from {cfg_dump_path}")
            with cfg_dump_path.open("r") as f:
                cfg = json.load(f)

        return cfg

    def _get_sweep_run_id_and_dir(self, config: JsonDict) -> (str, str):
        from wandb import env as wandb_env

        sweep_id = os.environ[wandb_env.SWEEP_ID]
        run_id = os.environ[wandb_env.RUN_ID]

        base_dir = Path(config.get("directory", "experiments"))
        sweep_root = base_dir / f"wandb_sweep_{sweep_id}"

        exps_dir = sweep_root / "exps"
        exps_dir.mkdir(parents=True, exist_ok=True)

        return run_id, str(exps_dir)

    def _patch_config_obj_for_sweep(self, config: JsonDict) -> JsonDict:
        import wandb
        from wandb import env as wandb_env

        sweep_id = os.environ[wandb_env.SWEEP_ID]
        run_id = os.environ[wandb_env.RUN_ID]

        orig_exp_name = os.environ.get(
            "APP_EXPERIMENT_NAME", unique_experiment_name(config)
        )

        base_dir = Path(config.get("directory", "experiments"))
        sweep_root = base_dir / f"wandb_sweep_{sweep_id}"

        exps_dir = sweep_root / "exps"
        exps_dir.mkdir(parents=True, exist_ok=True)
        config["directory"] = str(exps_dir)


        run = wandb.init(allow_val_change=True)
        new_hyperparams = run.config.as_dict()
        new_hyperparams = {
            k: v for k, v in new_hyperparams.items() if not k.startswith("_wandb")
        }
        run_name = sorted(
            [(k.split(".")[-1], v) for k, v in new_hyperparams.items()],
            key=lambda x: x[0],
        )
        run_name = "_".join(f"{k[:2]+k[-2:]}:{str(v)}" for k, v in run_name)
        run.name = run_name
        config["exp_name"] = run_name

        new_hyperparams = unflatten(new_hyperparams, ".")
        logger.info(f"New hyperparams: {new_hyperparams}")

        jsonnet_str = f"""
                    local base = {json.dumps(config)};
                    local diff = {new_hyperparams}; 
                    std.mergePatch(base, diff)
                    """
        patched_config = _jsonnet.evaluate_snippet("snippet", jsonnet_str)
        patched_config = json.loads(patched_config)

        run.config.update(patched_config)

        return patched_config

    def _get_config_for_manual_sweep(self, config: JsonDict) -> JsonDict:
        assert (
            "APP_LAUNCHED_BY_MANUAL_SWEEPER" in os.environ
        ), "This is not a manual sweep run"

        sweep_root = os.environ["APP_SWEEP_ROOT_DIR"]

        sweep_root = Path(sweep_root)
        exps_dir = sweep_root / "exps"
        exps_dir.mkdir(parents=True, exist_ok=True)
        config["directory"] = str(exps_dir)

        new_hyperparams_file = os.environ["APP_MANUAL_SWEEP_HYPERPARAMETER_FILE"]
        with open(new_hyperparams_file, "r") as f:
            new_hyperparams: Dict[str, Any] = json.load(f)
        run_name = new_hyperparams.pop("__exp_name__")
        config["exp_name"] = run_name

        new_hyperparams = unflatten(new_hyperparams, ".")

        jsonnet_str = f"""
                    local base = {json.dumps(config)};
                    local diff = {new_hyperparams}; 
                    std.mergePatch(base, diff)
                    """
        patched_config = _jsonnet.evaluate_snippet("snippet", jsonnet_str)
        patched_config = json.loads(patched_config)

        return patched_config

    def _patch_config_obj_for_di(self, config):
        if "runtime_type" in config:
            config["type"] = config["runtime_type"]
            del config["runtime_type"]
        return config

    def _dump_config_obj(self, config):
        exp_root = Path(config.get("directory", "experiments")) / config["exp_name"]
        exp_root.mkdir(parents=True, exist_ok=True)
        json.dump(
            config, (exp_root / "config.json").open("w"), indent=4, sort_keys=True
        )

    def _load_config_obj(self, filenames: List[str]) -> Dict[str, Any]:
        ext_vars = {k: v for k, v in os.environ.items() if k.startswith("APP_")}
        seed = os.environ.get("APP_SEED", DEFAULT_SEED)
        if not seed.isnumeric():
            seed = DEFAULT_SEED
        ext_vars["APP_SEED"] = seed
        jsonnet_str = "+".join([f'(import "{f}")' for f in filenames])
        json_str = _jsonnet.evaluate_snippet("snippet", jsonnet_str, ext_vars=ext_vars)
        config: Dict[str, Any] = json.loads(json_str)
        config["config_filenames"] = filenames

        orig_directory = config.get("directory", "experiments")
        config["directory"] = os.environ.get("APP_DIRECTORY", orig_directory)

        return config

    def __getattr__(self, attr):
        if attr in self.__class__.__dict__:
            return getattr(self, attr)
        else:
            return getattr(self._exp, attr)

    def __dir__(self):
        return sorted(set(super().__dir__() + self._exp.__dir__()))


if __name__ == "__main__":
    fire.Fire(EntryPoint)
