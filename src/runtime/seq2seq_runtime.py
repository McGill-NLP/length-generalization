import copy
import datetime
import json
import logging
import os
from collections import Sized
from pathlib import Path
from typing import Dict, Optional, Any, List

import _jsonnet
import jsonlines
import numpy as np
import torch
import torch.distributed as dist
import transformers
import wandb
from overrides import overrides
from torch import nn
from tqdm import tqdm
from transformers import (
    set_seed,
    Seq2SeqTrainer,
    WEIGHTS_NAME,
    ProgressCallback,
    EarlyStoppingCallback,
)
from transformers.integrations import WandbCallback
from transformers.trainer_pt_utils import metrics_format
from transformers.trainer_utils import get_last_checkpoint
from wandb.sdk.wandb_run import Run

from analyzers import Analyzer
from callbacks import Callback
from common.from_params import create_kwargs
from common.nest import unflatten
from common.torch_utils import is_world_process_zero
from hp_search_space import HPSearchSpace
from runtime.base_runtime import Runtime
from tokenization_utils import Tokenizer
from trainers import BaseTrainer, CustomTrainingArguments

transformers.logging.set_verbosity_info()

from common import (
    Lazy,
    gpu_utils,
    ExperimentStage,
    Params,
    JsonDict,
    py_utils,
)
from common.py_utils import get_human_readable_count, chunks
from data import DataLoaderFactory
from models import Model

logger = logging.getLogger("app")


def get_args_dict(**kwargs) -> Dict[str, Any]:
    return kwargs


class CustomWandbCallback(WandbCallback):
    @overrides
    def setup(self, args, state, model, **kwargs):
        if self._wandb is None:
            return
        self._initialized = True
        if state.is_world_process_zero:
            # define default x-axis (for latest wandb versions)
            if self._wandb.run is None:
                return

            if getattr(self._wandb, "define_metric", None):
                self._wandb.define_metric("train/global_step")
                self._wandb.define_metric(
                    "*", step_metric="train/global_step", step_sync=True
                )

            # keep track of model topology and gradients, unsupported on TPU
            from transformers import is_torch_tpu_available

            if (
                not is_torch_tpu_available()
                and os.getenv("WANDB_WATCH", "false") != "false"
            ):
                has_already_watched = getattr(self, "_has_already_watched", False)
                if not has_already_watched:
                    self._wandb.watch(
                        model,
                        log=os.getenv("WANDB_WATCH", "gradients"),
                        log_freq=max(100, args.logging_steps),
                        log_graph=True,
                    )
                    self._has_already_watched = True

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if self._wandb is None:
            return
        if state.is_world_process_zero:
            if self._wandb.run is None:
                return
            if not self._initialized:
                self.setup(args, state, model)
            logs = self.rewrite_logs(logs)
            self._wandb.log({**logs, "train/global_step": state.global_step})

    @staticmethod
    def rewrite_logs(d):
        new_d = {}
        eval_prefix = "eval_"
        eval_prefix_len = len(eval_prefix)
        test_prefix = "test_"
        test_prefix_len = len(test_prefix)
        pred_prefix = "pred_"
        pred_prefix_len = len(pred_prefix)
        for k, v in d.items():
            if k.startswith(eval_prefix):
                new_d["eval/" + k[eval_prefix_len:]] = v
            elif k.startswith(test_prefix):
                new_d["test/" + k[test_prefix_len:]] = v
            elif k.startswith(pred_prefix):
                new_d["pred/" + k[pred_prefix_len:]] = v
            else:
                new_d["train/" + k] = v
        return new_d


class CustomProgressCallback(ProgressCallback):
    @overrides
    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_local_process_zero and self.training_bar is not None:
            _ = logs.pop("total_flos", None)
            if len(state.log_history) >= 2:
                last_log = state.log_history[-2]
            else:
                last_log = {}

            last_log.update(logs)
            self.training_bar.set_postfix(**last_log)


from transformers import trainer

trainer.DEFAULT_PROGRESS_CALLBACK = CustomProgressCallback


@Runtime.register("seq2seq")
class Seq2SeqRuntime(Runtime):
    def __init__(
        self,
        exp_name: str,
        project_name: str,
        # model: Lazy[Model],
        model: JsonDict,
        dataset: Lazy[DataLoaderFactory],
        tokenizer: Optional[Lazy[Tokenizer]] = None,
        trainer: Optional[Dict[str, Any]] = None,
        directory: Optional[str] = "experiments",
        global_vars: Optional[Dict[str, Any]] = None,
        hp_search_space: Optional[Lazy[HPSearchSpace]] = None,
        config_dict: Optional[Dict[str, Any]] = None,
        sweep_run: Optional[bool] = False,
        analyzers: List[Optional[JsonDict]] = None,
        config_filenames: Optional[List[str]] = None,
        **kwargs,
    ):
        self.lazy_model = model
        assert "type" in model
        self.lazy_dataset = dataset
        self.exp_name = exp_name
        self.project_name = project_name
        self.sweep_run = sweep_run

        exp_root = Path(directory) / self.exp_name
        exp_root.mkdir(parents=True, exist_ok=True)
        self.exp_root = exp_root

        logs_dir = self.exp_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = logs_dir

        self.global_vars = global_vars or {"seed": 123}
        self.debug_mode = self.global_vars.get("debug_mode", False)
        self.dataset_debug_mode = self.global_vars.get(
            "dataset_debug_mode", self.debug_mode
        )

        self.force_offline = self.global_vars.get("force_offline", False)
        self.config_dict = config_dict

        self.training_args = trainer or {}

        seed = self.global_vars["seed"]
        logger.info(f"Setting seed = {seed}")
        set_seed(seed)

        if self.debug_mode:
            logger.info(">>>>>>>>>>>>>>>>> DEBUG MODE <<<<<<<<<<<<<<<<<<<")

        if self.dataset_debug_mode:
            logger.info(">>>>>>>>>>>>>>>>> DATASET DEBUG MODE <<<<<<<<<<<<<<<<<<<")

        if is_world_process_zero():
            assert self.logger is not None

        self.dl_factory = self.lazy_dataset.construct(
            log_dir=self.logs_dir,
            debug_mode=self.dataset_debug_mode,
            seed=seed,
        )
        self.write_meta_data()

        if tokenizer is not None:
            self.tokenizer = tokenizer.construct(
                dataset=self.dl_factory,
                experiment_root=self.exp_root,
            )
        else:
            self.tokenizer = None

        self.dl_factory.set_tokenizer(self.tokenizer)

        self.hp_search_space = hp_search_space or Lazy(HPSearchSpace)

        self.analyzers = analyzers or []

        def get_num_devices() -> int:
            world_size = os.environ.get("WORLD_SIZE", None)
            if world_size is not None and "RANK" in os.environ:
                num_devices = int(world_size)
            else:
                # Multi GPU training should be always launched by torchrun,
                # otherwise we assume single GPU training (even if there are
                # multiple GPUs available)
                num_devices = 1
                if gpu_utils.get_num_gpus() > 1:
                    logger.warning(
                        "You seem to be running on multiple GPUs. "
                        "If you want to use multiple GPUs, please use torchrun."
                    )

            return num_devices

        if "target_batch_size" in self.training_args:
            target_batch_size = self.training_args["target_batch_size"]
            num_devices = get_num_devices()
            num_devices = max(num_devices, 1)

            self.training_args["per_device_train_batch_size"] = (
                target_batch_size // num_devices
            )

            logger.info(f"Target batch size: {target_batch_size}")
            logger.info(f"Number of compute devices: {num_devices}")
            logger.info(
                f"Setting per_device_train_batch_size "
                f"to {self.training_args['per_device_train_batch_size']}"
            )

            if is_world_process_zero():
                self.logger.summary[
                    "computed_per_device_train_batch_size"
                ] = self.training_args["per_device_train_batch_size"]
                self.logger.summary["computed_num_process"] = num_devices

        if "target_eval_batch_size" in self.training_args:
            target_batch_size = self.training_args["target_eval_batch_size"]
            num_devices = get_num_devices()
            num_devices = max(num_devices, 1)

            self.training_args["per_device_eval_batch_size"] = (
                target_batch_size // num_devices
            )

            logger.info(f"Target eval batch size: {target_batch_size}")
            logger.info(f"Number of compute devices: {num_devices}")
            logger.info(
                f"Setting per_device_eval_batch_size "
                f"to {self.training_args['per_device_eval_batch_size']}"
            )

            if is_world_process_zero():
                self.logger.summary[
                    "computed_per_device_eval_batch_size"
                ] = self.training_args["per_device_eval_batch_size"]
                self.logger.summary["computed_num_process"] = num_devices

    def write_meta_data(self):
        if not is_world_process_zero():
            return

        gpu_info = gpu_utils.get_cuda_info()
        if len(gpu_info) != 0:
            # log_obj = {f"gpus_info/#{i}/": gi for i, gi in enumerate(gpu_info)}
            # self.logger.summary.update(log_obj)

            logger.info(f"GPUs Info: \n{json.dumps(gpu_info, indent=4)}")

        metadata = {"exp_name": self.exp_name, "gpus_info": gpu_info}
        with open(self.exp_root / "metadata.json", "w") as f:
            f.write(json.dumps(metadata, indent=4, sort_keys=True))

        conf_path = self.exp_root / "config.json"
        if conf_path.exists():
            self.logger.save(str(conf_path.absolute()), policy="now")

        dotenv_path = self.exp_root / "dotenv.txt"
        with dotenv_path.open("w") as f:
            for k, v in os.environ.items():
                if k.startswith("APP_"):
                    f.write(f"{k}={v}\n")
        self.logger.save(str(dotenv_path.absolute()), policy="now")

    @property
    def logger(self) -> Run:
        if not hasattr(self, "_logger"):
            if wandb.run is None and is_world_process_zero():
                if self.debug_mode:
                    mode = "disabled"
                else:
                    mode = None

                wandb_entity = self.global_vars.get("wandb_entity", None)

                settings = wandb.Settings()
                settings.update(
                    _save_requirements=True,
                    _disable_meta=False,
                )
                wandb.init(
                    config=self.config_dict,
                    project=self.project_name,
                    name=self.exp_name,
                    resume="allow",
                    mode=mode,
                    force=True,
                    entity=wandb_entity,
                )

            self._logger = wandb.run

        return self._logger

    def get_last_checkpoint_path(self) -> Optional[Path]:
        checkpoint_dir = self.exp_root / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        last_checkpoint = get_last_checkpoint(checkpoint_dir)
        if last_checkpoint is not None:
            last_checkpoint = Path(last_checkpoint)

        return last_checkpoint

    def create_model(self) -> Model:
        lazy_model = copy.deepcopy(self.lazy_model)
        model_type = lazy_model.pop("type", Model.default_implementation)
        if model_type is None:
            raise ValueError("Cannot recognize model")
        model_constructor = Model.by_name(model_type)
        model_class = Model.resolve_class_name(model_type)[0]

        from_pretrained = lazy_model.pop("from_pretrained", False)
        pretrained_path = lazy_model.pop("pretrained_path", None)

        model_kwargs = create_kwargs(
            model_constructor,
            model_class,
            params=Params(lazy_model),
            tokenizer=self.tokenizer,
        )

        has_handled_tokenizer = False
        if from_pretrained:
            if pretrained_path is not None:
                exp_root_dir = self.global_vars["dirs"]["experiments"]
                arg = str(Path(exp_root_dir) / pretrained_path / "checkpoints")
                has_handled_tokenizer = True
            else:
                arg = lazy_model["hf_model_name"]
                _ = model_kwargs.pop("tokenizer")

            logger.info(f"Loading initial model weights from {arg}...")
            model = model_class.from_pretrained(arg, **model_kwargs)
        else:
            model = model_constructor(**model_kwargs)
            has_handled_tokenizer = True

        if hasattr(model, "handle_tokenizer") and not has_handled_tokenizer:
            model.handle_tokenizer(self.tokenizer)

        return model

    def create_trainer(self, stage: ExperimentStage, **kwargs) -> Seq2SeqTrainer:
        if "model" not in kwargs:
            kwargs["model"] = self.create_model()

        model = kwargs.get("model")
        eval_split_name = kwargs.pop("eval_split_name", None)

        training_args = copy.deepcopy(self.training_args)
        training_args["output_dir"] = str(self.exp_root / "checkpoints")
        training_args["report_to"] = "none"

        _ = training_args.pop("target_batch_size", None)
        _ = training_args.pop("target_eval_batch_size", None)
        _ = training_args.pop("auto_compute_batch_size", None)

        if kwargs.get("eval_dataset", None) is None:
            training_args["evaluation_strategy"] = "no"

        callbacks = [CustomWandbCallback()]
        early_stopping = training_args.pop("early_stopping", None)
        if early_stopping is not None:
            logger.info(f"Enabled early stopping at {early_stopping}")
            callbacks.append(
                EarlyStoppingCallback(early_stopping_patience=early_stopping)
            )

        user_callbacks = training_args.pop("callbacks", None)
        if user_callbacks is not None:
            for config_obj in user_callbacks:
                cb = Callback.from_params(Params(config_obj))
                cb.init(
                    self,
                    kwargs.get("eval_dataset", None),
                    eval_split_name,
                )
                callbacks.append(cb)

        trainer_type = training_args.pop("type", BaseTrainer.default_implementation)
        trainer_class = BaseTrainer.resolve_class_name(trainer_type)[0]

        training_args = CustomTrainingArguments(**training_args)

        data_collator = self.dl_factory.get_collate_fn(stage)
        if stage == ExperimentStage.TRAINING:
            eval_data_collator = self.dl_factory.get_collate_fn(
                ExperimentStage.VALIDATION
            )
        else:
            eval_data_collator = None

        try:
            data_collator.model = model
        except Exception as exp:
            logger.warning(exp)

        trainer = trainer_class(
            args=training_args,
            tokenizer=getattr(self.dl_factory, "tokenizer", None),
            data_collator=self.dl_factory.get_collate_fn(stage),
            compute_metrics=self.dl_factory.get_compute_metric_fn(stage),
            callbacks=callbacks,
            **kwargs,
        )
        trainer.eval_data_collator = eval_data_collator

        for cb in callbacks:
            if hasattr(cb, "set_trainer") and callable(cb.set_trainer):
                cb.set_trainer(trainer)

        return trainer

    def log_number_of_parameters(self, model: nn.Module):
        try:
            total_parameters = sum(p.numel() for p in model.parameters())
            trainable_parameters = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )

            logger.info(
                "######## Number of Parameters ########\n"
                f"{get_human_readable_count(trainable_parameters)} Trainable params \n"
                f"{get_human_readable_count(total_parameters - trainable_parameters)} Non-trainable params \n"
                f"{get_human_readable_count(total_parameters)} Total params \n"
                "######################################\n"
            )

            if is_world_process_zero():
                self.logger.summary.update(
                    {
                        "num_trainable_params": trainable_parameters,
                        "num_non_trainable_params": total_parameters
                        - trainable_parameters,
                        "num_total_params": total_parameters,
                    },
                )

        except Exception as exp:
            logger.warning("Couldn't log the number of parameters because of ")
            logger.warning(str(exp))

    def log_metrics_to_console(
        self, split: str = "None", metrics: Dict[str, Any] = None
    ):
        if not is_world_process_zero():
            return

        if metrics is None:
            return

        log_str = f"***** {split} metrics *****\n"
        metrics_formatted = metrics_format(None, metrics)
        k_width = max(len(str(x)) for x in metrics_formatted.keys())
        v_width = max(len(str(x)) for x in metrics_formatted.values())
        for key in sorted(metrics_formatted.keys()):
            log_str += f"  {key: <{k_width}} = {metrics_formatted[key]:>{v_width}}\n"

        logger.info(log_str)

    def _load_last_checkpoint(self, trainer: Seq2SeqTrainer) -> Optional[str]:
        last_checkpoint = self.get_last_checkpoint_path()
        if last_checkpoint is not None:
            logger.info(f"Loading checkpoints from {last_checkpoint}")
            state_dict = torch.load(last_checkpoint / WEIGHTS_NAME, map_location="cpu")
            trainer._load_state_dict_in_model(state_dict)
        else:
            logger.info(f"Initializing model from scratch")

        return last_checkpoint

    def _load_best_checkpoint(self, trainer) -> str:
        ckpt_dir = self.exp_root / "checkpoints"
        last_checkpoint = self.get_last_checkpoint_path()
        if (ckpt_dir / "trainer_state.json").exists():
            trainer_state = json.load((ckpt_dir / "trainer_state.json").open())
            best_model_checkpoint = trainer_state.get("best_model_checkpoint", None)
        elif (
            last_checkpoint is not None
            and (last_checkpoint / "trainer_state.json").exists()
        ):
            trainer_state = json.load((last_checkpoint / "trainer_state.json").open())
            best_model_checkpoint = trainer_state.get("best_model_checkpoint", None)
        else:
            best_model_checkpoint = None

        if best_model_checkpoint is None:
            logger.warning("Could not find the best checkpoint")
            raise ValueError("Best checkpoint not found")

        logger.info(f"Loading checkpoints from {best_model_checkpoint}")
        state_dict = torch.load(
            Path(best_model_checkpoint) / WEIGHTS_NAME, map_location="cpu"
        )
        trainer._load_state_dict_in_model(state_dict)

        return best_model_checkpoint

    def train(self, eval_split: str = "valid", train_split: str = "train"):
        logger.info("\n" * 5 + "*" * 100)
        logger.info(f"* Training on {train_split} and evaluating on {eval_split}")
        logger.info("*" * 100 + "\n" * 5)
        torch.cuda.empty_cache()

        model = self.create_model()

        eval_ds_path = self.dl_factory.get_ds_file_path(
            ExperimentStage.from_split(eval_split)
        )
        eval_dataset = self.dl_factory.get_dataset(
            stage=ExperimentStage.VALIDATION, path=eval_ds_path
        )
        if eval_dataset is None:
            logger.info(
                "No evaluation dataset found. Disabled evaluation during training."
            )

        train_ds_path = self.dl_factory.get_ds_file_path(
            ExperimentStage.from_split(train_split)
        )
        train_dataset = self.dl_factory.get_dataset(
            stage=ExperimentStage.TRAINING, path=train_ds_path
        )

        trainer = self.create_trainer(
            ExperimentStage.TRAINING,
            model=model,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            eval_split_name=eval_split,
        )
        self.log_number_of_parameters(trainer.model)

        last_checkpoint = self.get_last_checkpoint_path()
        if last_checkpoint is not None:
            logger.info(f"Loading checkpoints from {last_checkpoint}")
        else:
            logger.info(f"Initializing training from scratch")

        auto_compute_batch_size = self.training_args.get(
            "auto_compute_batch_size", False
        )
        if not auto_compute_batch_size:
            train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            # Compute the maximum batch size fits into the gpu
            # and instead increase gradient accumulation steps
            training_is_done = False
            orig_training_args = copy.deepcopy(self.training_args)
            while not training_is_done:
                try:
                    train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
                    training_is_done = True
                except RuntimeError as e:
                    if "out of memory" not in str(e):
                        raise e

                    try:
                        is_oom_from_eval = trainer.control.should_evaluate
                    except:
                        is_oom_from_eval = False

                    if is_oom_from_eval:
                        logger.warning("OOM during evaluation.")

                        curr_batch_size = self.training_args[
                            "per_device_eval_batch_size"
                        ]
                        self.training_args["per_device_eval_batch_size"] = (
                            curr_batch_size // 2
                        )

                        logger.info(
                            f"Eval Batch size is too large. Reducing it to "
                            f"{self.training_args['per_device_eval_batch_size']} "
                        )
                    else:
                        # Divide the batch_size in half and increase the accumulation steps
                        curr_batch_size = self.training_args[
                            "per_device_train_batch_size"
                        ]
                        self.training_args["per_device_train_batch_size"] = (
                            curr_batch_size // 2
                        )
                        self.training_args["gradient_accumulation_steps"] = (
                            self.training_args.get("gradient_accumulation_steps", 1) * 2
                        )

                        logger.info(
                            f"Batch size is too large. Reducing it to "
                            f"{self.training_args['per_device_train_batch_size']} "
                            f"and increasing gradient accumulation steps to "
                            f"{self.training_args['gradient_accumulation_steps']}"
                        )

                    # Empty the cache
                    torch.cuda.empty_cache()

                    # Recreate the trainer with the new batch size
                    trainer = self.create_trainer(
                        ExperimentStage.TRAINING,
                        model=model,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        eval_split_name=eval_split,
                    )

                    last_checkpoint = self.get_last_checkpoint_path()

                    # Make sure all distributed processes are in sync
                    if dist.is_available() and dist.is_initialized():
                        dist.monitored_barrier(
                            timeout=datetime.timedelta(minutes=15), wait_all_ranks=True
                        )

            if trainer.is_world_process_zero():
                self.logger.summary["auto_computed_batch_size"] = self.training_args[
                    "per_device_train_batch_size"
                ]
                self.logger.summary[
                    "auto_computed_eval_batch_size"
                ] = self.training_args["per_device_eval_batch_size"]
                self.logger.summary[
                    "auto_computed_grad_acc_steps"
                ] = self.training_args.get("gradient_accumulation_steps", 1)

            self.training_args = orig_training_args

        trainer.save_model()

        metrics = train_result.metrics
        if isinstance(train_dataset, Sized) and trainer.is_world_process_zero():
            metrics["num_train_samples"] = len(trainer.train_dataset)
            self.logger.summary.update(
                {"num_train_samples": len(trainer.train_dataset)}
            )

        self.log_metrics_to_console("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    def evaluate(self, split: str = "test", load_best: bool = True):
        logger.info(f"*** Evaluate on {split} ***")
        torch.cuda.empty_cache()
        if "load_best_model_at_end" in self.training_args:
            self.training_args.pop("load_best_model_at_end")

        stage = ExperimentStage.PREDICTION

        trainer = self.create_trainer(stage)
        if load_best:
            try:
                self._load_best_checkpoint(trainer)
            except:
                logger.info("Loading last checkpoint...")
                self._load_last_checkpoint(trainer)
        else:
            logger.info("Loading last checkpoint...")
            self._load_last_checkpoint(trainer)

        ds_path = self.dl_factory.get_ds_file_path(ExperimentStage.from_split(split))
        dataset = self.dl_factory.get_dataset(stage=stage, path=ds_path)
        if dataset is None:
            logger.error(f"No dataset found for split = {split}")
            return

        if dataset is None:
            logger.error(f"No dataset found for split = {split}")
            return

        metrics = trainer.evaluate(
            eval_dataset=dataset, metric_key_prefix=f"eval_{split}"
        )
        if isinstance(dataset, Sized) and trainer.is_world_process_zero():
            metrics[f"eval_{split}_num_samples"] = len(dataset)
            self.logger.summary.update({f"eval_{split}_num_samples": len(dataset)})

        self.log_metrics_to_console(f"eval_{split}", metrics)
        trainer.save_metrics(f"eval_{split}", metrics)

    def predict(
        self,
        split: str = "test",
        enable_metrics: bool = False,
        load_best: bool = True,
        force: bool = False,
    ):
        logger.info("\n" * 5 + "*" * 100)
        logger.info(f"* Predict on {split} *")
        logger.info("*" * 100 + "\n" * 5)
        torch.cuda.empty_cache()
        if "load_best_model_at_end" in self.training_args:
            self.training_args.pop("load_best_model_at_end")

        def load_trainer():
            if load_best:
                try:
                    best_ckpt_path = self._load_best_checkpoint(trainer)
                    if trainer.is_world_process_zero():
                        self.logger.summary.update(
                            {f"predict_{split}_ckpt_path": f"best at {best_ckpt_path}"}
                        )
                except:
                    logger.info("Loading last checkpoint...")
                    last_ckpt_path = self._load_last_checkpoint(trainer)
                    if trainer.is_world_process_zero():
                        self.logger.summary.update(
                            {f"predict_{split}_ckpt_path": f"last at {last_ckpt_path}"}
                        )
            else:
                logger.info("Loading last checkpoint...")
                self._load_last_checkpoint(trainer)
                last_ckpt_path = self._load_last_checkpoint(trainer)
                if trainer.is_world_process_zero():
                    self.logger.summary.update(
                        {f"predict_{split}_ckpt_path": f"last at {last_ckpt_path}"}
                    )

        trainer = self.create_trainer(ExperimentStage.PREDICTION)
        load_trainer()

        stage = ExperimentStage.PREDICTION
        ds_path = self.dl_factory.get_ds_file_path(ExperimentStage.from_split(split))
        dataset = self.dl_factory.get_dataset(stage=stage, path=ds_path)
        if dataset is None:
            logger.error(f"No dataset found for split = {split}")
            return

        output_test_preds_file = self.exp_root / f"pred_out_{split}.jsonl"
        if not force:
            # Check if the predictions already exist and the
            # length of the dataset is the same as the predictions
            if output_test_preds_file.exists():
                try:
                    pred_objs = []
                    with jsonlines.open(output_test_preds_file) as reader:
                        for obj in reader:
                            pred_objs.append(obj)
                    if len(pred_objs) == len(dataset):
                        logger.info(
                            f"Predictions already exist for {split} with length {len(pred_objs)}"
                        )
                        return
                except:
                    pass

        auto_compute_batch_size = self.training_args.get(
            "auto_compute_batch_size", False
        )
        if not auto_compute_batch_size:
            test_results = trainer.predict(dataset, metric_key_prefix=f"pred_{split}")
        else:
            # Compute the maximum batch size fits into the gpu
            # and instead increase gradient accumulation steps
            prediction_is_done = False
            orig_training_args = copy.deepcopy(self.training_args)
            while not prediction_is_done:
                try:
                    test_results = trainer.predict(
                        dataset, metric_key_prefix=f"pred_{split}"
                    )
                    prediction_is_done = True
                except RuntimeError as e:
                    if "out of memory" not in str(e):
                        raise e

                    # Divide the batch_size in half
                    curr_batch_size = self.training_args["per_device_eval_batch_size"]
                    self.training_args["per_device_eval_batch_size"] = (
                        curr_batch_size // 2
                    )

                    logger.info(
                        f"Eval Batch size is too large. Reducing it to "
                        f"{self.training_args['per_device_eval_batch_size']} "
                    )

                    # Empty the cache
                    torch.cuda.empty_cache()

                    # Recreate the trainer with the new batch size
                    trainer = self.create_trainer(ExperimentStage.PREDICTION)

                    load_trainer()

                    # Make sure all distributed processes are in sync
                    if dist.is_available() and dist.is_initialized():
                        dist.monitored_barrier(
                            timeout=datetime.timedelta(minutes=15), wait_all_ranks=True
                        )

            if trainer.is_world_process_zero():
                self.logger.summary[
                    "auto_computed_batch_size_predict"
                ] = self.training_args["per_device_eval_batch_size"]

            self.training_args = orig_training_args

        if trainer.is_world_process_zero():
            metrics = test_results.metrics
            metrics[f"pred_{split}_num_samples"] = len(dataset)
            self.log_metrics_to_console(f"pred_{split}", metrics)

            trainer.save_metrics(f"pred_{split}", metrics)
            trainer.log(metrics)

            preds = test_results.predictions
            if isinstance(test_results.predictions, tuple):
                preds = preds[0]

            if len(preds.shape) == 3:
                preds = np.argmax(preds, axis=-1)

            with output_test_preds_file.open("w") as writer:
                all_objs = []
                for batch_preds in tqdm(
                    chunks(preds, 128),
                    total=len(preds) // 128,
                    desc="Decoding predictions",
                ):
                    # Convert -100 to 0
                    batch_preds = np.where(
                        batch_preds == -100, self.tokenizer.pad_token_id, batch_preds
                    )
                    pred_texts = self.tokenizer.batch_decode(
                        batch_preds,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    pred_texts = [pred.strip() for pred in pred_texts]

                    for pt in pred_texts:
                        all_objs.append({"prediction": pt})

                jsonlines.Writer(writer).write_all(all_objs)

            self.logger.save(str(output_test_preds_file.absolute()), policy="now")

    def train_and_evaluate(
        self,
        eval_split: str = "valid",
        test_split: str = "test",
        load_best: bool = True,
    ):
        self.train(eval_split)
        self.evaluate(test_split, load_best=load_best)

    def combine_pred(self, split: str = "test", force: bool = False):
        if not is_world_process_zero():
            return

        logger.info("\n" * 5 + "*" * 100)
        logger.info(f"* Combining predictions for split = {split}")
        logger.info("*" * 100 + "\n" * 5)

        prediction_path = self.exp_root / f"pred_out_{split}.jsonl"
        logger.info(f"Prediction path: {prediction_path}")
        assert prediction_path.exists()

        stage = ExperimentStage.from_split(split)

        import jsonlines
        import diff_match_patch as dmp_module

        lines_out = []
        with jsonlines.open(str(prediction_path)) as reader:
            for obj in reader:
                lines_out.append(obj)

        input_ds = self.dl_factory.get_dataset(stage)
        assert len(input_ds) == len(lines_out)
        combined_file = self.exp_root / f"pred_combined_{split}.jsonl"

        if not force:
            # Check if the predictions already exist and the
            # length of the dataset is the same as the predictions
            if combined_file.exists():
                try:
                    pred_objs = []
                    with jsonlines.open(combined_file) as reader:
                        for obj in reader:
                            pred_objs.append(obj)
                    if len(pred_objs) == len(input_ds):
                        logger.info(
                            f"Combined predictions already exist for {split} with length {len(pred_objs)}"
                        )
                        return
                except:
                    pass

        pred_table = wandb.Table(
            columns=["idx", "input", "gold", "prediction", "is_correct", "diff"]
        )

        with jsonlines.open(str(combined_file), mode="w") as writer:
            for (obj_ds, obj_pred) in tqdm(zip(input_ds, lines_out)):
                prompt = self.tokenizer.decode(
                    obj_ds["input_ids"],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                labels = [t for t in obj_ds["labels"] if t != -100]
                target = self.tokenizer.decode(
                    labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )

                idx = obj_ds["idx"]
                obj_pred["prompt"] = prompt
                obj_pred["target"] = target
                obj_pred["idx"] = idx

                writer.write(obj_pred)

                prediction = obj_pred["prediction"]

                if prediction[: len(prompt)] == prompt:
                    prediction = prediction[len(prompt) :]

                is_correct = prediction == target
                if not is_correct:
                    dmp = dmp_module.diff_match_patch()
                    diff = dmp.diff_main(target, prediction)
                    dmp.diff_cleanupSemantic(diff)
                    diff = dmp.diff_prettyHtml(diff)
                    diff = wandb.Html(diff)
                else:
                    diff = wandb.Html("")

                pred_table.add_data(idx, prompt, target, prediction, is_correct, diff)

        if os.environ.get("WANDB_MODE", "online") != "offline":
            # For some reason, this doesn't work in offline mode
            self.logger.log({f"pred_{split}/model_outputs": pred_table})

        self.logger.save(str(combined_file.absolute()), policy="now")

        logger.info(f"Done combing!")

    def hp_step(
        self,
        eval_split: str = "valid",
        load_best: bool = True,
    ):
        self.train(eval_split)
        self.predict(eval_split, load_best=load_best, enable_metrics=True)
        self.combine_pred(eval_split)
        self.analyze_all(load_best=load_best, split=eval_split)

    def full_step(
        self,
        eval_split: str = "valid",
        test_split: str = "test",
        load_best: bool = True,
        force: bool = False,
    ):
        self.train(eval_split)

        self.predict(test_split, force=force)
        self.combine_pred(test_split, force=force)
        self.analyze_all(split=test_split)

        self.predict(eval_split, force=force)
        self.combine_pred(eval_split, force=force)
        self.analyze_all(split=eval_split)

    def analyze(self, config_filenames: str, load_best: bool = True):
        if not is_world_process_zero():
            return

        config_filenames = [fn.strip() for fn in config_filenames.split(",")]
        config_obj = py_utils.load_jsonnet_config(config_filenames)

        logger.info(f"*** Analyzing ***")
        logger.info(f"config_files: {config_filenames}")
        torch.cuda.empty_cache()
        if "load_best_model_at_end" in self.training_args:
            self.training_args.pop("load_best_model_at_end")

        trainer = self.create_trainer(ExperimentStage.PREDICTION)
        if load_best:
            try:
                best_ckpt_path = self._load_best_checkpoint(trainer)
                if trainer.is_world_process_zero():
                    self.logger.summary.update(
                        {f"analyze_ckpt_path": f"best at {best_ckpt_path}"}
                    )
            except:
                logger.info("Loading last checkpoint...")
                last_ckpt_path = self._load_last_checkpoint(trainer)
                if trainer.is_world_process_zero():
                    self.logger.summary.update(
                        {f"analyze_ckpt_path": f"last at {last_ckpt_path}"}
                    )
        else:
            logger.info("Loading last checkpoint...")
            self._load_last_checkpoint(trainer)
            last_ckpt_path = self._load_last_checkpoint(trainer)
            if trainer.is_world_process_zero():
                self.logger.summary.update(
                    {f"analyze_ckpt_path": f"last at {last_ckpt_path}"}
                )

        analyzer = Analyzer.from_params(
            Params(config_obj),
            model=trainer.model,
            logger=self.logger,
            dl_factory=self.dl_factory,
            device=trainer.args.device,
            batch_size=trainer.args.eval_batch_size,
            num_beams=trainer.args.generation_num_beams,
            max_length=trainer.args.generation_max_length,
            exp_root=self.exp_root,
        )

        logger.info(f"Using {analyzer.__class__.__name__}...")
        analyzer.analyze()
        analyzer.flush_local_log()

    def analyze_all(self, load_best: bool = True, split: str = "test"):
        if not is_world_process_zero():
            return

        if self.analyzers is None:
            logger.warning("self.analyzers is None. Exiting...")
            return

        logger.info("\n" * 5 + "*" * 100)
        logger.info(f"* Running all analyzers on {split}...")
        logger.info("*" * 100 + "\n" * 5)

        torch.cuda.empty_cache()
        if "load_best_model_at_end" in self.training_args:
            self.training_args.pop("load_best_model_at_end")

        trainer = self.create_trainer(ExperimentStage.PREDICTION)
        has_loaded_best = False
        if load_best:
            try:
                best_ckpt_path = self._load_best_checkpoint(trainer)
                if trainer.is_world_process_zero():
                    self.logger.summary.update(
                        {f"analyze_all_{split}_ckpt_path": f"best at {best_ckpt_path}"}
                    )
                has_loaded_best = True
            except:
                logger.info("Loading last checkpoint...")
                last_ckpt_path = self._load_last_checkpoint(trainer)
                if trainer.is_world_process_zero():
                    self.logger.summary.update(
                        {f"analyze_all_{split}_ckpt_path": f"last at {last_ckpt_path}"}
                    )
        else:
            logger.info("Loading last checkpoint...")
            self._load_last_checkpoint(trainer)
            last_ckpt_path = self._load_last_checkpoint(trainer)
            if trainer.is_world_process_zero():
                self.logger.summary.update(
                    {f"analyze_all_{split}_ckpt_path": f"last at {last_ckpt_path}"}
                )

        # Move model to args.device
        trainer._move_model_to_device(trainer.model, trainer.args.device)

        analyzers = copy.deepcopy(self.analyzers)
        for config_obj in analyzers:
            analyzer = Analyzer.from_params(
                Params(config_obj),
                model=trainer.model,
                tokenizer=self.tokenizer,
                logger=self.logger,
                dl_factory=self.dl_factory,
                device=trainer.args.device,
                batch_size=trainer.args.eval_batch_size,
                num_beams=trainer.args.generation_num_beams,
                max_length=trainer.args.generation_max_length,
                exp_root=self.exp_root,
                split=split,
            )
            if analyzer.require_best_model and not has_loaded_best:
                raise RuntimeError(
                    "Analyzer requires best model but no best model was found."
                )

            logger.info(f"Using {analyzer.__class__.__name__}...")
            analyzer.analyze()
            analyzer.flush_local_log()

    def get_loaded_trainer(
        self, load_best: bool = False, checkpoint_name: str = None
    ) -> Seq2SeqTrainer:
        if "load_best_model_at_end" in self.training_args:
            self.training_args.pop("load_best_model_at_end")

        trainer = self.create_trainer(ExperimentStage.PREDICTION)

        if checkpoint_name:
            ckpt_dir = self.exp_root / "checkpoints" / checkpoint_name
            if ckpt_dir.exists():
                logger.info(f"Loading checkpoints from {ckpt_dir}")
                state_dict = torch.load(ckpt_dir / WEIGHTS_NAME, map_location="cpu")
                trainer._load_state_dict_in_model(state_dict)
        else:
            if load_best:
                try:
                    self._load_best_checkpoint(trainer)
                except:
                    logger.info(
                        "Failed to load best checkpoint, Loading last checkpoint..."
                    )
                    self._load_last_checkpoint(trainer)
            else:
                logger.info("Loading last checkpoint...")
                self._load_last_checkpoint(trainer)

        return trainer

    def console_inference(self, load_best: bool = False, ckpt_name: str = None):
        torch.cuda.empty_cache()
        trainer = self.get_loaded_trainer(
            load_best=load_best, checkpoint_name=ckpt_name
        )

        from runtime.model_inference_shell import ModelInferenceShell

        shell = ModelInferenceShell(self, trainer)
        shell.cmdloop()

    def hp_tune(self, eval_split: str = "valid"):
        logger.info(f"*** Hyperparameter Tuning (on split: {eval_split}) ***")
        torch.cuda.empty_cache()

        train_dataset = self.dl_factory.get_dataset(stage=ExperimentStage.TRAINING)
        eval_stage = ExperimentStage.from_split(eval_split)
        eval_dataset = self.dl_factory.get_dataset(stage=eval_stage)
        if eval_dataset is None:
            raise ValueError(
                "No evaluation dataset found. Disabled evaluation during training."
            )

        base_model_config = self.config_dict.get("model", {})
        tokenizer = self.tokenizer

        def model_init(config: Dict[str, Any]) -> Model:
            config = config or {}
            config = unflatten(config, "/")
            model_config = config.get("MHSP", {})

            base_cfg_str = json.dumps(base_model_config)
            diff_cfg_str = json.dumps(model_config)

            jsonnet_str = f"""
            local base = {base_cfg_str};
            local diff = {diff_cfg_str}; 
            std.mergePatch(base, diff)
            """
            new_config_json = _jsonnet.evaluate_snippet("snippet", jsonnet_str)
            new_config = json.loads(new_config_json)
            new_config = Params(new_config)

            model = Lazy(Model, params=new_config).construct(
                tokenizer=tokenizer,
            )

            return model

        trainer = self.create_trainer(
            ExperimentStage.TRAINING,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            model=None,
            model_init=model_init,
        )

        search_space = self.hp_search_space.construct()

        resources_per_trial = {
            int(os.environ.get("APP_HPT_RES_PER_TRIAL_CPU", "1")),
            int(os.environ.get("APP_HPT_RES_PER_TRIAL_GPU", "1")),
        }
        trainer.hyperparameter_search(
            backend="ray",
            hp_space=search_space.get_search_space_fn(),
            compute_objective=search_space.get_compute_obj_fn(),
            direction=search_space.direction,
            resources_per_trial=resources_per_trial,
            keep_checkpoints_num=1,
        )


Runtime.default_implementation = "seq2seq"
