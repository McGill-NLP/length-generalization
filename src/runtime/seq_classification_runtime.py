import copy
import logging
from pathlib import Path

import numpy as np
import torch
import transformers
import wandb
from tqdm import tqdm

from common.from_params import create_kwargs
from runtime import Seq2SeqRuntime
from runtime.base_runtime import Runtime

transformers.logging.set_verbosity_info()

from common import (
    ExperimentStage,
    Params,
)
from common.py_utils import chunks
from models import Model

logger = logging.getLogger("app")


@Runtime.register("seq_classification")
class SequenceClassificationRuntime(Seq2SeqRuntime):
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
            label2id=self.dl_factory.label2id,
            id2label=self.dl_factory.id2label,
            problem_type=self.dl_factory.get_problem_type(),
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
            model = model_class.from_pretrained(
                arg, **model_kwargs, cache_dir=str(self.cache_dir)
            )
        else:
            model = model_constructor(**model_kwargs)
            has_handled_tokenizer = True

        if hasattr(model, "handle_tokenizer") and not has_handled_tokenizer:
            model.handle_tokenizer(self.tokenizer)

        return model

    def predict(
        self, split: str = "test", enable_metrics: bool = False, load_best: bool = True
    ):
        logger.info(f"*** Predict on {split} ***")
        torch.cuda.empty_cache()
        if "load_best_model_at_end" in self.training_args:
            self.training_args.pop("load_best_model_at_end")

        trainer = self.create_trainer(ExperimentStage.PREDICTION)
        if load_best:
            try:
                self._load_best_checkpoint(trainer)
            except:
                logger.info("Loading last checkpoint...")
                self._load_last_checkpoint(trainer)
        else:
            logger.info("Loading last checkpoint...")
            self._load_last_checkpoint(trainer)

        stage = ExperimentStage.PREDICTION
        ds_path = self.dl_factory.get_ds_file_path(ExperimentStage.from_split(split))
        dataset = self.dl_factory.get_dataset(stage=stage, path=ds_path)
        if dataset is None:
            logger.error(f"No dataset found for split = {split}")
            return

        if not isinstance(dataset, dict):
            dataset_dict = {split: dataset}
        else:
            dataset_dict = dataset

        for split, dataset in dataset_dict.items():
            test_results = trainer.predict(dataset, metric_key_prefix=f"pred_{split}")

            metrics = test_results.metrics
            metrics[f"pred_{split}_num_samples"] = len(dataset)
            self.log_metrics_to_console(f"pred_{split}", metrics)
            trainer.save_metrics(f"pred_{split}", metrics)
            trainer.log(metrics)

            if trainer.is_world_process_zero():
                preds = test_results.predictions
                if isinstance(test_results.predictions, tuple):
                    preds = preds[0]

                is_regression = self.dl_factory.get_problem_type() == "regression"
                preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

                output_test_preds_file = self.exp_root / f"pred_out_{split}.jsonl"
                with output_test_preds_file.open("w") as writer:
                    all_objs = []
                    for batch_preds in tqdm(
                        chunks(preds, 128),
                        total=len(preds) // 128,
                        desc="Decoding predictions",
                    ):
                        pred_texts = [
                            float(p) if is_regression else self.dl_factory.id2label[p]
                            for p in batch_preds
                        ]

                        for pt in pred_texts:
                            all_objs.append({"prediction": pt})

                    import jsonlines

                    jsonlines.Writer(writer).write_all(all_objs)

                self.logger.save(str(output_test_preds_file.absolute()), policy="now")

    def combine_pred(self, split: str = "test"):
        logger.info(f"*** Combing predictions on split: {split} ***")

        import jsonlines

        stage = ExperimentStage.PREDICTION
        ds_path = self.dl_factory.get_ds_file_path(ExperimentStage.from_split(split))
        input_ds = self.dl_factory.get_dataset(stage=stage, path=ds_path)

        if not isinstance(input_ds, dict):
            input_ds_dict = {split: input_ds}
        else:
            input_ds_dict = input_ds

        for split, input_ds in input_ds_dict.items():
            prediction_path = self.exp_root / f"pred_out_{split}.jsonl"
            logger.info(f"Prediction path: {prediction_path}")
            assert prediction_path.exists()

            lines_out = []
            with jsonlines.open(str(prediction_path)) as reader:
                for obj in reader:
                    lines_out.append(obj)

            assert len(input_ds) == len(lines_out)

            pred_table = wandb.Table(
                columns=["idx", "input", "gold", "prediction", "is_correct", "diff"]
            )
            combined_file = self.exp_root / f"pred_combined_{split}.jsonl"

            is_regression = self.dl_factory.get_problem_type() == "regression"

            with jsonlines.open(str(combined_file), mode="w") as writer:
                for (obj_ds, obj_pred) in tqdm(zip(input_ds, lines_out)):
                    prompt = self.tokenizer.decode(
                        obj_ds["input_ids"],
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    labels = obj_ds["labels"]
                    if not is_regression:
                        target = self.dl_factory.id2label[labels]
                    else:
                        target = labels

                    idx = obj_ds["idx"]
                    obj_pred["prompt"] = prompt
                    obj_pred["target"] = target
                    obj_pred["idx"] = idx

                    writer.write(obj_pred)

                    prediction = obj_pred["prediction"]

                    is_correct = prediction == target

                    diff = 0
                    if is_regression:
                        if not is_correct:
                            diff = abs(labels - prediction)

                    pred_table.add_data(
                        idx, prompt, target, prediction, is_correct, diff
                    )

            self.logger.log({f"pred_{split}/model_outputs": pred_table})
            self.logger.save(str(combined_file.absolute()), policy="now")

        logger.info(f"Done combing!")
