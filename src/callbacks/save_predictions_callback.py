from typing import Optional

import numpy as np
from datasets import Dataset
from tqdm import tqdm
from transformers import (
    Trainer,
    TrainerState,
    PreTrainedTokenizer,
    TrainerControl,
    TrainingArguments,
)
from wandb.sdk.wandb_run import Run

from callbacks.base_callback import Callback
from common import ExperimentStage
from common.py_utils import chunks
from common.torch_utils import is_world_process_zero


@Callback.register("save_predictions")
class SavePredictionsCallback(Callback):
    def __init__(
        self, upload_predictions: bool = False, split: Optional[str] = "valid", **kwargs
    ):
        super().__init__(**kwargs)
        self.upload_predictions = upload_predictions
        self.split = split

    def init(self, runtime, eval_dataset: Dataset, eval_split: str, **kwargs):
        super().init(runtime, eval_dataset, eval_split, **kwargs)

        eval_split = self.split

        self.predictions_dir = runtime.exp_root / f"eval_on_{eval_split}_predictions"
        self.predictions_dir.mkdir(exist_ok=True, parents=True)

        eval_ds_path = runtime.dl_factory.get_ds_file_path(
            ExperimentStage.from_split(eval_split)
        )
        eval_dataset = runtime.dl_factory.get_dataset(
            stage=ExperimentStage.VALIDATION, path=eval_ds_path
        )

        from data import SequenceClassificationDataLoaderFactory

        self.is_seq_classification = isinstance(
            runtime.dl_factory, SequenceClassificationDataLoaderFactory
        )
        if self.is_seq_classification:
            self.is_regression = runtime.dl_factory.get_problem_type() == "regression"
            self.id2label = runtime.dl_factory.id2label
        else:
            self.is_regression = False
            self.id2label = False

        self.dataset = eval_dataset
        self._trainer: Trainer = None
        self._log_counts = 0

    def set_trainer(self, trainer: Trainer):
        self._trainer = trainer

    def on_evaluate(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        tokenizer: PreTrainedTokenizer = None,
        **kwargs,
    ):
        self._save_predictions(state, tokenizer)

    def _save_predictions(self, state: TrainerState, tokenizer: PreTrainedTokenizer):
        if hasattr(self._trainer, "eval_data_collator"):
            eval_data_collator = self._trainer.eval_data_collator
            old_data_collator = self._trainer.data_collator
            self._trainer.data_collator = eval_data_collator

        test_results = self._trainer.predict(self.dataset, metric_key_prefix=f"pred")
        if hasattr(self._trainer, "eval_data_collator"):
            self._trainer.data_collator = old_data_collator

        if state.is_world_process_zero:
            preds = test_results.predictions

            output_test_preds_file = (
                self.predictions_dir
                / f"{self._log_counts}_epoch-{str(state.epoch).zfill(5)}_step-{str(state.global_step).zfill(6)}.jsonl"
            )

            if self.is_seq_classification:
                self._write_predictions_to_file_for_sequence_classification(
                    output_test_preds_file, preds, tokenizer
                )
            else:
                self._write_predictions_to_file(
                    output_test_preds_file, preds, tokenizer
                )

        self._log_counts += 1

    def _write_predictions_to_file(self, output_test_preds_file, preds, tokenizer):
        if not is_world_process_zero():
            return

        if isinstance(preds, tuple):
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
                batch_preds = np.where(
                    batch_preds == -100, tokenizer.pad_token_id, batch_preds
                )
                pred_texts = tokenizer.batch_decode(
                    batch_preds,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                pred_texts = [pred.strip() for pred in pred_texts]

                for pt in pred_texts:
                    all_objs.append({"prediction": pt})

            import jsonlines

            jsonlines.Writer(writer).write_all(all_objs)

    def _write_predictions_to_file_for_sequence_classification(
        self, output_test_preds_file, preds, tokenizer
    ):
        if not is_world_process_zero():
            return

        if isinstance(preds, tuple):
            preds = preds[0]

        is_regression = self.is_regression
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

        with output_test_preds_file.open("w") as writer:
            all_objs = []
            for batch_preds in tqdm(
                chunks(preds, 128),
                total=len(preds) // 128,
                desc="Decoding predictions",
            ):
                pred_texts = [
                    float(p) if is_regression else self.id2label[p] for p in batch_preds
                ]

                for pt in pred_texts:
                    all_objs.append({"prediction": pt})

            import jsonlines

            jsonlines.Writer(writer).write_all(all_objs)

    def save_outputs(self, logger: Run):
        if self.upload_predictions and is_world_process_zero():
            logger.save("*.jsonl", base_path=self.predictions_dir, policy="now")
