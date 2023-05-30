import logging
import os
from collections import deque
from typing import Optional

import jsonlines
import numpy as np
from datasets import Dataset
from torch.utils.data import DataLoader
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

logger = logging.getLogger("app")


@Callback.register("cartography_measure")
class CartographyMeasureCallback(Callback):
    def __init__(
        self,
        upload_predictions: bool = False,
        label_index: Optional[int] = None,
        target_split: Optional[str] = "train",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.upload_predictions = upload_predictions
        self.label_index = label_index
        self.target_split = target_split
        if self.label_index is not None:
            assert self.label_index < 0, "label_index must be negative"

    def init(self, runtime, eval_dataset: Dataset, eval_split: str, **kwargs):
        super().init(runtime, eval_dataset, eval_split, **kwargs)

        from runtime.seq2seq_runtime import Seq2SeqRuntime

        runtime: Seq2SeqRuntime

        self.predictions_dir = (
            runtime.exp_root / f"cartography_measures_on_{self.target_split}"
        )
        self.predictions_dir.mkdir(exist_ok=True, parents=True)
        self.data_collator = runtime.dl_factory.get_collate_fn(
            state=ExperimentStage.TRAINING
        )
        self.dataset = runtime.dl_factory.get_dataset(
            stage=ExperimentStage.TRAINING,
            path=runtime.dl_factory.get_ds_file_path(
                stage=ExperimentStage.from_split(self.target_split)
            ),
        )

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
        if not hasattr(self._trainer, "prediction_step_loss_per_example"):
            logger.warning(
                "CartographyMeasureCallback requires `prediction_step_loss_per_example` to be set on the Trainer"
            )
            return

        if state.is_world_process_zero:
            sample_probs = deque()

            ds = self.dataset.remove_columns(
                [
                    c
                    for c in self.dataset.column_names
                    if c not in ["input_ids", "labels", "attention_mask"]
                ]
            )
            dataloader = DataLoader(
                ds,
                self._trainer.args.per_device_train_batch_size,
                collate_fn=self.data_collator,
                shuffle=False,
                pin_memory=True,
                num_workers=min(os.cpu_count(), 4),
            )
            model = self._trainer.model.eval()

            for batch in tqdm(dataloader, desc="Computing cartography measures"):
                batch_loss = self._trainer.prediction_step_loss_per_example(
                    model, batch
                )
                for i in range(len(batch_loss)):
                    loss = batch_loss[i]
                    attention_mask = batch["attention_mask"][i]
                    if self.label_index is not None:
                        # self.label_index is always negative. i.e. -1 for last token
                        sequence_length = attention_mask.sum()
                        # -1 here is because of the shifted nature of labels
                        # i.e.e labels = labels[1:]
                        target_index = sequence_length - 1 + self.label_index
                        loss = loss[target_index]
                    else:
                        loss = loss.sum()

                    try:
                        prob = np.exp(-loss.detach().cpu().numpy())
                    except OverflowError:
                        prob = 0.0

                    # check if prob is nan
                    if np.isnan(prob) or np.isinf(prob):
                        prob = 0.0

                    sample_probs.append(float(prob))

            assert len(sample_probs) == len(self.dataset)

            output_test_preds_file = (
                self.predictions_dir
                / f"{self._log_counts}_epoch-{str(state.epoch).zfill(5)}_step-{str(state.global_step).zfill(6)}.jsonl"
            )

            with output_test_preds_file.open("w") as writer:
                all_objs = []
                for i, prob in enumerate(sample_probs):
                    all_objs.append({"prob": prob, "id": i})

                jsonlines.Writer(writer).write_all(all_objs)

        self._log_counts += 1

    def save_outputs(self, logger: Run):
        if self.upload_predictions:
            logger.save("*.jsonl", base_path=self.predictions_dir, policy="now")
