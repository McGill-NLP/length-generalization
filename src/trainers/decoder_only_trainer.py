from typing import Dict, Union, Optional, List, Tuple, Any

import datasets
import torch
from datasets import Dataset
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import IntervalStrategy, is_datasets_available
from transformers.deepspeed import is_deepspeed_zero3_enabled
from transformers.trainer_pt_utils import IterableDatasetShard

from trainers.base_trainer import BaseTrainer
from trainers.trainer_with_metrics import Seq2SeqTrainerWithMetrics

import logging

logger = logging.getLogger("app")


@BaseTrainer.register("decoder_only")
class DecoderOnlyTrainer(Seq2SeqTrainerWithMetrics):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        should_log = (
            self.args.logging_strategy == IntervalStrategy.STEPS
            and self.state.global_step % self.args.logging_steps == 0
        )
        if should_log and isinstance(outputs, dict):
            metrics = outputs.get("metrics", {})
            if "logits" in outputs and (
                "labels" in inputs or "merged_input" in outputs
            ):
                metric_labels = (
                    outputs["merged_input"]["labels"]
                    if "merged_input" in outputs
                    else inputs["labels"]
                )
                metric_logits = outputs["logits"]
                if not self.model.config.is_encoder_decoder:
                    metric_logits = metric_logits[..., :-1, :]
                    metric_labels = metric_labels[..., 1:]

                computed_metrics = self._compute_metrics_during_training(
                    metric_logits, metric_labels
                )
                metrics.update(computed_metrics)

            for k, v in metrics.items():
                if isinstance(v, torch.Tensor):
                    v = v.clone().detach().cpu().numpy().tolist()
                    if isinstance(v, (list, tuple)):
                        v = v[0]
                    metrics[k] = v

            self.log(metrics)

        if labels is not None:
            loss = self.label_smoother(outputs, labels)
        else:
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def evaluate(
        self,
        eval_dataset: Optional[Dataset] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
    ) -> Dict[str, float]:
        if (
            getattr(self, "state", None) is not None
            and getattr(self.state, "epoch", None) is not None
            and self.state.epoch <= 3
        ):
            _max_length = self.args.generation_max_length if max_length is None else max_length
            max_length = min(_max_length, 256)
            logger.info(f"Reducing max_length to {max_length} for early epochs")
            self.log({"gen_max_length": max_length})
        else:
            if max_length is None:
                self.log({"gen_max_length": self.args.generation_max_length})

        return super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
            max_length=max_length,
            num_beams=num_beams,
        )

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.

        Return:
            Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
            labels (each being optional).
        """

        if not self.args.predict_with_generate or prediction_loss_only:
            return super().prediction_step(
                model,
                inputs,
                prediction_loss_only=prediction_loss_only,
                ignore_keys=ignore_keys,
            )

        has_labels = "labels" in inputs
        inputs = self._prepare_inputs(inputs)

        max_length = (
            self._max_length
            if self._max_length is not None
            else self.model.config.max_length
        )

        # XXX: adapt synced_gpus for fairscale as well
        gen_kwargs = {
            "max_length": max_length,
            "num_beams": self._num_beams
            if self._num_beams is not None
            else self.model.config.num_beams,
            "synced_gpus": True if is_deepspeed_zero3_enabled() else False,
        }

        # prepare generation inputs
        # some encoder-decoder models can have varying encder's and thus
        # varying model input names
        if (
            hasattr(self.model, "encoder")
            and self.model.encoder.main_input_name != self.model.main_input_name
        ):
            generation_inputs = inputs[self.model.encoder.main_input_name]
        else:
            generation_inputs = inputs[self.model.main_input_name]

        generated_tokens = self.model.generate(
            generation_inputs,
            attention_mask=inputs.get("attention_mask", None),
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **gen_kwargs,
        )
        # in case the batch is shorter than max length, the output should be padded
        if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
            generated_tokens = self._pad_tensors_to_max_len(
                generated_tokens, gen_kwargs["max_length"]
            )

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                if inputs["input_ids"].shape != inputs["labels"].shape:
                    prompt_input_ids = inputs["input_ids"]
                    labels_input_ids = inputs["labels"].clone().detach()

                    labels_input_ids.masked_fill_(labels_input_ids == -100, 0)
                    labels_input_ids = labels_input_ids[
                        :, prompt_input_ids.shape[1] :, ...
                    ]

                    new_input_ids = torch.cat(
                        [prompt_input_ids, labels_input_ids], dim=1
                    )
                    new_attn_mask = (new_input_ids != 0).int()
                    inputs["input_ids"] = new_input_ids
                    inputs["attention_mask"] = new_attn_mask

                    position_ids = new_attn_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(new_attn_mask == 0, 1)
                    inputs["position_ids"] = position_ids

                outputs = model(**inputs)

            if has_labels:
                if self.label_smoother is not None:
                    loss = (
                        self.label_smoother(outputs, inputs["labels"]).mean().detach()
                    )
                else:
                    loss = (
                        (outputs["loss"] if isinstance(outputs, dict) else outputs[0])
                        .mean()
                        .detach()
                    )
            else:
                loss = None

        if self.args.prediction_loss_only:
            return (loss, None, None)

        if has_labels:
            labels = inputs["labels"]
            if labels.shape[-1] < gen_kwargs["max_length"]:
                labels = self._pad_tensors_to_max_len(labels, gen_kwargs["max_length"])
        else:
            labels = None

        return (loss, generated_tokens, labels)

    def prediction_step_loss_per_example(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        ignore_keys: Optional[List[str]] = None,
    ) -> torch.Tensor:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            torch.Tensor: loss per example
        """

        has_labels = "labels" in inputs
        assert has_labels, "labels are required for loss_per_example"

        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            with self.autocast_smart_context_manager():
                if inputs["input_ids"].shape != inputs["labels"].shape:
                    prompt_input_ids = inputs["input_ids"]
                    labels_input_ids = inputs["labels"].clone().detach()

                    labels_input_ids.masked_fill_(labels_input_ids == -100, 0)
                    labels_input_ids = labels_input_ids[
                        :, prompt_input_ids.shape[1] :, ...
                    ]

                    new_input_ids = torch.cat(
                        [prompt_input_ids, labels_input_ids], dim=1
                    )
                    new_attn_mask = (new_input_ids != 0).int()
                    inputs["input_ids"] = new_input_ids
                    inputs["attention_mask"] = new_attn_mask

                    position_ids = new_attn_mask.long().cumsum(-1) - 1
                    position_ids.masked_fill_(new_attn_mask == 0, 1)
                    inputs["position_ids"] = position_ids

                outputs = model(**inputs)
                lm_logits = outputs.logits
                labels = inputs["labels"]

                # Compute loss in fp32 to match with mesh-tf version
                # https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
                lm_logits = lm_logits.to(torch.float32)

                # Shift so that tokens < n predict n
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = CrossEntropyLoss(reduction="none")
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                loss = loss.view(*shift_labels.size())

                per_example_loss = loss

        return per_example_loss

    def get_eval_dataloader(self, eval_dataset: Optional[datasets.Dataset] = None):
        """
        Returns the evaluation [`~torch.utils.data.DataLoader`].

        Subclass and override this method if you want to inject some custom behavior.

        Args:
            eval_dataset (`torch.utils.data.Dataset`, *optional*):
                If provided, will override `self.eval_dataset`. If it is an `datasets.Dataset`, columns not accepted by
                the `model.forward()` method are automatically removed. It must implement `__len__`.
        """
        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")
        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        data_collator = self.data_collator
        if getattr(self, "eval_data_collator", None) is not None:
            data_collator = self.eval_data_collator

        if is_datasets_available() and isinstance(eval_dataset, datasets.Dataset):
            eval_dataset = self._remove_unused_columns(
                eval_dataset, description="evaluation"
            )

        if isinstance(eval_dataset, torch.utils.data.IterableDataset):
            if self.args.world_size > 1:
                eval_dataset = IterableDatasetShard(
                    eval_dataset,
                    batch_size=self.args.per_device_eval_batch_size,
                    drop_last=self.args.dataloader_drop_last,
                    num_processes=self.args.world_size,
                    process_index=self.args.process_index,
                )
            return DataLoader(
                eval_dataset,
                batch_size=self.args.eval_batch_size,
                collate_fn=data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.eval_batch_size,
            collate_fn=data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
