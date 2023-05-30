from typing import Dict

import torch
from transformers import EvalPrediction, IntervalStrategy, Trainer, Seq2SeqTrainer
from transformers.trainer_pt_utils import nested_numpify

from trainers.base_trainer import BaseTrainer


class MetricsMixin(BaseTrainer):
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
        # TODO: this needs to be fixed and made cleaner later.
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
                computed_metrics = self._compute_metrics_during_training(
                    outputs["logits"], metric_labels
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

    def _compute_metrics_during_training(self, logits, labels) -> Dict[str, float]:
        logits = nested_numpify(logits.clone().detach())
        labels = nested_numpify(labels.clone().detach())

        metrics = self.compute_metrics(
            EvalPrediction(predictions=logits, label_ids=labels)
        )
        return metrics


@BaseTrainer.register("seq2seq_trainer_with_metrics")
class Seq2SeqTrainerWithMetrics(MetricsMixin, Seq2SeqTrainer, BaseTrainer):
    pass


@BaseTrainer.register("trainer_with_metrics")
class TrainerWithMetrics(MetricsMixin, Trainer, BaseTrainer):
    pass


BaseTrainer.default_implementation = "seq2seq_trainer_with_metrics"
