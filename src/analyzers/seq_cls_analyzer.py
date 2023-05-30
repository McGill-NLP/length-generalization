import copy
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Deque

import jsonlines
import wandb
from tqdm import tqdm

from analyzers import Analyzer
from common import ExperimentStage
from data import SequenceClassificationDataLoaderFactory

logger = logging.getLogger("app")


@Analyzer.register("seq_cls")
class SeqClsAnalyzer(Analyzer):
    dl_factory: SequenceClassificationDataLoaderFactory

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(self.dl_factory, SequenceClassificationDataLoaderFactory)

    def analyze(self):
        predictions_path = self.exp_root / f"pred_out_{self.split}.jsonl"
        if predictions_path.exists():
            evaluation_table, accuracies, distances = self._analyze_prediction(
                predictions_path
            )
            self.logger.log({f"evaluated_acc/{self.split}/table": evaluation_table})

            self.log_accuracies_and_distances(
                copy.deepcopy(accuracies), copy.deepcopy(distances)
            )

        self.analyze_all_evaluation_steps()

    def analyze_all_evaluation_steps(self):
        predictions_dir = self.exp_root / f"eval_on_{self.split}_predictions"
        if not predictions_dir.exists():
            return

        prediction_files = list(predictions_dir.glob("*.jsonl"))
        prediction_files.sort(key=lambda x: int(x.stem.split("_")[0]))

        aggregated_accuracies = dict()

        for prediction_file in prediction_files:
            step = int(prediction_file.stem.split("_")[-1].split("step-")[-1])
            _, accuracies, _ = self._analyze_prediction(prediction_file)
            accuracies: Dict[str, Deque]
            avg_accuracies = {
                k: round(sum(v) / len(v), 4) for k, v in accuracies.items()
            }
            aggregated_accuracies[step] = avg_accuracies

        steps = list(aggregated_accuracies.keys())
        steps.sort()

        # List average accuracies over sorted steps for each accuracy key
        avg_accuracies = defaultdict(list)
        for step in steps:
            accuracies = aggregated_accuracies[step]
            for k, v in accuracies.items():
                avg_accuracies[k].append(v)
        categories = list(avg_accuracies.keys())
        categories.sort()

        plot = wandb.plot.line_series(
            steps,
            [avg_accuracies[k] for k in categories],
            keys=categories,
            title="Average Accuracy over Evaluation Steps",
            xname="Global Step",
        )
        self.logger.log({f"evaluated_acc/{self.split}/during_training": plot})

    def _analyze_prediction(self, predictions_path: Path):

        assert (
            predictions_path.exists()
        ), f"Prediction file not found: {predictions_path}"

        pred_objs = []
        with jsonlines.open(str(predictions_path)) as reader:
            for obj in reader:
                pred_objs.append(obj)

        ds_path = self.dl_factory.get_ds_file_path(
            ExperimentStage.from_split(self.split)
        )
        logger.info(f"Evaluating against split: {self.split} at {ds_path}")

        dataset_objs = self.dl_factory.get_dataset(
            ExperimentStage.PREDICTION, path=ds_path
        )
        assert len(dataset_objs) == len(pred_objs)

        accuracies: Dict[str, Deque] = defaultdict(deque)
        distances: Dict[str, Deque] = defaultdict(deque)

        evaluation_table = wandb.Table(
            columns=[
                "idx",
                "prediction",
                "gold_answer",
                "is_correct",
                "edit_distance",
                "parse_error",
            ]
        )
        for idx, (pred_obj, ds_obj) in tqdm(
            enumerate(zip(pred_objs, dataset_objs)), total=len(pred_objs)
        ):
            category = ds_obj["category"]
            prediction = pred_obj["prediction"]

            if hasattr(self.dl_factory.instance_processor, "_create_answer"):
                gold_answer = self.dl_factory.instance_processor._create_answer(ds_obj)
            else:
                gold_answer = ds_obj["answer"]

            try:
                parsed_pred = prediction
                ed = -100

                is_correct = self.dl_factory.instance_processor.is_prediction_correct(
                    parsed_pred, ds_obj
                )
                exp_str = ""
            except Exception as exp:
                logger.warning(f"Couldn't parse the model's prediction {exp}")
                is_correct = False
                exp_str = str(exp)
                parsed_pred = prediction
                ed = 100

            accuracies[category].append(is_correct)
            distances[category].append(ed)

            evaluation_table.add_data(
                idx, parsed_pred, gold_answer, is_correct, ed, exp_str
            )

        return evaluation_table, accuracies, distances

    def log_accuracies_and_distances(self, accuracies, distances, prefix: str = ""):
        stats = []
        for key, acc_lst in accuracies.items():
            acc = sum(acc_lst) / len(acc_lst)
            acc = round(acc, 4)
            stats.append((f"{key}", acc))
            self.logger.log({f"pred/{self.split}_{prefix}acc_{key}": acc})

        all_predictions = [
            is_correct for acc_lst in accuracies.values() for is_correct in acc_lst
        ]
        overall_acc = sum(all_predictions) / len(all_predictions)
        overall_acc = round(overall_acc, 4)
        stats.append(("overall", overall_acc))
        self.logger.log({f"pred/{self.split}_{prefix}acc_overall": overall_acc})
        plot = wandb.plot.bar(
            wandb.Table(data=stats, columns=["split", "eAcc"]),
            label="split",
            value="eAcc",
            title=f"Evaluated {prefix} accuracy in split: {self.split}",
        )
        self.logger.log({f"evaluated_acc/{self.split}/{prefix}plot": plot})
        stats = []
        for key, dist_lst in distances.items():
            dist = sum(dist_lst) / len(dist_lst)
            dist = round(dist, 4)
            stats.append((f"{key}", dist))
            self.logger.log({f"pred/{self.split}_{prefix}editDistance_{key}": dist})
        distances = [dist for dist_lst in distances.values() for dist in dist_lst]
        overall_dist = sum(distances) / len(distances)
        overall_dist = round(overall_dist, 4)
        stats.append(("overall", overall_dist))
        self.logger.log(
            {f"pred/{self.split}_{prefix}editDistance_overall": overall_dist}
        )
        plot = wandb.plot.bar(
            wandb.Table(data=stats, columns=["split", "edist"]),
            label="split",
            value="edist",
            title=f"Evaluated {prefix} Edit Distance (editDistance) in split: {self.split}",
        )
        self.logger.log({f"evaluated_acc/{self.split}/{prefix}editDistance_plot": plot})
