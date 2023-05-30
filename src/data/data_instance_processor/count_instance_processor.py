import logging
from typing import Optional, Dict, Any

from data.data_instance_processor.data_instance_processor import DataInstanceProcessor
from data.data_instance_processor.sum_instance_processor import (
    SumDataInstanceProcessor,
)

logger = logging.getLogger("app")


@DataInstanceProcessor.register("count")
class CountDataInstanceProcessor(SumDataInstanceProcessor):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        category = str(example["cat"])
        src_str = example["src_seq"]
        answer = example["answer"]

        values_str = f"Count: {src_str}"
        answer = float(answer)

        targets = answer
        return {
            "category": category,
            self.source_seq_key: values_str,
            self.target_seq_key: targets,
            "answer": answer,
        }

    def is_prediction_correct(self, prediction: int, answer: int) -> int:
        return abs(prediction - answer)


@DataInstanceProcessor.register("count_mod")
class CountModularDataInstanceProcessor(CountDataInstanceProcessor):
    def __init__(
        self,
        include_scratchpad: Optional[bool] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if include_scratchpad is not None:
            logger.warning(
                "The `include_scratchpad` argument is not used and will be removed in a future version."
            )
        self.include_scratchpad = include_scratchpad

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        category = str(example["cat"])
        src_str = example["src_seq"]
        answer = example["answer"]

        values_str = f"Count: {src_str}"
        answer = int(answer)

        targets = answer
        return {
            "category": category,
            self.source_seq_key: values_str,
            self.target_seq_key: targets,
            "answer": answer,
        }

    def extract_answer_from_prediction(self, prediction: str) -> Any:
        raise NotImplementedError()

    def is_prediction_correct(self, prediction: int, data_instance: Dict[str, Any]) -> int:
        answer = data_instance["answer"]
        return int(prediction == answer)
