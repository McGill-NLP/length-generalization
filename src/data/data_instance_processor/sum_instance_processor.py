import logging
import re
from typing import Optional, Dict, Any, List

from data.data_instance_processor.data_instance_processor import (
    DataInstanceProcessor,
    DataInstanceProcessorWithUnifiedScratchpad,
    number_to_string,
    UnifedScratchpadStep,
    REGEX_CACHE,
)

logger = logging.getLogger("app")


@DataInstanceProcessor.register("sum", exist_ok=True)
class SumDataInstanceProcessor(DataInstanceProcessor):
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
        values = example["values"]
        answer = example["answer"]

        values_str = " ".join([str(v) for v in values])
        answer = float(answer)

        targets = answer
        return {
            "category": category,
            self.source_seq_key: values_str,
            self.target_seq_key: targets,
            "answer": answer,
        }

    def extract_answer_from_prediction(self, prediction: str) -> Any:
        raise NotImplementedError()

    def extract_scratchpad_from_prediction(self, prediction: str) -> Optional[str]:
        return None

    def is_prediction_correct(self, prediction: int, answer: int) -> bool:
        return abs(prediction - answer)


@DataInstanceProcessor.register("sum_mod", exist_ok=True)
class SumModularDataInstanceProcessor(SumDataInstanceProcessor):
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
        values = example["values"]
        answer = example["answer"]

        values_str = " ".join([str(v) for v in values])
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

    def extract_scratchpad_from_prediction(self, prediction: str) -> Optional[str]:
        return None


@DataInstanceProcessor.register("s2s_sum", exist_ok=True)
class S2SSumDataInstanceProcessor(DataInstanceProcessorWithUnifiedScratchpad):
    def __init__(self, modulo_factor: int, stripped: bool = False, **kwargs):
        scratchpad_bos = kwargs.pop("scratchpad_bos", "Thinking step by step: ")
        scratchpad_eos = kwargs.pop("scratchpad_eos", ". Thus, ")
        step_separator = kwargs.pop("step_separator", " ; Next, ")
        input_marker = kwargs.pop("input_marker", "For number")
        computation_marker = kwargs.pop("computation_marker", "We have:")
        output_marker = kwargs.pop("output_marker", "==")
        intermediate_variables_marker_begin = kwargs.pop(
            "intermediate_variables_marker_begin", ". and currently,"
        )
        intermediate_variables_marker_end = kwargs.pop(
            "intermediate_variables_marker_end", "."
        )
        remaining_input_marker = kwargs.pop(
            "remaining_input_marker", "So, the rest is:"
        )
        super().__init__(
            scratchpad_bos=scratchpad_bos,
            scratchpad_eos=scratchpad_eos,
            step_separator=step_separator,
            input_marker=input_marker,
            computation_marker=computation_marker,
            output_marker=output_marker,
            intermediate_variables_marker_begin=intermediate_variables_marker_begin,
            intermediate_variables_marker_end=intermediate_variables_marker_end,
            remaining_input_marker=remaining_input_marker,
            **kwargs,
        )
        self.modulo_factor = modulo_factor
        self.stripped = stripped

    def _create_prompt(self, example: Dict[str, Any]) -> str:
        values = example["values"]
        values_seq_str = " + ".join([str(v) for v in values])
        prompt = f"Compute: {values_seq_str} ."
        return prompt

    def _create_answer(self, example: Dict[str, Any]) -> int:
        answer = example["answer"]
        return answer % self.modulo_factor

    def _create_targets(self, answer: Any) -> str:
        targets = f"The answer is {number_to_string(answer)} ."
        return targets

    def _get_category(self, example: Dict[str, Any]) -> str:
        category = str(example["cat"])
        return category

    def create_scratchpad_steps_from_data_instance(
        self, data_instance: Dict[str, Any]
    ) -> List[UnifedScratchpadStep]:
        values: List[int] = data_instance["values"]

        scratchpad_steps = []
        the_sum = 0
        for i in range(len(values)):
            curr_num = values[i]

            step_input = f"{number_to_string(curr_num)}"
            computation = (
                f"{number_to_string(curr_num)} + sum = "
                f"{number_to_string(curr_num)} + {number_to_string(the_sum)} ="
            )
            step_output = number_to_string(curr_num + the_sum)
            if not self.stripped:
                intermediate_sum_str = (
                    f"sum: "
                    f"{step_output} % {self.modulo_factor} = "
                    f"{(curr_num + the_sum) % self.modulo_factor}"
                )
            else:
                intermediate_sum_str = (
                    f"sum: "
                    f"{step_output} = "
                    f"{(curr_num + the_sum) % self.modulo_factor}"
                )
            the_sum = (curr_num + the_sum) % self.modulo_factor
            remaining_input = " + ".join([number_to_string(v) for v in values[i + 1 :]])

            step = UnifedScratchpadStep(
                input=step_input,
                computation=computation,
                output=step_output,
                intermediate_variables=[intermediate_sum_str],
                remaining_input=remaining_input,
            )
            scratchpad_steps.append(step)

        return scratchpad_steps

    def extract_answer_from_final_output(self, final_output: str) -> Any:
        # Final output format is "The answer is <answer>."
        # Use regex to extract the answer
        regex = REGEX_CACHE.get(
            self.__class__.__name__ + "_answer",
            re.compile(r".*The answer is\s+(-?\s?[\d\.][\d\s\.]*)\s*\."),
        )

        match = regex.search(final_output)

        def convert_numeric(s):
            try:
                return int(s)
            except ValueError:
                try:
                    return float(s)
                except ValueError:
                    return "Cannot convert string to numeric value"

        if match:
            try:
                # Extract the answer from the match
                answer_str = match.group(1)
                answer_str_no_space = "".join(answer_str.split())
                answer = convert_numeric(answer_str_no_space)
                return answer
            except Exception:
                pass

        return -100000

    def is_answer_correct(
        self, final_answer: Any, data_instance: Dict[str, Any]
    ) -> bool:
        assert isinstance(final_answer, int)
        answer = self._create_answer(data_instance)
        return final_answer == answer


if __name__ == "__main__":
    data_instance = {
        "values": [3, 3, 1, 8, 6, 1, 2, 1, 9, 8, 1],
        "answer": 43,
        "cat": 11,
    }
    processor = S2SSumDataInstanceProcessor(modulo_factor=10, include_scratchpad=True, stripped=True)
    scratchpad_steps = processor.create_scratchpad_steps_from_data_instance(
        data_instance
    )

    answer = processor._create_answer(data_instance)
    model_answer = processor.extract_answer_from_final_output(
        f"{processor._create_prompt(data_instance)} {processor._create_targets(answer)}"
    )
    assert processor.is_answer_correct(model_answer, data_instance)

    print(processor.create_human_readable_scratchpad(scratchpad_steps))
