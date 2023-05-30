import re
from typing import Dict, Any, List

from data.data_instance_processor.data_instance_processor import (
    DataInstanceProcessor,
    DataInstanceProcessorWithUnifiedScratchpad,
    UnifedScratchpadStep,
    REGEX_CACHE,
)


@DataInstanceProcessor.register("clutrr", exist_ok=True)
class S2SClutrrDataInstanceProcessor(DataInstanceProcessorWithUnifiedScratchpad):
    def __init__(self, **kwargs):
        scratchpad_bos = kwargs.pop("scratchpad_bos", "Thinking step by step: ")
        scratchpad_eos = kwargs.pop("scratchpad_eos", ". Thus, ")
        step_separator = kwargs.pop("step_separator", " ; Next, ")
        computation_marker = kwargs.pop("computation_marker", "comp")

        if "include_input" not in kwargs:
            kwargs["include_input"] = False

        if "include_output" not in kwargs:
            kwargs["include_output"] = False

        if "include_remaining_input" not in kwargs:
            kwargs["include_remaining_input"] = False

        if "include_intermediate_variables" not in kwargs:
            kwargs["include_intermediate_variables"] = False

        forced_scratchpad_options_to_false = [
            "include_input",
            "include_output",
            "include_remaining_input",
            "include_intermediate_variables",
        ]
        for option in forced_scratchpad_options_to_false:
            if kwargs.get(option, False):
                raise ValueError(
                    f"Option {option} must be set to False for this data instance processor."
                )

        super().__init__(
            scratchpad_bos=scratchpad_bos,
            scratchpad_eos=scratchpad_eos,
            step_separator=step_separator,
            computation_marker=computation_marker,
            **kwargs,
        )

    def _create_prompt(self, example: Dict[str, Any]) -> str:
        prompt = example[self.source_seq_key]
        prompt = f"Solve: {prompt}"
        return prompt

    def _create_answer(self, example: Dict[str, Any]) -> str:
        answer = example[self.target_seq_key]
        return answer

    def _create_targets(self, answer: str) -> str:
        targets = f"The answer is {answer}."
        return targets

    def _get_category(self, example: Dict[str, Any]) -> str:
        category = str(example["category"])
        return category

    def create_scratchpad_steps_from_data_instance(
        self, data_instance: Dict[str, Any]
    ) -> List[UnifedScratchpadStep]:
        predefined_scratchpad_steps = data_instance["scratchpad"]
        steps = []
        for raw_step in predefined_scratchpad_steps:
            step = UnifedScratchpadStep(computation=raw_step)
            steps.append(step)

        return steps

    def extract_answer_from_final_output(self, final_output: str) -> str:
        # Final output format is "The answer is Yes/No."
        regex = REGEX_CACHE.get(
            self.__class__.__name__ + "_answer",
            re.compile(r".*The answer is\s*(.+)\s*\."),
        )

        match = regex.search(final_output)

        if match:
            try:
                # Extract the answer from the match
                answer = match.group(1)
                answer = answer.strip()
                return answer
            except Exception:
                pass

        return ""

    def is_answer_correct(
        self, final_answer: str, data_instance: Dict[str, Any]
    ) -> bool:
        assert isinstance(final_answer, str)
        gold_answer = data_instance.get("answer", None)
        if gold_answer is None:
            gold_answer = self._create_answer(data_instance)
        return final_answer.lower() == gold_answer.lower()


if __name__ == "__main__":
    data_instance = {
        "source": "ent_15 is the daughter of ent_12 . ent_16 is ent_13 's brother . ent_13 is a grandson to ent_8 . ent_12 is the mother of ent_16 . [query] How are ent_8 and ent_15 related to each other ?",
        "target": "ent_8 has a granddaughter who is ent_15",
        "scratchpad": [
            "since ent_15 is a sister to ent_16 , and ent_16 is the grandson of ent_8 , then ent_15 is a granddaughter of ent_8",
            "since ent_15 is a daughter to ent_12 , and ent_12 is a mother to ent_16 , then ent_15 is the sister of ent_16",
            "since ent_16 is the brother of ent_13 , and ent_13 is the grandson of ent_8 , then ent_16 is a grandson to ent_8",
        ],
        "category": "4",
    }
    processor = S2SClutrrDataInstanceProcessor(include_scratchpad=True)
    scratchpad_steps = processor.create_scratchpad_steps_from_data_instance(
        data_instance
    )

    answer = processor._create_answer(data_instance)
    model_answer = processor.extract_answer_from_final_output(
        f"{processor._create_prompt(data_instance)} {processor._create_targets(answer)}"
    )
    assert processor.is_answer_correct(model_answer, data_instance)

    print(processor(data_instance)["source"])
    print(processor(data_instance)["target"])
    print(answer)

    print(processor.create_human_readable_scratchpad(scratchpad_steps))
