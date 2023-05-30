import re
from typing import Dict, Any, List

from data.data_instance_processor.data_instance_processor import (
    DataInstanceProcessor,
    DataInstanceProcessorWithUnifiedScratchpad,
    UnifedScratchpadStep,
    REGEX_CACHE,
)


@DataInstanceProcessor.register("s2s_parity", exist_ok=True)
class S2SParityDataInstanceProcessor(DataInstanceProcessorWithUnifiedScratchpad):
    NICE_VALUES = {
        "0": "Yes",
        "1": "No",
        "Yes": "Yes",
        "No": "No",
        "yes": "Yes",
        "no": "No",
    }

    def __init__(self, **kwargs):
        scratchpad_bos = kwargs.pop("scratchpad_bos", "Thinking step by step: ")
        scratchpad_eos = kwargs.pop("scratchpad_eos", ". Thus, ")
        step_separator = kwargs.pop("step_separator", " ; Next, ")
        input_marker = kwargs.pop("input_marker", "For bit")
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

    def _create_prompt(self, example: Dict[str, Any]) -> str:
        values = [str(v) for v in example["values"]]
        values_str = ", ".join(values)
        prompt = f"Is the number of 1's even in the sequence [ {values_str} ] ? "
        return prompt

    def _create_answer(self, example: Dict[str, Any]) -> str:
        answer = str(example["answer"])
        answer = self.NICE_VALUES[answer]
        return answer

    def _create_targets(self, answer: str) -> str:
        targets = f"The answer is {answer}."
        return targets

    def _get_category(self, example: Dict[str, Any]) -> str:
        category = str(example["cat"])
        return category

    def create_scratchpad_steps_from_data_instance(
        self, data_instance: Dict[str, Any]
    ) -> List[UnifedScratchpadStep]:
        values: List[int] = data_instance["values"]

        num_steps = len(values)

        curr_parity = False
        scratchpad_steps = []
        for i in range(num_steps):
            curr_bit = values[i]

            # XOR opr: var1 ^ var2
            # 0 ^ 0 = 0
            # 1 ^ 1 = 0
            # 0 ^ 1 = 1
            # 1 ^ 0 = 1

            step_input = f"{curr_bit}"
            computation = f"{curr_bit} ** par = {curr_bit} ** {int(curr_parity)}"
            step_output = f"{int(curr_bit ^ curr_parity)}"
            curr_parity = curr_bit ^ curr_parity
            intermediate_parity = f"par: {int(curr_parity)}"
            remaining_inputs = ", ".join([str(v) for v in values[i + 1 :]])
            remaining_inputs = f"[ {remaining_inputs} ]"

            step = UnifedScratchpadStep(
                input=step_input,
                computation=computation,
                output=step_output,
                intermediate_variables=[intermediate_parity],
                remaining_input=remaining_inputs,
            )
            scratchpad_steps.append(step)

        return scratchpad_steps

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
        gold_answer = self._create_answer(data_instance)
        return final_answer.lower() == gold_answer.lower()


if __name__ == "__main__":
    data_instance = {"values": [0, 0, 0, 0, 0, 1, 1, 1], "answer": 1, "cat": 8}
    processor = S2SParityDataInstanceProcessor(
        include_scratchpad=True,
        step_separator=" # ",
        input_marker="in",
        computation_marker="comp",
        output_marker="out",
        intermediate_variables_marker_begin="[",
        intermediate_variables_marker_end="]",
        remaining_input_marker="re",
        stripped=True,
    )
    scratchpad_steps = processor.create_scratchpad_steps_from_data_instance(
        data_instance
    )

    answer = processor._create_answer(data_instance)
    model_answer = processor.extract_answer_from_final_output(
        f"{processor._create_prompt(data_instance)} {processor._create_targets(answer)}"
    )
    assert processor.is_answer_correct(model_answer, data_instance)

    print(processor.create_human_readable_scratchpad(scratchpad_steps))
