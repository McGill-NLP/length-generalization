import re
from typing import Dict, Any, List, Optional

from data.data_instance_processor import UnifedScratchpadStep
from data.data_instance_processor.data_instance_processor import (
    DataInstanceProcessor,
    DataInstanceProcessorWithUnifiedScratchpad,
    REGEX_CACHE,
    number_to_string,
)


@DataInstanceProcessor.register("s2s_addition", exist_ok=True)
class S2SAdditionDataInstanceProcessor(DataInstanceProcessorWithUnifiedScratchpad):
    def __init__(
        self,
        stripped: Optional[bool] = False,
        **kwargs,
    ):
        scratchpad_bos = kwargs.pop("scratchpad_bos", "Thinking step by step: ")
        scratchpad_eos = kwargs.pop("scratchpad_eos", ". Thus, ")
        step_separator = kwargs.pop("step_separator", " ; Next, ")
        input_marker = kwargs.pop("input_marker", "For digits")
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
        self.stripped = stripped

    def _create_prompt(self, example: Dict[str, Any]) -> str:
        num1: int = example["a"]
        num2: int = example["b"]

        prompt = f"Compute {number_to_string(num1)} + {number_to_string(num2)} ? "
        return prompt

    def _create_answer(self, example: Dict[str, Any]) -> int:
        answer: int = example["sum"]
        return answer

    def _create_targets(self, answer: int) -> str:
        targets = f"The answer is {number_to_string(answer)}."
        return targets

    def _get_category(self, example: Dict[str, Any]) -> str:
        category = str(example["cat"])
        return category

    def create_scratchpad_steps_from_data_instance(
        self, data_instance: Dict[str, Any]
    ) -> List[UnifedScratchpadStep]:
        num1: int = data_instance["a"]
        num2: int = data_instance["b"]

        num1_digits = [int(d) for d in str(num1)]
        num2_digits = [int(d) for d in str(num2)]

        # Pad the shorter number with zeros
        if len(num1_digits) > len(num2_digits):
            num2_digits = [0] * (len(num1_digits) - len(num2_digits)) + num2_digits
        elif len(num2_digits) > len(num1_digits):
            num1_digits = [0] * (len(num2_digits) - len(num1_digits)) + num1_digits

        # Add the numbers column by column
        carry = 0
        scratchpad_steps = []
        for i in range(len(num1_digits) - 1, -1, -1):
            a = num1_digits[i]
            b = num2_digits[i]
            sum = a + b + carry

            step_input = f"{a}, {b}"
            if not self.stripped:
                computation = (
                    f"( {a} + {b} + carry ) % 10 = ( {a+b} + {carry} ) % 10 = {sum} % 10"
                )
            else:
                computation = (
                    f"{a} + {b} + carry = {a + b} + {carry} = {sum}"
                )

            carry = sum // 10
            output = sum % 10

            step_output = f"{output}"
            if not self.stripped:
                intermediate_carry_str = f"carry: {sum} // 10 = {carry}"
            else:
                intermediate_carry_str = f"carry: {carry}"

            remaining_a_str = "".join([str(d) for d in num1_digits[:i]])
            remaining_a = number_to_string(int(remaining_a_str)) if remaining_a_str else ""

            remaining_b_str = "".join([str(d) for d in num2_digits[:i]])
            remaining_b = number_to_string(int(remaining_b_str)) if remaining_b_str else ""

            remaining_digits_str = f"{remaining_a} + {remaining_b}"

            step = UnifedScratchpadStep(
                input=step_input,
                computation=computation,
                output=step_output,
                intermediate_variables=[intermediate_carry_str],
                remaining_input=remaining_digits_str,
            )
            scratchpad_steps.append(step)

        # Add the carry if there is one
        if carry > 0:
            step_input = "0, 0"
            step_output = f"{carry}"
            if not self.stripped:
                computation = f"( 0 + 0 + carry ) % 10 = ( 0 + {carry} ) % 10 = {carry} % 10"
                intermediate_carry_str = f"carry: {carry} // 10 = 0"
            else:
                computation = f"0 + 0 + carry = 0 + {carry} = {carry}"
                intermediate_carry_str = f"carry: 0"
            remaining_digits_str = f" + "
            step = UnifedScratchpadStep(
                input=step_input,
                computation=computation,
                output=step_output,
                intermediate_variables=[intermediate_carry_str],
                remaining_input=remaining_digits_str,
            )
            scratchpad_steps.append(step)

        return scratchpad_steps

    def extract_answer_from_final_output(self, final_output: str) -> Any:
        # Final output format is "The answer is <answer>."
        # Use regex to extract the answer
        regex = REGEX_CACHE.get(
            self.__class__.__name__ + "_answer",
            re.compile(r".*The answer is\s+(-?\s?[\d\.][\d\s\.]*)\."),
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
        self, final_answer: int, data_instance: Dict[str, Any]
    ) -> bool:
        assert isinstance(final_answer, int)
        answer = self._create_answer(data_instance)
        return final_answer == answer


if __name__ == "__main__":
    # data_instance = {
    #     "token_ids": [21, 13, 14, 14, 23],
    #     "src_seq": ["M", "E", "F", "F", "O"],
    #     "tgt_seq": ["E", "F", "F", "M", "O"],
    #     "cat": 4,
    # }
    #
    # processor = S2SSortDataInstanceProcessor(
    #     include_intermediate_variables=False, include_scratchpad=True
    # )
    # scratchpad_steps = processor.create_scratchpad_steps_from_data_instance(
    #     data_instance
    # )
    #
    # answer = processor._create_answer(data_instance)
    # model_answer = processor.extract_answer_from_final_output(
    #     f"{processor._create_prompt(data_instance)} {processor._create_targets(answer)}")
    # assert processor.is_answer_correct(model_answer, data_instance)
    #
    # print(processor.create_human_readable_scratchpad(scratchpad_steps))

    data_instance = {"a": 53726, "b": 53726, "sum": 107452, "cat": "5-by-5"}

    processor = S2SAdditionDataInstanceProcessor(include_scratchpad=True)
    scratchpad_steps = processor.create_scratchpad_steps_from_data_instance(
        data_instance
    )
    print(processor.create_human_readable_scratchpad(scratchpad_steps))

    answer = processor._create_answer(data_instance)
    model_answer = processor.extract_answer_from_final_output(
        f"{processor._create_prompt(data_instance)} {processor._create_targets(answer)}"
    )
    assert processor.is_answer_correct(model_answer, data_instance)
    print(model_answer)


