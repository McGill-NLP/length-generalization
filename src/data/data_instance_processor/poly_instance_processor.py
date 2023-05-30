import re
from typing import Dict, Any, List

from data.data_instance_processor import UnifedScratchpadStep
from data.data_instance_processor.data_instance_processor import (
    DataInstanceProcessor,
    DataInstanceProcessorWithUnifiedScratchpad,
    REGEX_CACHE,
    number_to_string,
)


@DataInstanceProcessor.register("s2s_poly", exist_ok=True)
class S2SPolynomialDataInstanceProcessor(DataInstanceProcessorWithUnifiedScratchpad):
    def __init__(self, modulo_factor: int, stripped: bool = False, **kwargs):
        scratchpad_bos = kwargs.pop("scratchpad_bos", "Thinking step by step: ")
        scratchpad_eos = kwargs.pop("scratchpad_eos", ". Thus, ")
        step_separator = kwargs.pop("step_separator", " ; Next, ")
        input_marker = kwargs.pop("input_marker", "For term")
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
        coeffs: List[int] = example["coeffs"]
        degrees: List[int] = example["degrees"]
        x: int = example["x"]

        term_strs = [
            f"{number_to_string(coeff)} * x ** {number_to_string(degree)}"
            for coeff, degree in zip(coeffs, degrees)
        ]
        poly_str = " + ".join(term_strs)

        prompt = f"Evaluate at x = {number_to_string(x)} in {poly_str} ? "
        return prompt

    def _create_answer(self, example: Dict[str, Any]) -> Any:
        answer: int = example["answer"]
        answer = answer % self.modulo_factor
        return answer

    def _create_targets(self, answer: Any) -> str:
        targets = f"The answer is {number_to_string(answer)}."
        return targets

    def _get_category(self, example: Dict[str, Any]) -> str:
        category = str(example["cat"])
        return category

    def create_scratchpad_steps_from_data_instance(
        self, data_instance: Dict[str, Any]
    ) -> List[UnifedScratchpadStep]:
        coeffs: List[int] = data_instance["coeffs"]
        degrees: List[int] = data_instance["degrees"]
        x: int = data_instance["x"]

        scratchpad_steps = []
        curr_sum = 0
        for i in range(len(coeffs)):
            coeff = coeffs[i]
            degree = degrees[i]

            step_input = f"{number_to_string(coeff)} * x ** {number_to_string(degree)}"
            computation = (
                f"{number_to_string(coeff)} * {number_to_string(x)} ** {number_to_string(degree)} = "
                f"{number_to_string(coeff)} * {number_to_string(x ** degree)}"
            )
            step_output = number_to_string(coeff * x**degree)
            if not self.stripped:
                intermediate_sum_str = (
                    f"sum: "
                    f"({number_to_string(curr_sum)} + {step_output}) % {self.modulo_factor} = "
                    f"{number_to_string((curr_sum + coeff * x ** degree) % self.modulo_factor)}"
                )
            else:
                intermediate_sum_str = (
                    f"sum: "
                    f"{number_to_string(curr_sum)} + {step_output} = "
                    f"{number_to_string((curr_sum + coeff * x ** degree) % self.modulo_factor)}"
                )
            curr_sum = (curr_sum + coeff * x**degree) % self.modulo_factor
            remaining_input = " + ".join(
                [
                    f"{number_to_string(coeff)} * x ** {number_to_string(degree)}"
                    for coeff, degree in zip(coeffs[i + 1 :], degrees[i + 1 :])
                ]
            )

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
        self, final_answer: Any, data_instance: Dict[str, Any]
    ) -> bool:
        assert isinstance(final_answer, int)
        answer = self._create_answer(data_instance)
        return final_answer == answer


if __name__ == "__main__":
    data_instance = {
        "coeffs": [-2, -1, -3, -1, -3, -3],
        "degrees": [2, 0, 3, 1, 0, 2],
        "x": 1,
        "answer": -13,
        "cat": 6,
    }
    processor = S2SPolynomialDataInstanceProcessor(
        modulo_factor=10, include_scratchpad=True,
        **{
            "step_separator": ' # ',
            "input_marker": 'in',
            "computation_marker": 'comp',
            "output_marker": 'out',
            "intermediate_variables_marker_begin": '[',
            "intermediate_variables_marker_end": ']',
            "remaining_input_marker": 're',
            "stripped": True,
        }
    )
    scratchpad_steps = processor.create_scratchpad_steps_from_data_instance(
        data_instance
    )

    o = processor(data_instance)
    print(o)
    answer = processor._create_answer(data_instance)
    inp_out = f"{o['source']} {o['target']}"
    print(inp_out)
    model_answer = processor.extract_answer_from_final_output(inp_out)
    assert processor.is_answer_correct(model_answer, data_instance)

    print(processor.create_human_readable_scratchpad(scratchpad_steps))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
    encoding = tokenizer(inp_out, add_special_tokens=False)
    replicated_seq = tokenizer.decode(encoding["input_ids"], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert inp_out == inp_out

    processor.get_tokenization_info(data_instance,
                                    encoding,
                                    inp_out)
