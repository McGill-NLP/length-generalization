import re
from typing import Dict, Any, List

from data.data_instance_processor import UnifedScratchpadStep
from data.data_instance_processor.data_instance_processor import (
    DataInstanceProcessor,
    DataInstanceProcessorWithUnifiedScratchpad,
    REGEX_CACHE,
    number_to_string,
)


@DataInstanceProcessor.register("s2s_sort_single_digit", exist_ok=True)
class S2SSortDataInstanceProcessor(DataInstanceProcessorWithUnifiedScratchpad):
    def __init__(self, **kwargs):
        scratchpad_bos = kwargs.pop("scratchpad_bos", "Thinking step by step: ")
        scratchpad_eos = kwargs.pop("scratchpad_eos", ". Thus, ")
        step_separator = kwargs.pop("step_separator", " ; Next, ")
        input_marker = kwargs.pop("input_marker", "For")
        computation_marker = kwargs.pop("computation_marker", "We have:")
        output_marker = kwargs.pop("output_marker", "is:")
        intermediate_variables_marker_begin = kwargs.pop(
            "intermediate_variables_marker_begin", ". and currently,"
        )
        intermediate_variables_marker_end = kwargs.pop(
            "intermediate_variables_marker_end", "."
        )
        remaining_input_marker = kwargs.pop(
            "remaining_input_marker", "So, the unsorted portion is:"
        )

        if "include_intermediate_variables" not in kwargs:
            kwargs["include_intermediate_variables"] = False

        if kwargs["include_intermediate_variables"] == True:
            raise ValueError(
                "include_intermediate_variables must be False for this task."
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
        src_seq: List[str] = example["src_seq"]
        src_seq_str = ", ".join(src_seq)
        prompt = f"Sort the following list [ {src_seq_str} ]. "
        return prompt

    def _create_answer(self, example: Dict[str, Any]) -> List[str]:
        answer: List[str] = example["tgt_seq"]
        return answer

    def _create_targets(self, answer: List[str]) -> str:
        tgt_seq_str = ", ".join(answer)
        targets = f"The sorted list is [ {tgt_seq_str} ]."
        return targets

    def _get_category(self, example: Dict[str, Any]) -> str:
        category = str(example["cat"])
        return category

    def create_scratchpad_steps_from_data_instance(
        self, data_instance: Dict[str, Any]
    ) -> List[UnifedScratchpadStep]:
        # Implements the insertion sort algorithm.
        # At each step, we take the minimum element and insert it at the front.
        token_ranks: List[int] = data_instance["token_ids"].copy()
        unsorted_seq: List[str] = data_instance["src_seq"].copy()

        num_steps = len(token_ranks)

        scratchpad_steps = []
        for i in range(num_steps):
            current_token = unsorted_seq[i]

            min_index, min_rank = min(
                [(j, token_ranks[j]) for j in range(i, num_steps)], key=lambda x: x[1]
            )
            the_next_min_token = unsorted_seq[min_index]

            # Swap the current token with the next minimum
            if min_index != i:
                remaining_input = unsorted_seq[i + 1 :]
                remaining_input[min_index - i - 1] = current_token
                token_ranks[i], token_ranks[min_index] = (
                    token_ranks[min_index],
                    token_ranks[i],
                )
                unsorted_seq[i], unsorted_seq[min_index] = (
                    unsorted_seq[min_index],
                    unsorted_seq[i],
                )
            else:
                remaining_input = unsorted_seq[i + 1 :]

            step_input = f"{current_token}"
            computation = f"The minimum of the unsorted portion"
            step_output = str(the_next_min_token)
            remaining_input = ", ".join(remaining_input)
            remaining_input = f"[ {remaining_input} ]"

            step = UnifedScratchpadStep(
                input=step_input,
                computation=computation,
                output=step_output,
                remaining_input=remaining_input,
            )
            scratchpad_steps.append(step)

        return scratchpad_steps

    def extract_answer_from_final_output(self, final_output: str) -> List[str]:
        # Final output format is "The sorted list is [ aa, bb, cc, ... ]."
        regex = REGEX_CACHE.get(
            self.__class__.__name__ + "_answer",
            re.compile(r".*The sorted list is\s*\[\s*(.*)\s*\]\s*\.?"),
        )

        match = regex.search(final_output)

        if match:
            try:
                # Extract the answer from the match
                inside_bracket_expr = match.group(1)
                tokens = inside_bracket_expr.split(",")
                tokens = [t.strip() for t in tokens]
                return tokens
            except Exception:
                pass

        return []

    def is_answer_correct(
        self, final_answer: List[str], data_instance: Dict[str, Any]
    ) -> bool:
        assert isinstance(final_answer, list)
        gold_answer = self._create_answer(data_instance)
        return final_answer == gold_answer


@DataInstanceProcessor.register("s2s_sort_multi_digit", exist_ok=True)
class S2SSortMultiDigitDataInstanceProcessor(
    DataInstanceProcessorWithUnifiedScratchpad
):
    def __init__(self, **kwargs):
        scratchpad_bos = kwargs.pop("scratchpad_bos", "Thinking step by step: ")
        scratchpad_eos = kwargs.pop("scratchpad_eos", ". Thus, ")
        step_separator = kwargs.pop("step_separator", " ; Next, ")
        input_marker = kwargs.pop("input_marker", "For")
        computation_marker = kwargs.pop("computation_marker", "We have:")
        output_marker = kwargs.pop("output_marker", "is:")
        intermediate_variables_marker_begin = kwargs.pop(
            "intermediate_variables_marker_begin", ". and currently,"
        )
        intermediate_variables_marker_end = kwargs.pop(
            "intermediate_variables_marker_end", "."
        )
        remaining_input_marker = kwargs.pop(
            "remaining_input_marker", "So, the unsorted portion is:"
        )

        if "include_intermediate_variables" not in kwargs:
            kwargs["include_intermediate_variables"] = False

        if kwargs["include_intermediate_variables"] == True:
            raise ValueError(
                "include_intermediate_variables must be False for this task."
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
        src_seq: List[int] = example["src_seq"]
        src_seq_str = ", ".join([number_to_string(n) for n in src_seq])
        prompt = f"Sort the following list [ {src_seq_str} ]. "
        return prompt

    def _create_answer(self, example: Dict[str, Any]) -> List[int]:
        answer: List[int] = example["tgt_seq"]
        return answer

    def _create_targets(self, answer: List[int]) -> str:
        tgt_seq_str = ", ".join([number_to_string(n) for n in answer])
        targets = f"The sorted list is [ {tgt_seq_str} ]."
        return targets

    def _get_category(self, example: Dict[str, Any]) -> str:
        category = str(example["cat"])
        return category

    def create_scratchpad_steps_from_data_instance(
        self, data_instance: Dict[str, Any]
    ) -> List[UnifedScratchpadStep]:
        # Implements the insertion sort algorithm.
        # At each step, we take the minimum element and insert it at the front.
        unsorted_seq: List[int] = data_instance["src_seq"].copy()

        num_steps = len(unsorted_seq)

        scratchpad_steps = []
        for i in range(num_steps):
            list_front = unsorted_seq[i]

            min_index, min_rank = min(
                [(j, unsorted_seq[j]) for j in range(i, num_steps)], key=lambda x: x[1]
            )
            the_next_min_token = unsorted_seq[min_index]

            # Swap the current token with the next minimum
            if min_index != i:
                remaining_input = unsorted_seq[i + 1 :]
                remaining_input[min_index - i - 1] = list_front
                unsorted_seq[i], unsorted_seq[min_index] = (
                    unsorted_seq[min_index],
                    unsorted_seq[i],
                )
            else:
                remaining_input = unsorted_seq[i + 1 :]

            step_input = f"{number_to_string(list_front)}"
            computation = f"The minimum of the unsorted portion"
            step_output = number_to_string(the_next_min_token)
            remaining_input = ", ".join([number_to_string(n) for n in remaining_input])
            remaining_input = f"[ {remaining_input} ]"

            step = UnifedScratchpadStep(
                input=step_input,
                computation=computation,
                output=step_output,
                remaining_input=remaining_input,
            )
            scratchpad_steps.append(step)

        return scratchpad_steps

    def extract_answer_from_final_output(self, final_output: str) -> List[int]:
        # Final output format is "The sorted list is [ aa, bb, cc, ... ]."
        regex = REGEX_CACHE.get(
            self.__class__.__name__ + "_answer",
            re.compile(r".*The sorted list is\s*\[\s*(.*)\s*\]\s*\.?"),
        )

        match = regex.search(final_output)

        if match:
            try:
                # Extract the answer from the match
                inside_bracket_expr = match.group(1)
                tokens = inside_bracket_expr.split(",")
                tokens = [int("".join(t.strip().split())) for t in tokens]
                return tokens
            except Exception:
                pass

        return []

    def is_answer_correct(
        self, final_answer: List[int], data_instance: Dict[str, Any]
    ) -> bool:
        assert isinstance(final_answer, list)
        gold_answer = self._create_answer(data_instance)
        return final_answer == gold_answer


if __name__ == "__main__":
    data_instance = {
        "token_ids": [21, 13, 14, 14, 23],
        "src_seq": ["M", "E", "F", "F", "O"],
        "tgt_seq": ["E", "F", "F", "M", "O"],
        "cat": 4,
    }

    processor = S2SSortDataInstanceProcessor(
        include_intermediate_variables=False, include_scratchpad=True
    )
    scratchpad_steps = processor.create_scratchpad_steps_from_data_instance(
        data_instance
    )

    answer = processor._create_answer(data_instance)
    model_answer = processor.extract_answer_from_final_output(
        f"{processor._create_prompt(data_instance)} {processor._create_targets(answer)}")
    assert processor.is_answer_correct(model_answer, data_instance)

    print(processor.create_human_readable_scratchpad(scratchpad_steps))

    # data_instance = {
    #     "src_seq": [2468, 6863, 881, 4125],
    #     "tgt_seq": [881, 2468, 4125, 6863],
    #     "cat": 4,
    # }

    processor = S2SSortMultiDigitDataInstanceProcessor(
        include_intermediate_variables=False, include_scratchpad=True
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
