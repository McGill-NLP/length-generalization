import re
from typing import Dict, Any, List

from data.data_instance_processor.data_instance_processor import (
    DataInstanceProcessor,
    DataInstanceProcessorWithUnifiedScratchpad,
    UnifedScratchpadStep,
    REGEX_CACHE,
)


@DataInstanceProcessor.register("s2s_lego", exist_ok=True)
class S2SLegoDataInstanceProcessor(DataInstanceProcessorWithUnifiedScratchpad):
    NICE_VALUES = {"0": "-1", "1": "+1"}

    def __init__(self, **kwargs):
        scratchpad_bos = kwargs.pop("scratchpad_bos", "Thinking step by step: ")
        scratchpad_eos = kwargs.pop("scratchpad_eos", ". Thus, ")
        step_separator = kwargs.pop("step_separator", " ; Next, ")
        input_marker = kwargs.pop("input_marker", "For variable")
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
        query = example["query"]
        clauses = example["clauses"]

        clauses = ", ".join(clauses)
        prompt = f"If {clauses}. Then, what is the value of {query}? "
        return prompt

    def _create_answer(self, example: Dict[str, Any]) -> str:
        query = example["query"]
        truth_values = example["truth_values"]
        truth_values_dict = {v: self.NICE_VALUES[i] for v, i in truth_values}

        answer = truth_values_dict[query]
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
        clauses: List[str] = data_instance["clauses"]

        query = data_instance["query"]
        truth_values = data_instance["truth_values"]
        truth_values_dict = {v: self.NICE_VALUES[i] for v, i in truth_values}

        query_idx = [var for var, val in truth_values].index(query)
        num_steps = query_idx + 1

        regex_key = self.__class__.__name__ + "var"
        if regex_key not in REGEX_CACHE:
            REGEX_CACHE[regex_key] = re.compile(r"([\+-])\s*(.)")

        regex = REGEX_CACHE[regex_key]

        remaining_clauses = clauses.copy()

        clause_to_index = {c.split("=")[0].strip(): i for i, c in enumerate(clauses)}

        scratchpad_steps = []
        for i in range(num_steps):
            curr_clause = clause_to_index[truth_values[i][0]]
            curr_clause = clauses[curr_clause]
            lhs = curr_clause.split("=")[0].strip()
            rhs = curr_clause.split("=")[1].strip()
            match = regex.match(rhs)

            rhs_sign = match.group(1)
            rhs_var = match.group(2)

            step_input = f"{lhs}"
            if rhs_var == "1":
                computation = f""
            else:
                computation = (
                    f"{rhs_sign} {rhs_var} = {rhs_sign} {truth_values_dict[rhs_var]}"
                )
            step_output = truth_values_dict[lhs]

            remaining_clauses = [
                c for c in remaining_clauses if lhs not in c.split("=")[0].strip()
            ]

            remaining_inputs = ", ".join(remaining_clauses)
            remaining_inputs = f"{remaining_inputs}"

            step = UnifedScratchpadStep(
                input=step_input,
                computation=computation,
                output=step_output,
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
    data_instance = {
        "clauses": ["c = + t", "r = + f", "t = - q", "q = -1", "f = - c"],
        "truth_values": [["q", "0"], ["t", "1"], ["c", "1"], ["f", "0"], ["r", "0"]],
        "query": "c",
        "cat": [5, 4],
    }
    processor = S2SLegoDataInstanceProcessor(
        include_scratchpad=True
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
