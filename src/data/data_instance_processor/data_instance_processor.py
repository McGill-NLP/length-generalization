import re
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Optional, Dict, Any, List, Tuple, Union

from transformers import BatchEncoding

from common import Registrable
from tokenization_utils import substr_to_token_ids


class DataInstanceProcessor(Registrable):
    def __init__(
        self,
        source_seq_key: Optional[str] = "source",
        target_seq_key: Optional[str] = "target",
        dataset_name: Optional[str] = None,
        split_name: Optional[str] = None,
        input_output_sep_token: Optional[str] = None,
        **kwargs,
    ):
        self.source_seq_key = source_seq_key
        self.target_seq_key = target_seq_key
        self.dataset_name = dataset_name
        self.split_name = split_name
        self.input_output_sep_token = input_output_sep_token

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        raise NotImplementedError()

    def extract_answer_from_prediction(self, prediction: str) -> Any:
        raise NotImplementedError()

    def is_prediction_correct(
        self, prediction: str, data_instance: Dict[str, Any]
    ) -> bool:
        answer = data_instance["answer"]
        return prediction == answer

    def get_tokenization_info(
        self, data_instance: Dict[str, Any], encoding: BatchEncoding, orig_seq: str
    ) -> Dict[str, Any]:
        raise NotImplementedError()


@DataInstanceProcessor.register("identity")
class IdentityDataInstanceProcessor(DataInstanceProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.input_output_sep_token is not None

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        targets = self._convert_targets(example[self.target_seq_key])
        example.update(
            {
                self.target_seq_key: targets,
                "answer": targets.strip(),
                "category": self._get_category(example),
            }
        )
        return example

    def _convert_targets(self, targets: str) -> str:
        return targets

    def _get_category(self, example: Dict[str, Any]) -> str:
        potential_category_keys = ["category", "cat", "categories", "cats"]
        for key in potential_category_keys:
            if key in example:
                return str(example[key])
        return "unknown"

    def extract_answer_from_prediction(self, prediction: str) -> Any:
        prediction_str = prediction.split(self.input_output_sep_token)[1].strip()
        return prediction_str

    def is_prediction_correct(
        self, prediction: str, data_instance: Dict[str, Any]
    ) -> bool:
        gold_answer = data_instance.get("answer", None)
        if gold_answer is None:
            gold_answer = data_instance[self.target_seq_key].strip()
        predicted_answer = self.extract_answer_from_prediction(prediction)
        return predicted_answer.strip() == gold_answer

    def get_tokenization_info(
        self, data_instance: Dict[str, Any], encoding: BatchEncoding, orig_seq: str
    ) -> Dict[str, Any]:
        assert encoding.is_fast
        all_words: List[int] = sorted(set(encoding.word_ids()))

        # Find self.input_output_sep_token
        sep_word_id = None
        for w_id in all_words:
            span = encoding.word_to_chars(w_id)
            if self.input_output_sep_token.strip() in orig_seq[span.start : span.end]:
                sep_word_id = w_id
                break

        if sep_word_id is None:
            raise ValueError(
                f"Cannot find {self.input_output_sep_token} in {orig_seq}. Tokens: {encoding.tokens()}"
            )

        input_tokens_boundary = range(0, encoding.word_to_tokens(sep_word_id).end)
        sep_token_ids: List[int] = substr_to_token_ids(
            self.input_output_sep_token.strip(), encoding, orig_seq
        )

        seq_level_regions: List[str] = []
        for token_id, token in enumerate(encoding.tokens()):
            if token_id in sep_token_ids:
                region = "sep"
            elif token_id in input_tokens_boundary:
                region = "input"
            else:
                region = "output"
            seq_level_regions.append(region)

        info = {
            "seq_level_region": seq_level_regions,
            "encoding": encoding,
            "orig_seq": orig_seq,
            "sep_word_id": sep_word_id,
        }

        return info


@dataclass
class UnifedScratchpadStep:
    input: str = None
    computation: str = None
    output: str = None
    intermediate_variables: List[str] = None
    remaining_input: str = None

    def __eq__(self, other):
        if not isinstance(other, UnifedScratchpadStep):
            return False
        for f in fields(self):
            attr = f.name
            if (
                getattr(self, attr) is not None
                and getattr(other, attr) is not None
                and getattr(self, attr) != getattr(other, attr)
            ):
                return False
        return True


REGEX_CACHE: Dict[str, re.Pattern] = {}


class DataInstanceProcessorWithUnifiedScratchpad(DataInstanceProcessor):
    """
    This is the base class for all data instance processors that support a unified scratchpad.
    Unified scratchpad format looks like this:
    <scratch>
    ...
    >> <INPUT> :: <COMPUTATIONS> == <OUTPUT> {Var1: X1, Var2: X2, ...} => <REMAINING_INPUT>
    ...
    </scratch>
    """

    def __init__(
        self,
        include_scratchpad: Optional[bool] = False,
        include_input: Optional[bool] = True,
        include_computation: Optional[bool] = True,
        include_output: Optional[bool] = True,
        include_intermediate_variables: Optional[bool] = True,
        include_remaining_input: Optional[bool] = True,
        step_separator: Optional[str] = "#",
        scratchpad_bos: Optional[str] = "[scratch]",
        scratchpad_eos: Optional[str] = "[/scratch]",
        input_marker: Optional[str] = ">>",
        computation_marker: Optional[str] = "::",
        output_marker: Optional[str] = "==",
        intermediate_variables_marker_begin: Optional[str] = "[",
        intermediate_variables_marker_end: Optional[str] = "]",
        intermediate_variables_marker_sep: Optional[str] = ", ",
        remaining_input_marker: Optional[str] = "=>",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.include_scratchpad = include_scratchpad
        self.include_input = include_input
        self.include_computation = include_computation
        self.include_output = include_output
        self.include_intermediate_variables = include_intermediate_variables
        self.include_remaining_input = include_remaining_input
        self.step_separator = step_separator
        self.scratchpad_bos = scratchpad_bos
        self.scratchpad_eos = scratchpad_eos
        self.input_marker = input_marker
        self.computation_marker = computation_marker
        self.output_marker = output_marker
        self.intermediate_variables_marker_begin = intermediate_variables_marker_begin
        self.intermediate_variables_marker_end = intermediate_variables_marker_end
        self.intermediate_variables_marker_sep = intermediate_variables_marker_sep
        self.remaining_input_marker = remaining_input_marker

    def _get_hash_key(self) -> str:
        return (
            f"{self.include_scratchpad}__{self.include_input}__"
            f"{self.include_computation}__{self.include_output}__"
            f"{self.include_intermediate_variables}__{self.include_remaining_input}__"
            f"{self.step_separator}__{self.scratchpad_bos}__"
            f"{self.scratchpad_eos}__{self.input_marker}__"
            f"{self.computation_marker}__{self.output_marker}__"
            f"{self.intermediate_variables_marker_sep}__"
            f"{self.intermediate_variables_marker_begin}__"
            f"{self.intermediate_variables_marker_end}__{self.remaining_input_marker}"
        )

    def create_scratchpad_str(
        self, scratchpad_steps: List[UnifedScratchpadStep]
    ) -> str:
        scratchpad_strs = []
        for step in scratchpad_steps:
            scratchpad_str = ""
            if self.include_input:
                scratchpad_str += f"{self.input_marker} " + step.input
                scratchpad_str += " "

            if self.include_computation:
                scratchpad_str += f"{self.computation_marker} " + step.computation
                scratchpad_str += " "

            if self.include_output:
                scratchpad_str += f"{self.output_marker} " + step.output
                scratchpad_str += " "

            if self.include_intermediate_variables:
                scratchpad_str += self.intermediate_variables_marker_begin + " "
                scratchpad_str += self.intermediate_variables_marker_sep.join(
                    step.intermediate_variables
                )
                scratchpad_str += " " + self.intermediate_variables_marker_end
                scratchpad_str += " "

            if self.include_remaining_input:
                scratchpad_str += (
                    f"{self.remaining_input_marker} " + step.remaining_input
                )
                scratchpad_str += " "

            scratchpad_strs.append(scratchpad_str)

        scratchpad_str = self.step_separator.join(scratchpad_strs)
        scratchpad_str = (
            self.scratchpad_bos + " " + scratchpad_str + " " + self.scratchpad_eos
        )
        return scratchpad_str

    def _get_scratchpad_regex(self):
        regex_pattern_key = self.__class__.__name__ + "___" + self._get_hash_key()
        if regex_pattern_key not in REGEX_CACHE:
            pattern = r"\s*"
            if self.include_input:
                pattern += re.escape(self.input_marker) + r"\s*(?P<input>.*)\s*"
            if self.include_computation:
                pattern += (
                    re.escape(self.computation_marker) + r"\s*(?P<computation>.*)\s*"
                )
            if self.include_output:
                pattern += re.escape(self.output_marker) + r"\s*(?P<output>.*)\s*"
            if self.include_intermediate_variables:
                begin = self.intermediate_variables_marker_begin
                end = self.intermediate_variables_marker_end
                pattern += (
                    re.escape(begin)
                    + r"\s*(?P<intermediate_variables>.*)\s*"
                    + re.escape(end)
                    + r"\s*"
                )
            if self.include_remaining_input:
                pattern += (
                    re.escape(self.remaining_input_marker)
                    + r"\s*(?P<remaining_input>.*)"
                )
            pattern += r"\s*"
            REGEX_CACHE[regex_pattern_key] = re.compile(pattern)
        regex = REGEX_CACHE[regex_pattern_key]
        return regex

    def get_tokenization_info(
        self, data_instance: Dict[str, Any], encoding: BatchEncoding, orig_seq: str
    ) -> Dict[str, Any]:
        if not self.include_scratchpad:
            prompt = data_instance[self.source_seq_key]
            assert prompt in orig_seq

            prompt_token_ids = set(substr_to_token_ids(prompt, encoding, orig_seq))
            seq_level_regions = [
                "input" if token_id in prompt_token_ids else "output"
                for token_id in range(len(encoding.tokens()))
            ]
            info = {
                "seq_level_region": seq_level_regions,
                "encoding": encoding,
                "orig_seq": orig_seq,
            }
            return info

        scratchpad_bos_token_ids = substr_to_token_ids(
            self.scratchpad_bos, encoding, orig_seq
        )
        scratchpad_eos_token_ids = substr_to_token_ids(
            self.scratchpad_eos, encoding, orig_seq
        )

        scratchpad_bos_span = (
            scratchpad_bos_token_ids[0],
            scratchpad_bos_token_ids[-1],
        )
        scratchpad_eos_span = (
            scratchpad_eos_token_ids[0],
            scratchpad_eos_token_ids[-1],
        )

        start_index = orig_seq.find(self.scratchpad_bos)
        end_index = orig_seq.find(self.scratchpad_eos)

        if start_index != -1 and end_index != -1:
            scratchpad_str = orig_seq[
                start_index + len(self.scratchpad_bos) : end_index
            ]
        else:
            scratchpad_str = ""

        scratchpad_sep = self.step_separator.strip()
        char_idx_to_step_idx = {}
        step_idx = 0
        char_idx = 0
        while (char_idx + len(scratchpad_sep)) <= len(orig_seq):
            if char_idx in range(
                start_index, start_index + len(self.scratchpad_bos)
            ) or char_idx in range(end_index, end_index + len(self.scratchpad_eos)):
                char_idx += 1
                continue

            if orig_seq[char_idx : char_idx + len(scratchpad_sep)] == scratchpad_sep:
                step_idx += 1
                char_idx += len(scratchpad_sep)
            else:
                if char_idx < start_index:
                    char_idx_to_step_idx[char_idx] = -1
                elif char_idx >= (end_index + len(self.scratchpad_eos)):
                    char_idx_to_step_idx[char_idx] = -2
                else:
                    char_idx_to_step_idx[char_idx] = step_idx

                char_idx += 1

        # Step_id to list of char indices
        step_idx_to_char_idx = defaultdict(list)
        for char_idx, step_idx in char_idx_to_step_idx.items():
            step_idx_to_char_idx[step_idx].append(char_idx)

        # Step_id to (start, end) char indices
        step_idx_to_char_idx_span = {}
        for step_idx, char_idx_list in step_idx_to_char_idx.items():
            step_idx_to_char_idx_span[step_idx] = (
                min(char_idx_list),
                max(char_idx_list),
            )

        # Step_id to string
        step_idx_to_str = {}
        for step_idx, char_idx_list in step_idx_to_char_idx.items():
            step_idx_to_str[step_idx] = "".join(
                [orig_seq[i] for i in sorted(char_idx_list)]
            )

        scratchpad_step_regex = self._get_scratchpad_regex()

        char_idx_to_scratchpad_components = {}
        for step_idx in step_idx_to_str.keys():
            if step_idx < 0:
                continue
            step_span = step_idx_to_char_idx_span[step_idx]
            step_str = step_idx_to_str[step_idx]
            match = scratchpad_step_regex.match(step_str)
            if match is None:
                continue
            scratchpad_components = match.groupdict()
            for key in scratchpad_components:
                span = match.span(key)
                for char_idx in range(step_span[0] + span[0], step_span[0] + span[1]):
                    char_idx_to_scratchpad_components[char_idx] = key

        # Convert char_idx_to_* to token_idx_to_*
        token_idx_to_scratchpad_components = {}
        for char_idx in char_idx_to_scratchpad_components:
            token_idx = encoding.char_to_token(char_idx)
            if token_idx is None:
                continue

            scratchpad_component = char_idx_to_scratchpad_components[char_idx]
            if token_idx in token_idx_to_scratchpad_components:
                assert (
                    token_idx_to_scratchpad_components[token_idx]
                    == scratchpad_component
                )
            else:
                token_idx_to_scratchpad_components[token_idx] = scratchpad_component

        token_idx_to_scratchpad_step = {}
        for char_idx in char_idx_to_step_idx:
            token_idx = encoding.char_to_token(char_idx)
            if token_idx is None:
                continue

            step_idx = char_idx_to_step_idx[char_idx]
            if token_idx in token_idx_to_scratchpad_step:
                assert token_idx_to_scratchpad_step[token_idx] == step_idx
            else:
                token_idx_to_scratchpad_step[token_idx] = step_idx

        scratchpad_steps = []
        scratchpad_components = []
        seq_level_regions = []
        for token_id, token in enumerate(encoding.tokens()):
            if token_id < scratchpad_bos_span[0]:
                region = "input"
            elif token_id > scratchpad_eos_span[1]:
                region = "output"
            else:
                region = "scratchpad"
            seq_level_regions.append(region)
            scratchpad_steps.append(token_idx_to_scratchpad_step.get(token_id, None))
            scratchpad_components.append(
                token_idx_to_scratchpad_components.get(token_id, None)
            )

        info = {
            "seq_level_region": seq_level_regions,
            "scratchpad_step": scratchpad_steps,
            "scratchpad_component": scratchpad_components,
            "encoding": encoding,
            "orig_seq": orig_seq,
        }
        return info

    def parse_scratchpad_str(
        self, scratchpad_str: str
    ) -> Tuple[List[UnifedScratchpadStep], List[str]]:
        """
        We use regex to parse scratchpad str into its components.
        """
        scratchpad_str = scratchpad_str.split(self.scratchpad_bos)[-1]
        scratchpad_str = scratchpad_str.split(self.scratchpad_eos)[0]
        scratchpad_steps = scratchpad_str.split(self.step_separator)

        regex = self._get_scratchpad_regex()

        parsed_scratchpad_steps: List[UnifedScratchpadStep] = []
        failed_steps: List[str] = []

        for step in scratchpad_steps:
            match = regex.match(step)
            if match is None:
                failed_steps.append(step)
                continue
            else:
                step = match.groupdict()

            # Strip the spaces from the beginning and end of each component.
            for key in step.keys():
                if isinstance(step[key], str):
                    step[key] = step[key].strip()

            if (
                self.include_intermediate_variables
                and step["intermediate_variables"] is not None
            ):
                step["intermediate_variables"] = [
                    var.strip() for var in step["intermediate_variables"].split(",")
                ]
            else:
                step["intermediate_variables"] = None

            parsed_step = UnifedScratchpadStep(**step)
            parsed_scratchpad_steps.append(parsed_step)

        return parsed_scratchpad_steps, failed_steps

    def evaluate_parsed_scratchpad(
        self,
        predicted_scratchpad_steps: List[UnifedScratchpadStep],
        failed_steps: List[str],
        gold_scratchpad_steps: List[UnifedScratchpadStep],
    ) -> Dict[str, Any]:
        scratchpad_metrics = {}
        if self.include_input:
            scratchpad_metrics["correct_inputs"] = 0
        if self.include_computation:
            scratchpad_metrics["correct_computations"] = 0
        if self.include_output:
            scratchpad_metrics["correct_outputs"] = 0
        if self.include_intermediate_variables:
            scratchpad_metrics["correct_intermediate_variables"] = 0
        if self.include_remaining_input:
            scratchpad_metrics["correct_remaining_inputs"] = 0

        scratchpad_metrics["correct_steps"] = 0

        if (
            len(predicted_scratchpad_steps) != len(gold_scratchpad_steps)
            or len(failed_steps) > 0
        ):
            for key in scratchpad_metrics:
                scratchpad_metrics[key] = float(0)
            return scratchpad_metrics

        for step_idx, (predicted_step, gold_step) in enumerate(
            zip(predicted_scratchpad_steps, gold_scratchpad_steps)
        ):
            is_step_correct = True
            if self.include_input:
                is_input_correct = (
                    predicted_step.input.strip() == gold_step.input.strip()
                )
                scratchpad_metrics["correct_inputs"] += is_input_correct
                is_step_correct &= is_input_correct

            if self.include_computation:
                is_computation_correct = (
                    predicted_step.computation.strip() == gold_step.computation.strip()
                )
                scratchpad_metrics["correct_computations"] += is_computation_correct
                is_step_correct &= is_computation_correct

            if self.include_output:
                is_output_correct = (
                    predicted_step.output.strip() == gold_step.output.strip()
                )
                scratchpad_metrics["correct_outputs"] += is_output_correct
                is_step_correct &= is_output_correct

            if self.include_intermediate_variables:
                is_intermediate_variables_correct = (
                    predicted_step.intermediate_variables
                    == gold_step.intermediate_variables
                )
                scratchpad_metrics[
                    "correct_intermediate_variables"
                ] += is_intermediate_variables_correct
                is_step_correct &= is_intermediate_variables_correct

            if self.include_remaining_input:
                is_remaining_input_correct = (
                    " ".join(predicted_step.remaining_input.split()).strip()
                    == " ".join(gold_step.remaining_input.split()).strip()
                )
                scratchpad_metrics[
                    "correct_remaining_inputs"
                ] += is_remaining_input_correct
                is_step_correct &= is_remaining_input_correct

            scratchpad_metrics["correct_steps"] += is_step_correct

        # We normalize the metrics by the number of steps.
        for key in scratchpad_metrics:
            scratchpad_metrics[key] /= len(gold_scratchpad_steps)

        return scratchpad_metrics

    def evaluate_scratchpad(
        self,
        prediction: str,
        data_instance: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not self.include_scratchpad:
            return {"failed_steps": 0}

        gold_scratchpad_steps = self.create_scratchpad_steps_from_data_instance(
            data_instance
        )

        start_index = prediction.find(self.scratchpad_bos)
        end_index = prediction.find(self.scratchpad_eos)

        if start_index != -1 and end_index != -1:
            scratchpad_str = prediction[
                start_index + len(self.scratchpad_bos) : end_index
            ]
        else:
            scratchpad_str = ""

        predicted_scratchpad_steps, failed_steps = self.parse_scratchpad_str(
            scratchpad_str
        )

        scratchpad_metrics = self.evaluate_parsed_scratchpad(
            predicted_scratchpad_steps,
            failed_steps,
            gold_scratchpad_steps,
        )

        if scratchpad_str == "":
            scratchpad_metrics["failed_steps"] = 1
        else:
            scratchpad_metrics["failed_steps"] = len(failed_steps) / len(
                gold_scratchpad_steps
            )

        return scratchpad_metrics

    def extract_answer_from_prediction(self, prediction: str) -> Any:
        final_output = (
            prediction.split(self.scratchpad_eos)[1]
            if self.scratchpad_eos in prediction
            else prediction
        )
        return self.extract_answer_from_final_output(final_output)

    def is_prediction_correct(
        self, prediction: str, data_instance: Dict[str, Any]
    ) -> bool:
        answer = self.extract_answer_from_prediction(prediction)
        return self.is_answer_correct(answer, data_instance)

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self._create_prompt(example)
        answer = self._create_answer(example)
        targets = self._create_targets(answer)
        category = self._get_category(example)

        if self.include_scratchpad:
            scratchpad_steps = self.create_scratchpad_steps_from_data_instance(example)
            scratchpad_str = self.create_scratchpad_str(scratchpad_steps) + " "
            eval_result = self.evaluate_scratchpad(
                prompt + scratchpad_str + targets, example
            )
            for key in eval_result:
                if key != "failed_steps":
                    assert (
                        eval_result[key] == 1.0
                    ), f"Scratchpad self evaluation failed for {example['idx']} on {key}"
        else:
            scratchpad_str = ""

        targets = f"{scratchpad_str}{targets}"

        return {
            "category": category,
            self.source_seq_key: prompt,
            self.target_seq_key: targets,
            "answer": answer,
        }

    def create_human_readable_scratchpad(
        self, scratchpad_steps: List[UnifedScratchpadStep]
    ) -> str:
        scratchpad_strs = []
        for step in scratchpad_steps:
            scratchpad_str = ""
            if self.include_input:
                scratchpad_str += f"{self.input_marker} " + step.input
                scratchpad_str += "   "

            if self.include_computation:
                scratchpad_str += f"{self.computation_marker} " + step.computation
                scratchpad_str += "   "

            if self.include_output:
                scratchpad_str += f"{self.output_marker} " + step.output
                scratchpad_str += "   "

            if self.include_intermediate_variables:
                scratchpad_str += self.intermediate_variables_marker_begin
                scratchpad_str += self.intermediate_variables_marker_sep.join(
                    step.intermediate_variables
                )
                scratchpad_str += self.intermediate_variables_marker_end
                scratchpad_str += "   "

            if self.include_remaining_input:
                scratchpad_str += (
                    f"{self.remaining_input_marker} " + step.remaining_input
                )
                scratchpad_str += "   "

            scratchpad_strs.append(scratchpad_str)

        scratchpad_str = "\n".join(scratchpad_strs)
        return scratchpad_str

    @abstractmethod
    def _create_prompt(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError()

    @abstractmethod
    def _create_answer(self, example: Dict[str, Any]) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def _create_targets(self, answer: Any) -> str:
        raise NotImplementedError()

    @abstractmethod
    def _get_category(self, example: Dict[str, Any]) -> str:
        raise NotImplementedError()

    @abstractmethod
    def create_scratchpad_steps_from_data_instance(
        self, data_instance: Dict[str, Any]
    ) -> List[UnifedScratchpadStep]:
        raise NotImplementedError()

    @abstractmethod
    def extract_answer_from_final_output(self, final_output: str) -> Any:
        raise NotImplementedError()

    @abstractmethod
    def is_answer_correct(
        self, final_answer: Any, data_instance: Dict[str, Any]
    ) -> bool:
        raise NotImplementedError()


def number_to_string(number: Union[int, float]) -> str:
    """
    Converts a number to a string such that digits are separated by spaces.
    Also, it handles both negative and positive numbers.
    """
    if number < 0:
        return "- " + number_to_string(-number)
    else:
        return " ".join(str(number))
