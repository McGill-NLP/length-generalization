import logging
from typing import Optional, Dict, Any

from data.data_instance_processor.data_instance_processor import (
    DataInstanceProcessor,
    IdentityDataInstanceProcessor,
)

logger = logging.getLogger("app")


@DataInstanceProcessor.register("scan", exist_ok=True)
class S2SScanDataInstanceProcessor(IdentityDataInstanceProcessor):
    target_tokens_map = {
        "I_WALK": "Walk",
        "I_RUN": "Run",
        "I_JUMP": "Jump",
        "I_LOOK": "Look",
        "I_TURN_LEFT": "Left",
        "I_TURN_RIGHT": "Right",
    }

    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        example = super().__call__(example)
        converted_targets = self._replace_target_tokens(example[self.target_seq_key])
        example.update(
            {
                self.target_seq_key: converted_targets,
                "answer": converted_targets,
            }
        )
        return example

    def _replace_target_tokens(self, targets: str) -> str:
        for k, v in self.target_tokens_map.items():
            targets = targets.replace(k, v)
        return targets


@DataInstanceProcessor.register("scan_bos", exist_ok=True)
class S2SScanBOSDataInstanceProcessor(S2SScanDataInstanceProcessor):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        example = super().__call__(example)
        example.update({self.source_seq_key: f"Solve: {example[self.source_seq_key]}"})
        return example


if __name__ == "__main__":
    data_instance = {
        "source": "walk after turn right",
        "target": "I_TURN_RIGHT I_WALK",
        "category": 2,
    }
    processor = S2SScanBOSDataInstanceProcessor(input_output_sep_token="[SEP] ")

    o = processor(data_instance)

    s = "jump opposite left twice and look around right twice[SEP] Left Left Jump Left Left Jump Right Look Right Look Right Look Right Look Right Look Right Look Right Look Right Look"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
    encoding = tokenizer(s, add_special_tokens=False)
    replicated_seq = tokenizer.decode(
        encoding["input_ids"],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    assert s == replicated_seq
    processor.get_tokenization_info(data_instance, encoding, s)

    print(o)
