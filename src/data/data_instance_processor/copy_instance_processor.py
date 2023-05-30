import logging
from typing import Dict, Any

from data.data_instance_processor.data_instance_processor import (
    DataInstanceProcessor,
    IdentityDataInstanceProcessor,
)

logger = logging.getLogger("app")


@DataInstanceProcessor.register("s2s_copy", exist_ok=True)
class S2SCopyDataInstanceProcessor(IdentityDataInstanceProcessor):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        category = str(example["cat"])
        src_str = example["src_seq"]
        tgt_str = example["tgt_seq"]

        if "target_char" in example and "source_char" in example:
            prompt = f'Replace {example["source_char"]} with {example["target_char"]} in {src_str} '
        else:
            prompt = f"Copy: {src_str} "

        targets = f"{tgt_str}"
        return {
            "category": category,
            self.source_seq_key: prompt,
            self.target_seq_key: targets,
            "answer": tgt_str,
        }


@DataInstanceProcessor.register("s2s_reverse", exist_ok=True)
class S2SReverseDataInstanceProcessor(IdentityDataInstanceProcessor):
    def __call__(self, example: Dict[str, Any]) -> Dict[str, Any]:
        category = str(example["cat"])
        src_str = example["src_seq"]
        tgt_str = example["tgt_seq"]

        prompt = f"Reverse: {src_str} "

        targets = f"{tgt_str}"
        return {
            "category": category,
            self.source_seq_key: prompt,
            self.target_seq_key: targets,
            "answer": tgt_str,
        }


if __name__ == "__main__":
    data_instance = {
        "repeat": 3,
        "source_char": "6",
        "target_char": ">",
        "src_seq": "6 6 6",
        "tgt_seq": "> > >",
        "cat": 3,
    }
    processor = S2SCopyDataInstanceProcessor(input_output_sep_token="[SEP] ")

    o = processor(data_instance)

    s = "Replace 6 with > in 6 6 6 [SEP] > > >"
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
    encoding = tokenizer(s, add_special_tokens=False)

    print(o)
    print(encoding.tokens())
