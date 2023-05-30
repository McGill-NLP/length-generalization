from typing import Optional

from transformers import PreTrainedTokenizerBase, AutoTokenizer

from common import JsonDict
from tokenization_utils.base_tokenizer import Tokenizer


class DIPreTrainedTokenizer(Tokenizer):
    @classmethod
    def from_di(
        cls,
        hf_model_name: str,
        pretrained_args: Optional[JsonDict] = None,
        use_fast: Optional[bool] = False,
        **kwargs
    ) -> PreTrainedTokenizerBase:
        if pretrained_args is None:
            pretrained_args = {}

        tokenizer = AutoTokenizer.from_pretrained(
            hf_model_name, use_fast=use_fast, **pretrained_args
        )
        return tokenizer


Tokenizer.register("pretrained", constructor="from_di", exist_ok=True)(
    DIPreTrainedTokenizer
)

if __name__ == "__main__":
    from common import Params

    tokenizer = Tokenizer.from_params(
        Params(
            {
                "type": "pretrained",
                "hf_model_name": "t5-small",
                "use_fast": False,
            }
        ),
    )

    print(tokenizer)
