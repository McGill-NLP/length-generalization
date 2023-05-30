import logging
from typing import Optional, Tuple, List

import jsonlines
from tokenizers import Tokenizer as HfTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

from common import Params
from data.base_dl_factory import DataLoaderFactory
from tokenization_utils import SpecialTokens
from tokenization_utils.base_tokenizer import Tokenizer

logger = logging.getLogger("app")


def read_jsonlines(path):
    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            yield obj["source"]
            yield obj["target"]


class ManualWhitespaceTokenizer(PreTrainedTokenizerFast, Tokenizer):
    @classmethod
    def from_di(
        cls,
        dataset: DataLoaderFactory,
        experiment_root: str,
        vocab: List[str],
        vocab_size: Optional[int] = 100000,
        special_tokens: Optional[Tuple[str, ...]] = (
            SpecialTokens.PAD,
            SpecialTokens.END,
            SpecialTokens.UNK,
        ),
        **kwargs,
    ) -> "ManualWhitespaceTokenizer":
        logger.info("Building tokenizer from Scratch...")
        tokenizer = HfTokenizer(WordLevel(unk_token="<unk>"))
        tokenizer.pre_tokenizer = WhitespaceSplit()
        tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
        tokenizer
        trainer = WordLevelTrainer(
            vocab_size=vocab_size,
            show_progress=True,
            special_tokens=list(special_tokens),
        )
        tokenizer.train_from_iterator(
            iter([" ".join(vocab)]),
            trainer,
        )
        tokenizer.post_processor = TemplateProcessing(
            single=f"$A {SpecialTokens.END}",
            special_tokens=[
                (SpecialTokens.END, tokenizer.token_to_id(SpecialTokens.END)),
            ],
        )

        tokenizer = cls(
            tokenizer_object=tokenizer,
            eos_token=SpecialTokens.END,
            unk_token=SpecialTokens.UNK,
            pad_token=SpecialTokens.PAD,
        )

        logger.info("Finished building tokenizer!")
        logger.info(tokenizer.__repr__())

        return tokenizer


Tokenizer.register("manual", constructor="from_di")(ManualWhitespaceTokenizer)


if __name__ == "__main__":
    tokenizer = Tokenizer.from_params(
        Params(
            {
                "type": "manual",
                "dataset": None,
                "experiment_root": None,
                "vocab_size": 100,
                "vocab": ["a", "b", "c", "d"]
            }
        ),
    )

    print(tokenizer)
