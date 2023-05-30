import logging
from pathlib import Path
from typing import Optional, Tuple

import jsonlines
from tokenizers import Tokenizer as HfTokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import WordLevelTrainer
from transformers import PreTrainedTokenizerFast

from common import Params, ExperimentStage
from data.base_dl_factory import DataLoaderFactory
from tokenization_utils import SpecialTokens
from tokenization_utils.base_tokenizer import Tokenizer

logger = logging.getLogger("app")


def read_jsonlines(path):
    with jsonlines.open(path, "r") as reader:
        for obj in reader:
            yield obj["source"]
            yield obj["target"]


class WhitespaceTokenizer(PreTrainedTokenizerFast, Tokenizer):
    @classmethod
    def from_di(
        cls,
        dataset: DataLoaderFactory,
        experiment_root: str,
        vocab_size: Optional[int] = 100000,
        special_tokens: Optional[Tuple[str, ...]] = (
            SpecialTokens.PAD,
            SpecialTokens.END,
            SpecialTokens.UNK,
        ),
        **kwargs,
    ) -> "WhitespaceTokenizer":
        experiment_root = Path(experiment_root)

        if (experiment_root / "tokenizer.json").exists():
            tokenizer = HfTokenizer.from_file(str(experiment_root / "tokenizer.json"))
        elif (dataset.dataset_dir / "tokenizer.json").exists():
            tokenizer = HfTokenizer.from_file(dataset.dataset_dir / "tokenizer.json")
        else:
            logger.info("Building tokenizer from Scratch...")
            tokenizer = HfTokenizer(WordLevel(unk_token="<unk>"))
            tokenizer.pre_tokenizer = WhitespaceSplit()
            tokenizer.enable_padding(pad_id=0, pad_token="<pad>")
            trainer = WordLevelTrainer(
                vocab_size=vocab_size,
                show_progress=True,
                special_tokens=list(special_tokens),
            )
            train_file_path = dataset.get_ds_file_path(ExperimentStage.TRAINING)
            if train_file_path.endswith(".jsonl"):
                tokenizer.train_from_iterator(
                    read_jsonlines(dataset.get_ds_file_path(ExperimentStage.TRAINING)),
                    trainer,
                )
            else:
                tokenizer.train(
                    [dataset.get_ds_file_path(ExperimentStage.TRAINING)], trainer
                )
            tokenizer.post_processor = TemplateProcessing(
                single=f"$A {SpecialTokens.END}",
                special_tokens=[
                    (SpecialTokens.END, tokenizer.token_to_id(SpecialTokens.END)),
                ],
            )
            logger.info("Finished building tokenizer!")
            logger.info(tokenizer.__repr__())
            tokenizer.save(str(experiment_root / "tokenizer.json"))

        tokenizer = cls(
            tokenizer_object=tokenizer,
            eos_token=SpecialTokens.END,
            unk_token=SpecialTokens.UNK,
            pad_token=SpecialTokens.PAD,
        )

        logger.info("Finished building tokenizer!")
        logger.info(tokenizer.__repr__())

        return tokenizer


Tokenizer.register("whitespace", constructor="from_di")(WhitespaceTokenizer)


if __name__ == "__main__":
    tokenizer = Tokenizer.from_params(
        Params(
            {
                "type": "whitespace",
                "dataset": {
                    "type": "base",
                    "data_root": "data",
                    "name": "scan",
                    "split": "switch",
                },
                "experiment_root": "experiments/base",
                "vocab_size": 100,
            }
        ),
    )

    print(tokenizer)
