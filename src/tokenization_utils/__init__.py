from typing import List

from transformers import BatchEncoding


class SpecialTokens:
    PAD = "<pad>"
    START = "<s>"
    END = "</s>"
    UNK = "<unk>"

    @classmethod
    def all(cls) -> List[str]:
        return [cls.PAD, cls.END, cls.UNK, cls.START]


from .base_tokenizer import Tokenizer
from .pre_trained import DIPreTrainedTokenizer
from .whitespace import WhitespaceTokenizer
from .manual import ManualWhitespaceTokenizer


def substr_to_token_ids(
    substr: str, encoding: BatchEncoding, orig_seq: str
) -> List[int]:
    assert encoding.is_fast
    assert substr in orig_seq
    substr_start = orig_seq.index(substr)
    substr_end = substr_start + len(substr)
    token_ids = [encoding.char_to_token(i) for i in range(substr_start, substr_end)]
    token_ids = [i for i in token_ids if i is not None]
    token_ids = sorted(set(token_ids))
    return token_ids
