from transformers import PreTrainedTokenizerBase

from common import Registrable


class Tokenizer(PreTrainedTokenizerBase, Registrable):
    pass
