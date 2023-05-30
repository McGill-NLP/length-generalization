import logging
from typing import Optional, Dict

from transformers import RobertaConfig, RobertaForSequenceClassification

from models.base_model import Model, HfModelConfig
from tokenization_utils import Tokenizer

logger = logging.getLogger("app")


@HfModelConfig.register("roberta", "from_di", exist_ok=True)
class DiRobertaConfig(RobertaConfig, HfModelConfig):
    pass


@Model.register("roberta_seq_classifier", exist_ok=True)
class SeqClassifierRoberta(RobertaForSequenceClassification, Model):
    def __init__(
        self,
        config: Optional[HfModelConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
        problem_type: Optional[str] = "single_label_classification",
        label2id: Optional[Dict[str, int]] = None,
        **kwargs,
    ):
        assert config is not None

        if problem_type is not None:
            config.problem_type = problem_type
            if problem_type == "regression":
                config.num_labels = 1

        if label2id is not None:
            config.label2id = label2id
            config.id2label = {idx: lbl for lbl, idx in label2id.items()}

        super().__init__(config)

        self.handle_tokenizer(tokenizer)

    def handle_tokenizer(self, tokenizer: Optional[Tokenizer] = None):
        if tokenizer is None:
            return

        self.config.eos_token_id = tokenizer.eos_token_id
        self.config.bos_token_id = tokenizer.bos_token_id
        self.config.pad_token_id = tokenizer.pad_token_id

        embedding = self.roberta.get_input_embeddings()

        if (
            len(tokenizer) > embedding.num_embeddings
            or len(tokenizer) < self.config.vocab_size
        ):
            logger.info(
                f"Resizing num_embeddings to {len(tokenizer)} (Previously, {embedding.num_embeddings})"
            )
            self.resize_token_embeddings(len(tokenizer))


if __name__ == "__main__":
    from common import Params

    model = Model.from_params(
        Params(
            {
                "type": "roberta_seq_classifier",
                "config": {
                    "type": "roberta",
                    "hf_model_name": "roberta-base",
                    "position_embedding_type": "saaag",
                },
            }
        )
    )
    o = model(**model.dummy_inputs)
    print(model)
