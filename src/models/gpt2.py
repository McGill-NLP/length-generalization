import logging
from typing import Optional, Dict

from transformers import GPT2Config, GPT2ForSequenceClassification, GPT2LMHeadModel

from models.base_model import Model, HfModelConfig
from tokenization_utils import Tokenizer

logger = logging.getLogger("app")


@HfModelConfig.register("gpt2", "from_di")
class DiGPT2Config(GPT2Config, HfModelConfig):
    pass


@Model.register("gpt2")
class CausalGPT2(GPT2LMHeadModel, Model):
    def __init__(
        self,
        config: Optional[HfModelConfig] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ):
        assert config is not None
        super().__init__(config)

        self.handle_tokenizer(tokenizer)

    def handle_tokenizer(self, tokenizer: Optional[Tokenizer] = None):
        if tokenizer is None:
            return

        self.config.eos_token_id = tokenizer.eos_token_id
        self.config.bos_token_id = self.config.eos_token_id
        self.config.pad_token_id = tokenizer.pad_token_id

        if (
            len(tokenizer) > self.transformer.wte.num_embeddings
            or len(tokenizer) < self.config.vocab_size
        ):
            logger.info(
                f"Resizing num_embeddings to {len(tokenizer)} (Previously, {self.transformer.wte.num_embeddings})"
            )
            self.resize_token_embeddings(len(tokenizer))


@Model.register("gpt2_seq_classifier")
class SeqClassifierGPT2(GPT2ForSequenceClassification, Model):
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
        if label2id is not None:
            config.label2id = label2id
            config.id2label = {idx: lbl for lbl, idx in label2id.items()}

        super().__init__(config)

        self.handle_tokenizer(tokenizer)

    def handle_tokenizer(self, tokenizer: Optional[Tokenizer] = None):
        if tokenizer is None:
            return

        self.config.eos_token_id = tokenizer.eos_token_id
        self.config.bos_token_id = self.config.eos_token_id
        self.config.pad_token_id = tokenizer.pad_token_id

        if (
            len(tokenizer) > self.transformer.wte.num_embeddings
            or len(tokenizer) < self.config.vocab_size
        ):
            logger.info(
                f"Resizing num_embeddings to {len(tokenizer)} (Previously, {self.transformer.wte.num_embeddings})"
            )
            self.resize_token_embeddings(len(tokenizer))


if __name__ == "__main__":
    from common import Params

    # DiT5Config.default_implementation = "t5"
    model = Model.from_params(
        Params(
            {
                "type": "gpt2",
                "config": {
                    "type": "gpt2",
                    "hf_model_name": "EleutherAI/gpt-neo-125M",
                    "activation_function": "gelu_new",
                    "architectures": ["GPTNeoForCausalLM"],
                    "attention_dropout": 0,
                    "attention_layers": [
                        "global",
                        "local",
                        "global",
                        "local",
                        "global",
                        "local",
                        "global",
                        "local",
                        "global",
                        "local",
                        "global",
                        "local",
                    ],
                    "attention_types": [[["global", "local"], 6]],
                    "bos_token_id": 50256,
                    "embed_dropout": 0,
                    "eos_token_id": 50256,
                    "gradient_checkpointing": False,
                    "hidden_size": 768,
                    "initializer_range": 0.02,
                    "intermediate_size": None,
                    "layer_norm_epsilon": 1e-05,
                    "max_position_embeddings": 2048,
                    "model_type": "gpt_neo",
                    "num_heads": 12,
                    "num_layers": 12,
                    "resid_dropout": 0,
                    "use_cache": True,
                    "vocab_size": 50257,
                    "window_size": 256,
                },
                "hf_model_name": "EleutherAI/gpt-neo-125M",
            }
        )
    )
    print(model)
