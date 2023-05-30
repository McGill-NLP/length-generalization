from transformers import PretrainedConfig

from common import Registrable


class Model(Registrable):
    pass


class HfModelConfig(PretrainedConfig, Registrable):
    @classmethod
    def from_di(cls, **kwargs) -> "HfModelConfig":
        pretrained_name = kwargs.pop("hf_model_name", None)
        if pretrained_name is None:
            return cls(**kwargs)
        else:
            return cls.from_pretrained(pretrained_name, **kwargs)