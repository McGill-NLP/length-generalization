from datasets import Dataset
from transformers import TrainerCallback, Trainer
from wandb.sdk.wandb_run import Run

from common import Registrable


class Callback(TrainerCallback, Registrable):
    def __init__(self, **kwargs):
        pass

    def init(self, runtime, eval_dataset: Dataset, eval_split: str, **kwargs):
        pass

    def set_trainer(self, trainer: Trainer):
        self._trainer = trainer

    def save_outputs(self, logger: Run):
        pass
