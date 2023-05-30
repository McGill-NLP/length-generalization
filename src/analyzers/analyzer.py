import json
from pathlib import Path
from typing import Optional

from wandb.sdk.wandb_run import Run

from common import Registrable
from common.torch_utils import is_world_process_zero
from data import DataLoaderFactory
from models import Model
from tokenization_utils import Tokenizer


class Analyzer(Registrable):
    def __init__(
        self,
        model: Model,
        tokenizer: Tokenizer,
        logger: Run,
        dl_factory: DataLoaderFactory,
        exp_root: Path,
        split: str,
        require_best_model: Optional[bool] = False,
        **kwargs,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.logger = logger
        self.dl_factory = dl_factory
        self.exp_root = exp_root
        self.split = split
        self.require_best_model = require_best_model

        analysis_name = self.__class__.__name__ + "__" + self.split
        analysis_root = self.exp_root / "analysis" / analysis_name
        analysis_root.mkdir(parents=True, exist_ok=True)
        self.analysis_root = analysis_root

        self._local_log_obj = {}

    def analyze(self):
        pass

    def log(self, obj):
        self._local_log_obj.update(obj)

    def flush_local_log(self):
        analysis_name = self.__class__.__name__ + "__" + self.split
        log_file = self.analysis_root / "log.json"
        with log_file.open("w") as f:
            json.dump(self._local_log_obj, f, indent=4)

        log_file_copy = self.analysis_root / f"log_{analysis_name}.json"
        with log_file_copy.open("w") as f:
            json.dump(self._local_log_obj, f, indent=4)

        if is_world_process_zero() and self.logger is not None:
            self.logger.save(str(log_file_copy.absolute()), policy="now")
