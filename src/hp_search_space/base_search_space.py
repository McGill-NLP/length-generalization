import abc
import copy
from dataclasses import dataclass, field
from typing import Callable, Dict

from overrides import overrides

from common import Registrable, JsonDict, Params
from common.nest import flatten


@dataclass
class SearchSpace:
    model: JsonDict = field(default_factory=dict)
    trainer: JsonDict = field(default_factory=dict)


class HPSearchSpace(Registrable):
    def __init__(self, metric: str, direction: str):
        self.metric = metric
        self.direction = direction

    @abc.abstractmethod
    def get_search_space(self, *args, **kwargs) -> SearchSpace:
        raise NotImplementedError()

    def get_compute_obj_fn(self) -> Callable[[Dict[str, float]], float]:
        metric_to_optimize = self.metric

        def compute_obj(metrics: Dict[str, float]):
            metrics = copy.deepcopy(metrics)
            obj = metrics.get(f"eval_{metric_to_optimize}", None)
            return obj

        return compute_obj

    def get_search_space_fn(self) -> Callable[[...], JsonDict]:
        def search_space_fn(*args, **kwargs):
            search_space = self.get_search_space(*args, **kwargs)
            nested_dict = {"MHSP": search_space.model, **search_space.trainer}
            d = flatten(nested_dict, separator="/")
            return d

        return search_space_fn


@HPSearchSpace.register("hf_default")
class HFDefaultSearchSpace(HPSearchSpace):
    @overrides
    def get_search_space(self, *args, **kwargs) -> SearchSpace:
        from ray import tune

        trainer_search_space = {
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            "num_train_epochs": tune.choice(list(range(1, 6))),
            "seed": tune.uniform(1, 40),
            "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
        }

        return SearchSpace(trainer=trainer_search_space)


if __name__ == "__main__":
    hp_sp = HPSearchSpace.from_params(
        Params(
            {
                "type": "hf_default",
                "metric": "acc",
                "direction": "maximize"
            }
        )
    )

    print(hp_sp.get_search_space_fn()())
