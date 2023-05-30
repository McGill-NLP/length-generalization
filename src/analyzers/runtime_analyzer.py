import copy
import json
import time
from collections import defaultdict
from logging import getLogger
from typing import Callable, List, Dict, Any, Protocol

import torch
from tqdm import tqdm
from transformers import PreTrainedModel
from transformers.trainer_utils import TrainerMemoryTracker

from analyzers import Analyzer
from common.torch_utils import (
    garbage_collection_cuda,
    is_oom_error,
    is_world_process_zero,
)

logger = getLogger("app")


class MemoryTracker(TrainerMemoryTracker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_reported = True

    def derive_stage(self):
        return self.stages["train"]


class ComputeFuncCreator(Protocol):
    def __call__(
        self, batch_size: int, seq_length: int, device: str, steps: int = 1
    ) -> Callable:
        ...


@Analyzer.register("runtime")
class RuntimeAnalyzer(Analyzer):
    def __init__(
        self,
        batch_size: int,
        max_length: int,
        length_bucket_width: int,
        num_try_per_length_bucket: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.max_length = max_length
        self.length_bucket_width = length_bucket_width
        self.num_try_per_length_bucket = num_try_per_length_bucket

        self.model = self.model.to("cpu")
        self.orig_state_dict = copy.deepcopy(self.model.state_dict().copy())

    def analyze(self):
        self._collect_inference_metrics()
        self._collect_generation_metrics()
        self._collect_training_metrics()

    def _collect_metrics(
        self, create_compute_fn: ComputeFuncCreator, prefix: str
    ) -> Dict[str, Any]:
        length_buckets = list(range(1, self.max_length + 1, self.length_bucket_width))

        stats_funcs = [
            ("gpu", self._measure_gpu_time),
            ("cpu", self._measure_cpu_time),
            ("mem", self._measure_memory),
        ]

        # Save the stats
        stats_dir = self.analysis_root / f"{prefix}_stat_files"
        stats_dir.mkdir(exist_ok=True, parents=True)

        stats_dict = defaultdict(dict)
        for length_bucket in tqdm(length_buckets):
            stats_file = (
                stats_dir
                / f"{str(length_bucket).zfill(len(str(self.max_length)))}.json"
            )
            if stats_file.exists():
                with open(stats_file, "r") as f:
                    stats_dict[length_bucket] = json.load(f)
                continue

            logger.info(f"Length bucket {length_bucket}")
            for stat_name, stat_func in tqdm(stats_funcs):
                device = "cpu" if stat_name == "cpu" else "cuda"
                try:
                    compute_fn = create_compute_fn(
                        self.batch_size, length_bucket, device
                    )
                    if stat_name == "cpu":
                        num_try = 10
                    else:
                        num_try = self.num_try_per_length_bucket
                    stats_dict[length_bucket][stat_name] = stat_func(
                        compute_fn, num_try
                    )
                except RuntimeError as e:
                    if is_oom_error(e):
                        logger.warning(
                            f"OOM error when measuring {stat_name} for length {length_bucket}. Skipping."
                        )
                        continue

            # Measure throughput
            # Find the optimal batch size for the given length bucket
            def loop_fn(bs: int):
                create_compute_fn(bs, length_bucket, "cuda", steps=5)()

            try:
                optimal_batch_size = self._find_optimal_batch_size(loop_fn)
                logger.info(f"Found optimal batch_size {optimal_batch_size}")
                compute_fn = create_compute_fn(
                    optimal_batch_size, length_bucket, "cuda"
                )
                stats_dict[length_bucket]["optimal_batch_size"] = optimal_batch_size

                stats_dict[length_bucket]["throughput"] = self._measure_gpu_time(
                    compute_fn, self.num_try_per_length_bucket
                )
                stats_dict[length_bucket][
                    "throughput_num_tries"
                ] = self.num_try_per_length_bucket
            except RuntimeError as e:
                if is_oom_error(e):
                    logger.warning(
                        f"OOM error when measuring throughput for length {length_bucket}. Skipping."
                    )
                    continue

            with stats_file.open("w") as f:
                json.dump(stats_dict[length_bucket], f, indent=4)

            if is_world_process_zero() and self.logger is not None:
                # Copy the `stats_file` to `prefix_<length_bucket>.json` for the logger
                copy_file = self.analysis_root / f"{prefix}_{length_bucket}.json"
                with copy_file.open("w") as f:
                    json.dump(stats_dict[length_bucket], f, indent=4)

                self.logger.save(str(copy_file.absolute()), policy="now")

        return stats_dict

    def _collect_training_metrics(self):
        logger.info("Collecting training metrics")

        def create_train_compute_fn(
            batch_size: int, seq_length: int, device: str, steps: int = 1
        ) -> Callable[[], None]:
            garbage_collection_cuda()

            model = self.model
            model.load_state_dict(self.orig_state_dict)
            model = model.train().to(device)

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(trainable_params, lr=1e-5)

            inputs = self._get_inputs(batch_size, seq_length, device)

            def fn():
                for _ in range(steps):
                    optimizer.zero_grad()
                    outputs = model(**inputs)
                    loss = outputs.loss
                    loss.backward()
                    optimizer.step()

            return fn

        stats_dict = self._collect_metrics(create_train_compute_fn, "train")

        # Save the stats
        stats_file = self.analysis_root / "training_stats.json"
        with stats_file.open("w") as f:
            json.dump(stats_dict, f, indent=4)

        if is_world_process_zero() and self.logger is not None:
            self.logger.save(str(stats_file.absolute()), policy="now")

    def _collect_inference_metrics(self):
        logger.info("Collecting inference metrics")

        def create_inference_compute_fn(
            batch_size: int, seq_length: int, device: str, steps: int = 1
        ) -> Callable[[], None]:
            garbage_collection_cuda()

            model = self.model
            model.load_state_dict(self.orig_state_dict)
            model = model.eval().to(device)

            inputs = self._get_inputs(
                batch_size, seq_length, device, include_labels=False
            )

            def fn():
                for _ in range(steps):
                    with torch.no_grad():
                        model(**inputs)

            return fn

        stats_dict = self._collect_metrics(create_inference_compute_fn, "inference")

        # Save the stats
        stats_file = self.analysis_root / "inference_stats.json"
        with stats_file.open("w") as f:
            json.dump(stats_dict, f, indent=4)

        if is_world_process_zero() and self.logger is not None:
            self.logger.save(str(stats_file.absolute()), policy="now")

    def _collect_generation_metrics(self):
        logger.info("Collecting generation metrics")

        def create_generation_compute_fn(
            batch_size: int, seq_length: int, device: str, steps: int = 1
        ) -> Callable[[], None]:
            garbage_collection_cuda()

            model: PreTrainedModel = self.model
            model.load_state_dict(self.orig_state_dict)
            model = model.eval().to(device)

            inputs = self._get_inputs(batch_size, 1, device, include_labels=False)

            def fn():
                for _ in range(steps):
                    with torch.no_grad():
                        model.generate(
                            inputs["input_ids"],
                            attention_mask=inputs["attention_mask"],
                            min_length=seq_length,
                            max_length=seq_length,
                            do_sample=False,
                        )

            return fn

        stats_dict = self._collect_metrics(create_generation_compute_fn, "generation")

        # Save the stats
        stats_file = self.analysis_root / "generation_stats.json"
        with stats_file.open("w") as f:
            json.dump(stats_dict, f, indent=4)

        if is_world_process_zero() and self.logger is not None:
            self.logger.save(str(stats_file.absolute()), policy="now")

    @staticmethod
    def _get_inputs(
        batch_size: int, length: int, device: str, include_labels: bool = True
    ) -> Dict[str, torch.Tensor]:
        input_ids = torch.randint(
            2,
            1000,
            (batch_size, length),
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.ones(
            (batch_size, length),
            device=device,
            dtype=torch.int,
        )
        output = {"input_ids": input_ids, "attention_mask": attention_mask}
        if include_labels:
            labels = input_ids.clone()
            output["labels"] = labels

        return output

    @staticmethod
    def _measure_gpu_time(
        compute_fn: Callable, repeat: int, num_warm_ups: int = 20
    ) -> List[float]:
        # Borrowed from https://deci.ai/blog/measure-inference-time-deep-neural-networks/

        for _ in range(num_warm_ups):
            compute_fn()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

        timings = []
        for rep in range(repeat):
            starter.record()
            compute_fn()
            ender.record()

            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings.append(curr_time)

        return timings

    @staticmethod
    def _measure_cpu_time(compute_fn: Callable, repeat: int) -> List[float]:
        timings = []
        for rep in range(repeat):
            start = time.time()
            compute_fn()
            end = time.time()
            timings.append(end - start)

        return timings

    @staticmethod
    def _measure_memory(
        compute_fn: Callable, repeat: int, num_warm_ups: int = 10
    ) -> List[Dict[str, Any]]:
        for _ in range(num_warm_ups):
            compute_fn()

        torch.cuda.reset_peak_memory_stats()

        metrics = []
        for rep in range(repeat):
            tracker = MemoryTracker()
            tracker.start()
            compute_fn()
            obj = {}
            tracker.stop_and_update_metrics(obj)
            metrics.append(obj)

        return metrics

    def _find_optimal_batch_size(
        self, loop_func: Callable[[int], None], max_trials: int = 15
    ) -> int:
        """Find the optimal batch size for training.
        Borrowed from https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/tuner/batch_size_scaling.py
        """

        new_size = self.batch_size

        # progress_bar = tqdm.t

        low = 1
        high = None
        count = 0
        while True:
            garbage_collection_cuda()

            # reset after each try

            try:
                # run loop
                loop_func(new_size)
                count += 1
                if count > max_trials:
                    break
                # Double in size
                low = new_size
                if high:
                    if (high - low) <= 20:
                        break
                    midval = (high + low) // 2
                    new_size = midval
                else:
                    new_size *= 2

            except RuntimeError as exception:
                # Only these errors should trigger an adjustment
                if is_oom_error(exception):
                    # If we fail in power mode, half the size and return
                    garbage_collection_cuda()

                    high = new_size
                    midval = (high + low) // 2
                    new_size = midval

                    if high - low <= 1:
                        break
                else:
                    raise  # some other error not memory related

        return new_size
