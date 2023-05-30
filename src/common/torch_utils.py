import gc
import os
from typing import List

import torch


def pad_batches_to_same_length(
    batches: List[torch.Tensor], padding_value: int = 0
) -> torch.Tensor:
    num_batches = len(batches)
    batch_size = batches[0].shape[0]
    max_length = max(b.shape[1] for b in batches)
    out_dims = (batch_size * num_batches, max_length, *batches[0].shape[2:])
    out_tensor = batches[0].new(*out_dims).fill_(padding_value)
    for i, b in enumerate(batches):
        length = b.shape[1]
        out_tensor[i * batch_size : (i + 1) * batch_size, :length] = b
    return out_tensor


def get_rank() -> int:
    from transformers import is_torch_tpu_available
    import torch.distributed as dist

    if is_torch_tpu_available():
        import torch_xla.core.xla_model as xm

        return xm.get_ordinal()
    elif dist.is_available():
        if dist.is_initialized():
            return dist.get_rank()
        else:
            return int(os.environ.get("RANK", 0))

    return -1


def is_world_process_zero() -> bool:
    return get_rank() in [0, -1]


# based on https://github.com/Lightning-AI/lightning/blob/master/src/lightning/pytorch/utilities/memory.py
def is_oom_error(exception: BaseException) -> bool:
    return (
        is_cuda_out_of_memory(exception)
        or is_cudnn_snafu(exception)
        or is_out_of_cpu_memory(exception)
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cuda_out_of_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "CUDA" in exception.args[0]
        and "out of memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def is_cudnn_snafu(exception: BaseException) -> bool:
    # For/because of https://github.com/pytorch/pytorch/issues/4107
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "cuDNN error: CUDNN_STATUS_NOT_SUPPORTED." in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/cpu_memory.py
def is_out_of_cpu_memory(exception: BaseException) -> bool:
    return (
        isinstance(exception, RuntimeError)
        and len(exception.args) == 1
        and "DefaultCPUAllocator: can't allocate memory" in exception.args[0]
    )


# based on https://github.com/BlackHC/toma/blob/master/toma/torch_cuda_memory.py
def garbage_collection_cuda() -> None:
    """Garbage collection Torch (CUDA) memory."""
    gc.collect()
    try:
        # This is the last thing that should cause an OOM error, but seemingly it can.
        torch.cuda.empty_cache()
    except RuntimeError as exception:
        if not is_oom_error(exception):
            # Only handle OOM errors
            raise
