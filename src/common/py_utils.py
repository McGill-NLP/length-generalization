import hashlib
import logging
import os
from argparse import Namespace
from collections import MutableMapping
from typing import Dict, Any, List

import numpy as np

from common import JsonDict


class NewLineFormatter(logging.Formatter):
    def __init__(self, fmt, datefmt=None):
        super().__init__(fmt, datefmt)

    def format(self, record):
        msg = logging.Formatter.format(self, record)

        if record.message != "":
            parts = msg.split(record.message)
            msg = msg.replace("\n", "\n" + parts[0])

        return msg


# def unique_experiment_name(config):
#     configs = "_".join(
#         [os.path.splitext(os.path.basename(p))[0] for p in config.config_filenames]
#     )
#
#     unique_name = f"{configs}"
#
#     return unique_name


def unique_experiment_name(config):
    configs = "_".join(
        [os.path.splitext(os.path.basename(p))[0] for p in config["config_filenames"]]
    )

    unique_name = f"{configs}"

    return unique_name


def unique_experiment_name_from_filenames(config_filenames):
    configs = "_".join(
        [os.path.splitext(os.path.basename(p))[0] for p in config_filenames]
    )

    unique_name = f"{configs}"

    return unique_name


def get_num_total_lines(p):
    import subprocess

    # result = subprocess.run(['wc', '-l', p], stdout=subprocess.PIPE).stdout.decode('utf-8')
    if not isinstance(p, str):
        p = str(p)

    result = subprocess.check_output(["wc", "-l", p])
    if isinstance(result, bytes):
        result = result.decode("utf-8")

    return int(result.strip().split(" ")[0])


def softmax(X, theta=1.0, axis=None):
    """
    Copyright: https://nolanbconaway.github.io/blog/2017/softmax-numpy.html
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    import numpy as np

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def flatten_dict(params: Dict[Any, Any], delimiter: str = "/") -> Dict[str, Any]:
    def _dict_generator(input_dict, prefixes=None):
        prefixes = prefixes[:] if prefixes else []
        if isinstance(input_dict, MutableMapping):
            for key, value in input_dict.items():
                key = str(key)
                if isinstance(value, (MutableMapping, Namespace)):
                    value = vars(value) if isinstance(value, Namespace) else value
                    yield from _dict_generator(value, prefixes + [key])
                else:
                    yield prefixes + [key, value if value is not None else str(None)]
        else:
            yield prefixes + [input_dict if input_dict is None else str(input_dict)]

    return {delimiter.join(keys): val for *keys, val in _dict_generator(params)}


PARAMETER_NUM_UNITS = [" ", "K", "M", "B", "T"]


def get_human_readable_count(number: int) -> str:
    """
    Abbreviates an integer number with K, M, B, T for thousands, millions,
    billions and trillions, respectively.

    Examples:
        >>> get_human_readable_count(123)
        '123  '
        >>> get_human_readable_count(1234)  # (one thousand)
        '1.2 K'
        >>> get_human_readable_count(2e6)   # (two million)
        '2.0 M'
        >>> get_human_readable_count(3e9)   # (three billion)
        '3.0 B'
        >>> get_human_readable_count(4e14)  # (four hundred trillion)
        '400 T'
        >>> get_human_readable_count(5e15)  # (more than trillion)
        '5,000 T'

    Args:
        number: a positive integer number

    Return:
        A string formatted according to the pattern described above.

    """
    assert number >= 0
    labels = PARAMETER_NUM_UNITS
    num_digits = int(np.floor(np.log10(number)) + 1 if number > 0 else 1)
    num_groups = int(np.ceil(num_digits / 3))
    num_groups = min(num_groups, len(labels))  # don't abbreviate beyond trillions
    shift = -3 * (num_groups - 1)
    number = number * (10**shift)
    index = num_groups - 1
    if index < 1 or number >= 100:
        return f"{int(number):,d} {labels[index]}"

    return f"{number:,.1f} {labels[index]}"


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def load_config_object(filenames: List[str]) -> JsonDict:
    import _jsonnet
    import json

    ext_vars = {k: v for k, v in os.environ.items() if k.startswith("APP_")}
    ext_vars["seed"] = os.environ.get("APP_SEED", "123")
    jsonnet_str = "+".join([f'(import "{f}")' for f in filenames])
    json_str = _jsonnet.evaluate_snippet("snippet", jsonnet_str, ext_vars=ext_vars)
    config: Dict[str, Any] = json.loads(json_str)
    config["config_filenames"] = filenames
    return config


def load_jsonnet_config(filenames: List[str]) -> JsonDict:
    import _jsonnet
    import json

    ext_vars = {k: v for k, v in os.environ.items() if k.startswith("APP_")}
    ext_vars["seed"] = os.environ.get("APP_SEED", "12345")
    jsonnet_str = "+".join([f'(import "{f}")' for f in filenames])
    json_str = _jsonnet.evaluate_snippet("snippet", jsonnet_str, ext_vars=ext_vars)
    config: Dict[str, Any] = json.loads(json_str)
    return config


def eval_bool_env_var(key: str) -> bool:
    return os.environ.get(key, "False").lower() in ["true", "1", "yes", "y", "t"]


def get_run_name_from_config_obj(config_obj: Dict[str, Any], sep: str = ".") -> str:
    def param_name(param: str) -> str:
        param_parts = param.split(sep)
        if len(param_parts) == 1:
            return param[:2] + param[-2:]
        last_part = param_parts[-1]
        name = sep.join(
            [p[0] for p in param_parts[:-1]] + [last_part[:2] + last_part[-2:]]
        )
        return name

    config_pairs = [(param_name(k), v) for k, v in config_obj.items()]
    run_name = sorted(config_pairs, key=lambda x: x[0])
    run_name = "__".join(f"{k}-{str(v)}" for k, v in run_name)
    return run_name


def create_md5_hash(inp: str):
    # Create MD5 hash object
    md5 = hashlib.md5()
    # Update the hash with the string
    md5.update(inp.encode("utf-8"))
    # Get the hexadecimal representation of the hash
    return md5.hexdigest()


def generate_deterministic_hp_run_id(sweep_name: str, run_name: str) -> str:
    hash_str = create_md5_hash(sweep_name + "__" + run_name)
    return "sw_" + hash_str
