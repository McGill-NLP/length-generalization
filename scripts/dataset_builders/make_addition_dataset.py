import argparse
import itertools
import random
from collections import deque, Counter
from pathlib import Path
from pprint import pprint
from typing import Deque, Dict, List, Any, Tuple

import jsonlines
import numpy as np


def set_seed(seed: int):
    """
    # Taken from https://huggingface.co/transformers/v3.0.2/_modules/transformers/trainer_utils.html
    Helper function for reproducible behavior to set the seed in ``random``, ``numpy``, ``torch`` and/or ``tf``
    (if installed).

    Args:
        seed (:obj:`int`): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)

    # import torch
    #
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # # ^^ safe to call this function even if cuda is not available
    # import tensorflow as tf
    #
    # tf.random.set_seed(seed)


def generate_data_for_bin(
    a_num_digits: int, b_num_digits: int, num_examples: int
) -> List[Dict[str, Any]]:
    samples: Deque[Dict[str, Any]] = deque()
    for _ in range(num_examples):
        low = 10 ** (a_num_digits - 1)
        if low == 1:
            low = 0
        a = np.random.randint(
            low,
            10 ** (a_num_digits),
        )  # second arg is exclusive with randint

        low = 10 ** (b_num_digits - 1)
        if low == 1:
            low = 0
        b = np.random.randint(
            low, 10 ** (b_num_digits)
        )  # second arg is exclusive with randint

        samples.append(
            {
                "a": a,
                "b": b,
                "sum": a + b,
                "cat": f"{a_num_digits}-by-{b_num_digits}",
            }
        )

    return list(samples)


def generate_data(
    seq_lengths: Tuple[int, int], num_examples: int, args: argparse.Namespace
) -> List[Dict[str, Any]]:
    a_num_digits, b_num_digits = seq_lengths
    return generate_data_for_bin(a_num_digits, b_num_digits, num_examples)


def main(args: argparse.Namespace):
    set_seed(9842497)

    print(args)

    num_train = args.num_train
    num_train_validation = 1.15 * num_train

    min_length = args.train_min
    max_length = args.train_max

    train_and_validation = []
    length_bins = list(
        itertools.product(
            range(min_length, max_length + 1), range(min_length, max_length + 1)
        )
    )
    num_examples_per_bin = num_train_validation / len(length_bins)
    for length in length_bins:
        num_examples = num_examples_per_bin
        data = generate_data(
            seq_lengths=length, num_examples=int(num_examples), args=args
        )
        train_and_validation += data

    # Randomly split the data into train and validation
    random.shuffle(train_and_validation)
    train = train_and_validation[:num_train]
    validation = train_and_validation[num_train:]

    num_test = args.num_test

    min_length = max_length + 1 if args.test_min is None else args.test_min
    max_length = max_length * 2 if args.test_max is None else args.test_max

    test = []
    length_bins = list(
        itertools.product(
            range(min_length, max_length + 1), range(args.train_min, max_length + 1)
        )
    )
    length_bins += list(
        itertools.product(
            range(args.train_min, max_length + 1), range(min_length, max_length + 1)
        )
    )
    length_bins = sorted(set(length_bins))
    assert all(
        (
            (a >= min_length and a <= max_length)
            or (b >= min_length and b <= max_length)
        )  # at least one operand is OOD
        for a, b in length_bins
    )
    num_examples_per_bin = num_test / len(length_bins)
    for length in length_bins:
        num_examples = num_examples_per_bin
        test += generate_data(
            seq_lengths=length, num_examples=int(num_examples), args=args
        )

    random.shuffle(test)

    train_counter = Counter([d["cat"] for d in train])
    valid_counter = Counter([d["cat"] for d in validation])
    test_counter = Counter([d["cat"] for d in test])
    pprint(train_counter)
    pprint(valid_counter)
    pprint(test_counter)

    if args.output_dir is None:
        base_dir = Path(f"addition_{args.split_name}")
    else:
        base_dir = Path(args.output_dir)
    output_dir = base_dir / "s2s_addition" / args.split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_dir / "train.jsonl", mode="w") as writer:
        writer.write_all(train)

    with jsonlines.open(output_dir / "validation.jsonl", mode="w") as writer:
        writer.write_all(validation)

    with jsonlines.open(output_dir / "test.jsonl", mode="w") as writer:
        writer.write_all(test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Seq2seq version of sort dataset"
    )

    parser.add_argument(
        "--train_min",
        metavar="MIN",
        type=int,
        help="Max # variables in train",
        default=4,
    )

    parser.add_argument(
        "--train_max",
        metavar="MAX",
        type=int,
        help="Max # variables in train",
        required=True,
    )

    parser.add_argument(
        "--test_min",
        metavar="MIN",
        type=int,
        help="Max # variables in test",
    )

    parser.add_argument(
        "--test_max",
        metavar="MAX",
        type=int,
        help="Max # variables in test",
    )

    parser.add_argument(
        "--num_train",
        metavar="NUM",
        type=int,
        help="# generated examples in train",
        required=True,
    )

    parser.add_argument(
        "--num_test",
        metavar="NUM",
        type=int,
        help="# generated examples in test",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory",
    )

    parser.add_argument(
        "--split_name",
        type=str,
        help="Split name",
        required=True,
    )

    args = parser.parse_args()

    main(args)
