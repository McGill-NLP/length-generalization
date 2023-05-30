import argparse
import itertools
import math
import random
from collections import deque, Counter
from pathlib import Path
from pprint import pprint
from typing import Deque, Dict, List

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


UNIQUE_TOKEN_LIST = None


def get_unique_tokens() -> List[str]:
    global UNIQUE_TOKEN_LIST
    if UNIQUE_TOKEN_LIST is None:

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("t5-base")

        repeat_num = 10

        UNIQUE_TOKEN_LIST = []

        ascii_chars = list(
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!$%&()*+,-./:;<=>"
        )
        for tok in ascii_chars:
            seq_str = " ".join([tok] * repeat_num)
            if len(tokenizer.tokenize(seq_str)) == repeat_num:
                UNIQUE_TOKEN_LIST.append(tok)

        valid_ascii_chars = UNIQUE_TOKEN_LIST[:]

        for tok, tok_id in tokenizer.vocab.items():
            tok_str = tok[1:] if tok.startswith("▁") else tok
            seq_str = " ".join([tok] * repeat_num)
            if (
                len(tokenizer.tokenize(seq_str)) == repeat_num
                and tok_str not in valid_ascii_chars
            ):
                UNIQUE_TOKEN_LIST.append(tok_str)

    return UNIQUE_TOKEN_LIST


def generate_count_task_data_repeat_single_char(
    seq_length: int, num_examples: int, modular: bool = True,
) -> List[Dict[str, str]]:
    samples: Deque[Dict[str, str]] = deque()

    unique_tokens = get_unique_tokens()[:num_examples]

    answer = seq_length % 10 if modular else seq_length

    for tok in unique_tokens:
        d = {
            "repeat": seq_length,
            "target_char": tok,
            "src_seq": " ".join([tok] * seq_length),
            "answer": answer,
            "cat": seq_length,
        }
        samples.append(d)

    return list(samples)


def generate_count_task_data_multi_char(
    seq_length: int, num_examples: int, modular: bool = True,
) -> List[Dict[str, str]]:
    samples: Deque[Dict[str, str]] = deque()

    unique_tokens = get_unique_tokens()[:26]

    answer = seq_length % 10 if modular else seq_length

    for _ in range(num_examples):
        token_ids = np.random.randint(0, len(unique_tokens), seq_length)
        tokens = [unique_tokens[i] for i in token_ids]

        d = {
            "repeat": seq_length,
            "src_seq": " ".join(tokens),
            "answer": answer,
            "cat": seq_length,
        }
        samples.append(d)

    return list(samples)

def generate_task_data(
    seq_length: int, num_examples: int, args: argparse.Namespace
) -> List[Dict[str, str]]:
    if args.count_task_type == "single_char":
        return generate_count_task_data_repeat_single_char(seq_length, num_examples, modular=args.modular)
    elif args.count_task_type == "multi_char":
        return generate_count_task_data_multi_char(seq_length, num_examples, modular=args.modular)
    else:
        raise ValueError(f"Invalid count task type: {args.count_task_type}")


def main(args: argparse.Namespace):
    set_seed(9842497)

    print(args)

    num_train = args.num_train
    num_train_validation = 1.15 * num_train

    min_length = args.train_min
    max_length = args.train_max

    train_and_validation = []
    length_bins = range(min_length, max_length + 1)
    num_examples_per_bin = num_train_validation / len(length_bins)
    for length in length_bins:
        num_examples = num_examples_per_bin
        data = generate_task_data(
            seq_length=length,
            num_examples=int(num_examples),
            args=args,
        )
        train_and_validation += data

    # Randomly split the data into train and validation
    random.shuffle(train_and_validation)
    train = train_and_validation[:num_train]
    validation = train_and_validation[num_train:]

    min_length = max_length + 1 if args.test_min is None else args.test_min
    max_length = max_length * 2 if args.test_max is None else args.test_max

    test = []
    length_bins = range(min_length, max_length + 1)
    if args.num_test is not None:
        num_examples_per_bin_test = args.num_test / len(length_bins)
    else:
        num_examples_per_bin_test = num_examples_per_bin

    for length in length_bins:
        num_examples = num_examples_per_bin
        bin_data = generate_task_data(
            seq_length=length,
            num_examples=int(num_examples),
            args=args,
        )

        if num_examples_per_bin_test < num_examples_per_bin:
            bin_data = random.sample(bin_data, int(num_examples_per_bin_test))

        test += bin_data

    random.shuffle(test)

    train_counter = Counter([d["cat"] for d in train])
    valid_counter = Counter([d["cat"] for d in validation])
    test_counter = Counter([d["cat"] for d in test])
    print("Train:")
    pprint(train_counter)
    print("Validation:")
    pprint(valid_counter)
    print("Test:")
    pprint(test_counter)

    if args.split_name is None:
        args.split_name = (
            f"count_task_{args.count_task_type}_tr{args.train_max}_ts{max_length}"
        )

    base_dir = Path(args.output_dir)
    ds_name = "sc_count" if not args.modular else "sc_count_mod"
    output_dir = base_dir / ds_name / args.split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_dir / "train.jsonl", mode="w") as writer:
        writer.write_all(train)

    with jsonlines.open(output_dir / "validation.jsonl", mode="w") as writer:
        writer.write_all(validation)

    with jsonlines.open(output_dir / "test.jsonl", mode="w") as writer:
        writer.write_all(test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate SeqCls version of the Count dataset"
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
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory",
        required=True,
    )

    parser.add_argument(
        "--split_name",
        type=str,
        help="Split name",
    )

    parser.add_argument(
        "--count_task_type",
        type=str,
        help="Type of count task",
    )

    parser.add_argument(
        "--modular",
        help="Whether to use modular arithmetic",
        action="store_true",
        default=False,
    )

    args = parser.parse_args()

    main(args)
