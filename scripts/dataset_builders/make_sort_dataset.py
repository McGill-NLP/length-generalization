import argparse
import random
from collections import deque, Counter
from pathlib import Path
from pprint import pprint
from typing import Deque, Dict, List, Any

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
            "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
        )
        for tok in ascii_chars:
            seq_str = " ".join([tok] * repeat_num)
            if len(tokenizer.tokenize(seq_str)) == repeat_num:
                UNIQUE_TOKEN_LIST.append(tok)

        valid_ascii_chars = UNIQUE_TOKEN_LIST[:]

        for tok in tokenizer.vocab.keys():
            # tok_id = tokenizer.vocab[tok]
            tok_str = tok[1:] if tok.startswith("▁") else tok
            if len(set(list(tok_str)) & set(list("()*+,-./:;<=>\"'\\^_`{|}~"))) >= 1:
                continue
            seq_str = " ".join([tok] * repeat_num)
            if (
                len(tokenizer.tokenize(seq_str)) == repeat_num
                and tok_str not in valid_ascii_chars
            ):
                UNIQUE_TOKEN_LIST.append(tok_str)

    return UNIQUE_TOKEN_LIST


def generate_data_single_digit(
    seq_length: int, num_examples: int
) -> List[Dict[str, Any]]:

    unique_tokens = get_unique_tokens()[:50]

    samples: Deque[Dict[str, Any]] = deque()
    for _ in range(num_examples):
        token_ids = np.random.randint(0, len(unique_tokens), seq_length)
        unsorted_tokens = [unique_tokens[i] for i in token_ids]
        sorted_tokens = [unique_tokens[i] for i in np.sort(token_ids)]

        samples.append(
            {
                "token_ids": token_ids.tolist(),
                "src_seq": unsorted_tokens,
                "tgt_seq": sorted_tokens,
                "cat": seq_length,
            }
        )

    return list(samples)


def generate_data_multi_digit(
    seq_length: int, num_examples: int
) -> List[Dict[str, Any]]:
    samples: Deque[Dict[str, Any]] = deque()
    for _ in range(num_examples):
        nums = np.random.randint(10, 10000, seq_length)
        unsorted = nums.tolist()
        sorted = np.sort(nums).tolist()

        samples.append(
            {
                "src_seq": unsorted,
                "tgt_seq": sorted,
                "cat": seq_length,
            }
        )

    return list(samples)


def generate_data(
    seq_length: int, num_examples: int, args: argparse.Namespace
) -> List[Dict[str, Any]]:
    if args.single_digit:
        return generate_data_single_digit(seq_length, num_examples)
    else:
        return generate_data_multi_digit(seq_length, num_examples)


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
        data = generate_data(
            seq_length=length, num_examples=int(num_examples), args=args
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
    length_bins = range(min_length, max_length + 1)
    num_examples_per_bin = num_test / len(length_bins)
    for length in length_bins:
        num_examples = num_examples_per_bin
        test += generate_data(
            seq_length=length, num_examples=int(num_examples), args=args
        )

    random.shuffle(test)

    train_counter = Counter([d["cat"] for d in train])
    valid_counter = Counter([d["cat"] for d in validation])
    test_counter = Counter([d["cat"] for d in test])
    pprint(train_counter)
    pprint(valid_counter)
    pprint(test_counter)

    if args.output_dir is None:
        digit_str = "sngd" if args.single_digit else "mltd"
        base_dir = Path(f"sort_{args.split_name}_{digit_str}")
    else:
        base_dir = Path(args.output_dir)
    output_dir = base_dir / "s2s_sort" / args.split_name
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
        "--single_digit",
        help="Whether to use single digit numbers",
        action="store_true",
        default=False,
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
