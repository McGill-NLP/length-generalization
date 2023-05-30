from collections import deque, defaultdict, Counter
from pathlib import Path
import zipfile

import fire
import jsonlines
import requests

import numpy as np

import random

from datasets import load_dataset


def main(output_dir: str):
    ds = load_dataset("scan", "length", split="train+test")

    train = []
    test = []
    for d in ds:
        tgt = d["actions"]
        src = d["commands"]
        target_length = len(tgt.split())
        if target_length > 25:
            split = test
        else:
            split = train

        split.append(
            {
                "source": src,
                "target": tgt,
                "category": target_length,
            }
        )

    # Randomly sample 10% of the training data for validation
    random.shuffle(train)
    val_data = train[: (len(train) // 10)]
    train_data = train[(len(train) // 10 ):]

    # Print out the number of examples in each split
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test)}")

    # Print distribution of categories using counter
    for split, data in zip(["train", "val", "test"], [train_data, val_data, test]):
        print(f"{split} distribution:")
        print(Counter([d["category"] for d in data]))

    base_dir = Path(output_dir)
    ds_name = "scan"
    split_name = "len_tr25_ts48"
    output_dir = base_dir / ds_name / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_dir / "train.jsonl", mode="w") as writer:
        writer.write_all(train_data)

    with jsonlines.open(output_dir / "validation.jsonl", mode="w") as writer:
        writer.write_all(val_data)

    with jsonlines.open(output_dir / "test.jsonl", mode="w") as writer:
        writer.write_all(test)


if __name__ == "__main__":
    fire.Fire(main)
