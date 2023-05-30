import random
import zipfile
from collections import deque, defaultdict, Counter
from pathlib import Path

import fire
import jsonlines
import requests

func_names = {
    "copy",
    "reverse",
    "shift",
    "swap",
    "repeat",
    "echo",
    "append",
    "prepend",
    "remove_first",
    "remove_second",
    "swap_first_last",
}


def main(output_dir: str):
    # Download repo zip file
    url = "https://github.com/i-machine-think/am-i-compositional/archive/refs/heads/master.zip"
    repo_dir = Path("am-i-compositional-master")
    if not repo_dir.exists():
        r = requests.get(url)
        with open("am-i-compositional.zip", "wb") as f:
            f.write(r.content)

        # Unzip repo
        with zipfile.ZipFile("am-i-compositional.zip", "r") as zip_ref:
            zip_ref.extractall(".")

    productivity_split_dir = repo_dir / "data" / "pcfgset" / "productivity"

    def read_split(split):
        data = deque()
        with open(productivity_split_dir / f"{split}.src") as fsrc, open(
            productivity_split_dir / f"{split}.tgt"
        ) as ftgt:
            for src, tgt in zip(fsrc, ftgt):
                src = src.strip()
                # Count the number of functions in the source
                count_dict = defaultdict(int)
                for tok in src.split():
                    if tok in func_names:
                        count_dict[tok] += 1

                num_total_functions = sum(count_dict.values())

                data.append(
                    {
                        "source": src.strip(),
                        "target": tgt.strip(),
                        "category": num_total_functions,
                    }
                )
        return list(data)

    train_data = read_split("train")
    test = read_split("test")

    # Build vocab
    vocab = set()
    for d in train_data:
        for tok in d["source"].split():
            if tok not in func_names and tok != ",":
                vocab.add(tok)

        for tok in d["target"].split():
            if tok not in func_names and tok != ",":
                vocab.add(tok)

    for d in test:
        for tok in d["source"].split():
            if tok not in func_names and tok != ",":
                assert tok in vocab, f"Found new token {tok} in test set"

        for tok in d["target"].split():
            if tok not in func_names and tok != ",":
                assert tok in vocab, f"Found new token {tok} in test set"

    vocab = sorted(list(vocab))
    print(f"Vocab size: {len(vocab)}")

    # Randomly sample 10% of the training data for validation
    random.shuffle(train_data)
    val_data = train_data[: (len(train_data) // 10)]
    train_data = train_data[(len(train_data) // 10) :]

    # Print out the number of examples in each split
    print(f"Train: {len(train_data)}")
    print(f"Val: {len(val_data)}")
    print(f"Test: {len(test)}")

    # Print distribution of categories using counter
    for split, data in zip(["train", "val", "test"], [train_data, val_data, test]):
        print(f"{split} distribution:")
        print(Counter([d["category"] for d in data]))

    base_dir = Path(output_dir)
    ds_name = "pcfg"
    split_name = "productivity"
    output_dir = base_dir / ds_name / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with jsonlines.open(output_dir / "train.jsonl", mode="w") as writer:
        writer.write_all(train_data)

    with jsonlines.open(output_dir / "validation.jsonl", mode="w") as writer:
        writer.write_all(val_data)

    with jsonlines.open(output_dir / "test.jsonl", mode="w") as writer:
        writer.write_all(test)

    with open(output_dir / "vocab.src.txt", "w") as f:
        f.write("\n".join(vocab))


if __name__ == "__main__":
    fire.Fire(main)
