import json
import os
from typing import List, Any

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from common import Lazy, Params, ExperimentStage, py_utils
from data import DataLoaderFactory
from tokenization_utils import Tokenizer

datasets_list = [
    ("scan", "len_tr25_ts48"),
    ("clutrr", "bwd_short"),
    ("pcfg", "productivity"),
]

MAX_STATS = {
    tuple(key): {
        "input_ids": -1,
        "labels": -1,
        "name": "",
    }
    for key in datasets_list
}


def get_instance_processor_type(ds_name: str, split_name: str) -> str:
    if ds_name == "sum":
        return "s2s_sum"

    if ds_name == "s2s_sort":
        if split_name == "len_mltd_tr8_ts16":
            return "s2s_sort_multi_digit"
        elif split_name == "len_sngd_tr8_ts16":
            return "s2s_sort_single_digit"

    return ds_name


def generate_boolean_configs(config):
    configs = []
    for i in range(len(config)):
        new_config = config.copy()
        new_config[i] = not config[i]
        configs.append(new_config)
    return configs


def generate_boolean_configs(config):
    configs = []
    for i in range(len(config)):
        new_config = config.copy()
        new_config[i] = not config[i]
        configs.append(new_config)
    return configs


def modify_array(arr: List[Any], idx: int, val: Any) -> List[Any]:
    arr[idx] = val
    return arr


def test_ds(
    ds: Dataset,
    dl_factory: DataLoaderFactory,
    stage: ExperimentStage,
    tokenizer: Tokenizer,
) -> None:
    dataloader = DataLoader(
        ds.remove_columns(
            [
                c
                for c in ds.column_names
                if c not in ["input_ids", "labels", "attention_mask"]
            ]
        ),
        batch_size=10,
        shuffle=False,
        collate_fn=dl_factory.get_collate_fn(stage),
    )
    for batch_idx, batch in enumerate(dataloader):
        for i in range(len(batch["input_ids"])):
            global_idx = batch_idx * 10 + i
            labels = batch["labels"][i]
            labels = [l for l in labels if l >= 0]
            assert tokenizer.unk_token_id not in labels
            labels = tokenizer.decode(
                labels, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            is_prediction_correct = dl_factory.instance_processor.is_prediction_correct(
                labels, ds[global_idx]
            )

            assert (
                is_prediction_correct
            ), f"Prediction is incorrect for {ds[global_idx]}"


def test_dataset(ds_name: str, split_name: str) -> None:
    # Create config
    os.environ["APP_DS_NAME"] = ds_name
    os.environ["APP_DS_SPLIT"] = split_name
    os.environ["APP_SEED"] = "42"

    config_obj = py_utils.load_jsonnet_config(
        [
            "configs/base.jsonnet",
            "configs/data/seq2seq.jsonnet",
            f"configs/data/{ds_name}.jsonnet",
        ]
    )

    config = config_obj["dataset"]
    config["is_decoder_only"] = True
    # config["debug_mode"] = True
    dl_factory = DataLoaderFactory.from_params(Params(config.copy()))
    tokenizer = Lazy(
        Tokenizer,
        params=Params(
            {"hf_model_name": "t5-small", "type": "pretrained", "use_fast": True}
        ),
    ).construct(dataset=dl_factory, experiment_root="experiments/base")
    dl_factory.set_tokenizer(tokenizer)

    # Test the train set
    stage = ExperimentStage.TRAINING
    ds = dl_factory.get_dataset(
        path=dl_factory.get_ds_file_path(ExperimentStage.TRAINING),
        stage=stage,
    )
    ds = ds.shuffle().select(range(100))
    try:
        test_ds(ds, dl_factory, stage, tokenizer)
    except AssertionError as e:
        print(f"Failed for config: {config}")
        raise e
    # Get training max input_ids length
    max_input_ids_length = max([len(x["input_ids"]) for x in ds])
    max_labels_length = max([len(x["labels"]) for x in ds])

    # Update max stats
    key = (ds_name, split_name)
    if max_input_ids_length > MAX_STATS[key]["input_ids"]:
        MAX_STATS[key]["input_ids"] = max_input_ids_length
        MAX_STATS[key]["labels"] = max_labels_length
        MAX_STATS[key]["name"] = json.dumps(config, indent=4)

    # Test the test set
    stage = ExperimentStage.PREDICTION
    ds = dl_factory.get_dataset(
        path=dl_factory.get_ds_file_path(ExperimentStage.TEST),
        stage=stage,
    )
    ds = ds.shuffle().select(range(100))
    try:
        test_ds(ds, dl_factory, stage, tokenizer)
    except AssertionError as e:
        print(f"Failed for config: {config}")
        raise e

    print("--------------------------------------")
    print("Max input_ids length:", max_input_ids_length)
    print("Max labels length:", max_labels_length)
    print("--------------------------------------")


def test_all_datasets():
    set_seed(42)
    for ds_name, split_name in tqdm(datasets_list):
        print(f"Testing {ds_name} {split_name}")
        test_dataset(ds_name, split_name)

    print("All tests passed.")

    # Print max stats
    print("Max stats:")
    for key in MAX_STATS:
        print(key)
        print("\tinput_ids", MAX_STATS[key]["input_ids"])
        print("\tlabels", MAX_STATS[key]["labels"])
        print()


if __name__ == "__main__":
    test_all_datasets()
