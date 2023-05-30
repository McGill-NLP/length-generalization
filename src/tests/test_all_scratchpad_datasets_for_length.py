import json
import random
import string
from pathlib import Path
from typing import List, Any, Dict

from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from common import Lazy, Params, ExperimentStage
from common.py_utils import load_config_object
from data import DataLoaderFactory
from tokenization_utils import Tokenizer
datasets_with_scratchpad_support = [
    ["sum", "len_tr8_ts16"],
    # ["pcfg", "md_productivity"],
    # ["s2s_addition", "len_tr8_ts16"],
    # ["s2s_poly", "n_terms_tr8_ts16"],
    # ["s2s_sort", "len_mltd_tr8_ts16"],
    # ["s2s_sort", "len_sngd_tr8_ts16"],
    # ["s2s_lego", "len_tr8_ts16"],
    # ["s2s_lego", "len_tr8_ts16_perm"],
    # ["s2s_parity", "len_tr8_ts16"],
    # ["clutrr", "bwd_short"],
    # ["scan", "len_tr25_ts48"],
    # ["s2s_copy", "rsc_tr20_ts40"],
    # ["s2s_copy", "rsc2x_tr20_ts40"],
    # ["s2s_copy", "rdc_tr20_ts40"],
    # ["s2s_copy", "cmc_tr20_ts40"],
    # ["s2s_copy", "cmc2x_tr20_ts40"],
    # ["s2s_reverse", "mc_tr20_ts40"],
    # ["s2s_reverse", "mc2x_tr20_ts40"],
    # ["s2s_reverse", "mcrv_tr20_ts40"],
]


CONFIG_TEMPLATE: Dict[str, Any] = {
    "append_vocab": "no",
    "data_root": "data",
    "decoder_only_block_size": 128,
    "decoder_only_group_samples": False,
    "decoder_only_include_position_ids": False,
    "decoder_only_input_output_sep_token": "",
    "decoder_only_mask_inputs": True,
    "decoder_only_padding_side": "right",
    "is_decoder_only": True,
    "max_source_length": 256,
    "max_target_length": 200000,
    "source_seq_key": "source",
    "target_seq_key": "target",
    "test_filename": "test.jsonl",
    "train_filename": "train.jsonl",
    "type": "seq2seq",
    "validation_filename": "validation.jsonl",
}

INSTANCE_PROCESSOR_TEMPLATE: Dict[str, Any] = {
    "include_scratchpad": True,
    "include_input": True,
    "include_computation": True,
    "include_output": True,
    "include_intermediate_variables": True,
    "include_remaining_input": True,
}

MAX_STATS = {
    tuple(key): {
        "input_ids": -1,
        "labels": -1,
        "name": "",
    }
    for key in datasets_with_scratchpad_support
}


def get_instance_processor_type(ds_name: str, split_name: str) -> str:
    if ds_name == "sum":
        return "s2s_sum"

    if ds_name == "s2s_sort":
        if split_name == "len_mltd_tr8_ts16":
            return "s2s_sort_multi_digit"
        elif split_name == "len_sngd_tr8_ts16":
            return "s2s_sort_single_digit"

    if ds_name == "pcfg":
        if split_name == "md_productivity":
            return "pcfg_bos"
        else:
            return "identity"

    return ds_name


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


def generate_all_scratchpad_configs(ds_name, split_name) -> List[Dict[str, Any]]:
    configs = []
    for include_scratchpad in [True, False]:
        if not include_scratchpad:
            configs.append({"include_scratchpad": include_scratchpad})
            continue

        if ds_name in ["scan", "s2s_copy", "s2s_reverse", "pcfg"]:
            continue

        scratchpad_bool_config = [
            True,  # include_input
            True,  # include_computation
            True,  # include_output
            True,  # include_intermediate_variables
            True,  # include_remaining_input
        ]
        all_bool_configs = [
            [not c for c in scratchpad_bool_config.copy()],
            scratchpad_bool_config.copy(),
            [
                False,  # include_input
                True,  # include_computation
                True,  # include_output
                False,  # include_intermediate_variables
                False,  # include_remaining_input
            ],
        ] + generate_boolean_configs(scratchpad_bool_config)

        if ds_name in ["s2s_sort", "s2s_lego"]:
            # Don't include intermediate variables for sort
            all_bool_configs = [modify_array(c, 3, False) for c in all_bool_configs]

        if ds_name in ["clutrr"]:
            # Only include computation for clutrr
            all_bool_configs = [
                [
                    False,  # include_input
                    True,  # include_computation
                    False,  # include_output
                    False,  # include_intermediate_variables
                    False,  # include_remaining_input
                ]
            ]

        # Make the configs unique by converting each to tuple
        # and then the list of configs to a set
        all_bool_configs = [tuple(c) for c in all_bool_configs]
        all_bool_configs = sorted(set(all_bool_configs))

        for config in all_bool_configs:
            configs.append(
                {
                    "include_scratchpad": include_scratchpad,
                    "include_input": config[0],
                    "include_computation": config[1],
                    "include_output": config[2],
                    "include_intermediate_variables": config[3],
                    "include_remaining_input": config[4],
                }
            )

    return configs


def generate_random_string(length: int) -> str:
    """
    Generate a random string of length `length` including alphanumeric characters and spaces.
    But no spaces in the beginning or end.
    """
    alphanumeric = string.ascii_lowercase + string.ascii_uppercase + string.digits
    letters = string.ascii_lowercase + string.ascii_uppercase
    result_str = " ".join(
        [
            "".join(
                random.choice(letters) if i == 0 else random.choice(alphanumeric)
                for i in range(random.randint(1, 10))
            )
            for _ in range(20)
        ]
    )
    return result_str[:length].strip()


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

            eval_result = dl_factory.instance_processor.evaluate_scratchpad(
                labels,
                ds[global_idx],
            )

            parsed_pred = dl_factory.instance_processor.extract_answer_from_prediction(
                labels
            )

            is_prediction_correct = dl_factory.instance_processor.is_prediction_correct(
                labels, ds[global_idx]
            )

            assert parsed_pred is not None

            assert (
                is_prediction_correct
            ), f"Prediction is incorrect for {ds[global_idx]}"
            for key in eval_result:
                if key != "failed_steps":
                    assert (
                        eval_result[key] == 1.0
                    ), f"Scratchpad evaluation failed for {ds[global_idx]} on {key}"


def get_scratchpad_config_name(config: Dict[str, bool]) -> str:
    keys_to_abrv = {
        "include_input": "i",
        "include_computation": "c",
        "include_output": "o",
        "include_intermediate_variables": "v",
        "include_remaining_input": "r",
    }
    keys_in_order = [
        "include_input",
        "include_computation",
        "include_output",
        "include_intermediate_variables",
        "include_remaining_input",
    ]
    filename = ""
    for key in keys_in_order:
        abrv = keys_to_abrv[key]
        if key not in config:
            print(config)
            raise ValueError(f"Key {key} not in config")
        val = str(int(config[key]))
        filename += f"{abrv}{val}_"

    return filename[:-1]


def test_dataset(ds_name: str, split_name: str) -> Dict[str, int]:
    # Create config
    minimal_template_config = load_config_object(
        [
            str(
                Path(__file__).parent.parent.parent
                / "configs"
                / "data"
                / "minimal_template.jsonnet"
            )
        ]
    )

    scratchpad_configs = generate_all_scratchpad_configs(ds_name, split_name)
    factory_configs = []
    for scratchpad_config in scratchpad_configs:
        instance_processor_config = scratchpad_config.copy()
        instance_processor_config.update(
            {
                "type": get_instance_processor_type(ds_name, split_name),
                "modulo_factor": 10,
            }
        )
        instance_processor_config.update(minimal_template_config)
        config = CONFIG_TEMPLATE.copy()
        config["debug_mode"] = True
        config.update(
            {
                "name": ds_name,
                "split": split_name,
                "instance_processor": instance_processor_config,
            }
        )
        factory_configs.append(config)

    stats = {}
    for factory_config in tqdm(factory_configs):
        config = factory_config
        try:
            scratchpad_config_name = get_scratchpad_config_name(
                config["instance_processor"]
            )
        except ValueError:
            scratchpad_config_name = "no_scratchpad"
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
        # Get training max input_ids length
        max_input_ids_length = max([len(x["input_ids"]) for x in ds])

        stats[scratchpad_config_name] = max_input_ids_length

    return stats


def test_all_scratchpad_datasets():
    set_seed(42)
    all_stats = {}
    for ds_name, split_name in tqdm(datasets_with_scratchpad_support):
        print(f"Testing {ds_name} {split_name}")
        stats = test_dataset(ds_name, split_name)

        all_stats[f"{ds_name},{split_name}"] = stats

    print("\n\n\n-----------------------------------------------")
    for name, stats in all_stats.items():
        print("\n\n\n")
        print(name)
        for key in sorted(stats.keys()):
            print(f"{key}: {stats[key]}")
        print("\n\n\n")

    # Update "scripts/scratchpad_stats.json" with the new stats if the file exists
    stats_file = Path("scripts/scratchpad_stats.json")
    if stats_file.exists():
        with open(stats_file, "r") as f:
            old_stats = json.load(f)
        old_stats.update(all_stats)
        with open(stats_file, "w") as f:
            json.dump(old_stats, f, indent=4)
    else:
        with open(stats_file, "w") as f:
            json.dump(all_stats, f, indent=4)



if __name__ == "__main__":
    test_all_scratchpad_datasets()
