from datasets import tqdm
from torch.utils.data import DataLoader

from common import Lazy, Params, ExperimentStage
from data import DataLoaderFactory
from tokenization_utils import Tokenizer


def main():
    dl_factory = DataLoaderFactory.from_params(
        Params(
            {
                "append_vocab": "no",
                "data_root": "data",
                "decoder_only_block_size": 128,
                "decoder_only_group_samples": False,
                "decoder_only_include_position_ids": False,
                "decoder_only_input_output_sep_token": "",
                "decoder_only_mask_inputs": True,
                "decoder_only_padding_side": "right",
                "instance_processor": {
                    "include_scratchpad": False,
                    "type": "s2s_parity",
                },
                "is_decoder_only": True,
                "max_source_length": 256,
                "max_target_length": 256,
                "name": "s2s_parity",
                "source_seq_key": "source",
                "split": "len_tr20_ts40",
                "target_seq_key": "target",
                "test_filename": "test.jsonl",
                "train_filename": "train.jsonl",
                "type": "seq2seq",
                "validation_filename": "validation.jsonl",
            }
        )
    )
    tokenizer = Lazy(
        Tokenizer,
        params=Params(
            {
                "hf_model_name": "t5-small",
                "type": "pretrained"
            }
        ),
    ).construct(dataset=dl_factory, experiment_root="experiments/base")

    dl_factory.set_tokenizer(tokenizer)

    paths = [
        dl_factory.get_ds_file_path(ExperimentStage.TRAINING),
        dl_factory.get_ds_file_path(ExperimentStage.VALIDATION),
        dl_factory.get_ds_file_path(ExperimentStage.TEST),
    ]

    # # TRAINING stage
    # for path in paths:
    #     print(f"Processing {path}")
    #     dc = dl_factory.get_collate_fn(ExperimentStage.TRAINING)
    #     ds = dl_factory.get_dataset(ExperimentStage.TRAINING, path)
    #     ds = ds.remove_columns(
    #         [
    #             c
    #             for c in ds.column_names
    #             if c not in ["input_ids", "labels", "attention_mask"]
    #         ]
    #     )
    #
    #
    #     dataloader = DataLoader(
    #         dataset=ds,
    #         batch_size=256,
    #         collate_fn=dc,
    #         drop_last=False,
    #         shuffle=False,
    #     )
    #
    #     for batch in tqdm(dataloader, total=len(dataloader)):
    #         batch_size = batch["input_ids"].shape[0]
    #         for i in range(batch_size):
    #             input_str = tokenizer.decode(batch["input_ids"][i])
    #             # Exclude -100 from labels
    #             labels_str = tokenizer.decode(
    #                 batch["labels"][i][batch["labels"][i] != -100]
    #             )
    #             # List of ones and zeros
    #             input_lst = input_str.split("[")[1].split("]")[0].strip().split(", ")
    #             input_lst = [int(x) for x in input_lst]
    #             gold_answer = labels_str.split("The answer is ")[1].split(".")[0]
    #             gold_answer = gold_answer == "Yes"
    #
    #             # Calculate parity
    #             parity = sum(input_lst) % 2 == 0
    #             if parity != gold_answer:
    #                 print(f"ERROR: {input_str} -> {labels_str}")
    #                 print(f"ERROR: {input_lst} -> {parity} != {gold_answer}")
    #                 raise ValueError("Parity mismatch")

    # Prediction stage
    for path in paths:
        print(f"Processing {path}")
        dc = dl_factory.get_collate_fn(ExperimentStage.PREDICTION)
        ds = dl_factory.get_dataset(ExperimentStage.PREDICTION, path)
        ds = ds.remove_columns(
            [
                c
                for c in ds.column_names
                if c not in ["input_ids", "attention_mask", "labels"]
            ]
        )

        dataloader = DataLoader(
            dataset=ds,
            batch_size=256,
            collate_fn=dc,
            drop_last=False,
            shuffle=False,
        )

        for batch in tqdm(dataloader, total=len(dataloader)):
            batch_size = batch["input_ids"].shape[0]
            for i in range(batch_size):
                input_str = tokenizer.decode(batch["input_ids"][i])
                labels_str = tokenizer.decode(
                    batch["labels"][i][batch["labels"][i] != -100]
                )
                # List of ones and zeros
                input_lst = input_str.split("[")[1].split("]")[0].strip().split(", ")
                input_lst = [int(x) for x in input_lst]
                gold_answer = labels_str.split("The answer is ")[1].split(".")[0]
                gold_answer = gold_answer == "Yes"

                # Calculate parity
                parity = sum(input_lst) % 2 == 0
                if parity != gold_answer:
                    print(f"ERROR: {input_str} -> {parity} != {gold_answer}")
                    raise ValueError("Parity mismatch")



    print("Done")



if __name__ == "__main__":
    main()
