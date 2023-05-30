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
                    "type": "s2s_lego",
                },
                "is_decoder_only": True,
                "max_source_length": 256,
                "max_target_length": 256,
                "name": "s2s_lego",
                "source_seq_key": "source",
                "split": "len_tr8_ts16",
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
        params=Params({"hf_model_name": "t5-small", "type": "pretrained"}),
    ).construct(dataset=dl_factory, experiment_root="experiments/base")

    dl_factory.set_tokenizer(tokenizer)

    paths = [
        dl_factory.get_ds_file_path(ExperimentStage.TRAINING),
        dl_factory.get_ds_file_path(ExperimentStage.VALIDATION),
        dl_factory.get_ds_file_path(ExperimentStage.TEST),
    ]

    # TRAINING stage
    for path in paths:
        print(f"Processing {path}")
        dc = dl_factory.get_collate_fn(ExperimentStage.TRAINING)
        ds = dl_factory.get_dataset(ExperimentStage.TRAINING, path)
        ds = ds.remove_columns(
            [
                c
                for c in ds.column_names
                if c not in ["input_ids", "labels", "attention_mask"]
            ]
        )

        dataloader = DataLoader(
            dataset=ds,
            batch_size=256,
            collate_fn=dc,
            drop_last=False,
            shuffle=False,
        )

        gold_answers = []

        for batch in tqdm(dataloader, total=len(dataloader)):
            batch_size = batch["input_ids"].shape[0]
            for i in range(batch_size):
                # input_str = tokenizer.decode(batch["input_ids"][i])
                # Exclude -100 from labels
                labels_str = tokenizer.decode(
                    batch["labels"][i][batch["labels"][i] != -100]
                )
                # List of ones and zeros
                gold_answer = (
                    labels_str.lower().split("the answer is")[1].split(".")[0].strip()
                )
                gold_answers.append(gold_answer)

        from collections import Counter

        print(Counter(gold_answers))

    print("Done")


if __name__ == "__main__":
    main()
