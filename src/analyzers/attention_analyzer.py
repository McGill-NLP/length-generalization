import json
import pickle
from collections import defaultdict
from typing import Optional

import torch
from datasets import Dataset, tqdm
from transformers import PreTrainedModel

from analyzers import Analyzer
from common import ExperimentStage
from common.torch_utils import is_world_process_zero


@Analyzer.register("attention")
class AttentionAnalyzer(Analyzer):
    def __init__(self, num_examples_per_category: Optional[int] = 10, **kwargs):
        super().__init__(**kwargs)
        self.num_examples_per_category = num_examples_per_category

    def analyze(self):
        self._analyze_attentions()

    def _analyze_attentions(self):
        def get_ds(split: str) -> Dataset:
            stage = ExperimentStage.TRAINING
            ds_path = self.dl_factory.get_ds_file_path(
                ExperimentStage.from_split(split)
            )
            ds = self.dl_factory.get_dataset(path=ds_path, stage=stage)
            return ds

        validation_ds = get_ds("validation")
        test_ds = get_ds("test")
        validation_categories = validation_ds.unique("category")
        test_categories = test_ds.unique("category")

        all_categories = list(set(validation_categories + test_categories))
        all_categories.sort()

        # Put model in eval
        model: PreTrainedModel = self.model
        model = model.eval()

        for cat in tqdm(all_categories, desc="Computing attentions"):
            output_dir = self.analysis_root / f"category__{cat}"
            output_dir.mkdir(parents=True, exist_ok=True)
            metadata_file = output_dir / f"attn_metadata_{cat}.json"

            if metadata_file.exists():
                # Make sure the file uploaded to the logger
                if is_world_process_zero() and self.logger is not None:
                    self.logger.save(str(metadata_file.absolute()))

            ds = validation_ds if cat in validation_categories else test_ds
            ds = ds.filter(lambda x: x["category"] == cat, num_proc=4)
            ds = ds.shuffle(seed=42)
            ds = ds.select(range(min(self.num_examples_per_category, len(ds))))

            # Save the dataset
            ds.to_parquet(output_dir / "dataset.parquet")

            attn_maps = defaultdict(list)
            tokenization_infos = []
            for i in range(len(ds)):
                input_ids = ds["input_ids"][i]
                attention_mask = ds["attention_mask"][i]

                # Save tokenization info
                input_str = self.tokenizer.decode(
                    input_ids,
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                encoding = self.tokenizer(
                    input_str, add_special_tokens=False, truncation=False
                )
                assert input_ids == encoding.input_ids
                assert len(encoding.tokens()) == len(input_ids)
                tokenization_infos.append(
                    self.dl_factory.instance_processor.get_tokenization_info(
                        ds[i], encoding, input_str
                    )
                )

                # Get the attention map
                input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
                attention_mask = (
                    torch.tensor(attention_mask).unsqueeze(0).to(model.device)
                )

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                )
                attention_map_keys = sorted(outputs.attentions[0].keys())
                for key in attention_map_keys:

                    attentions = [
                        t[key].detach().cpu().numpy() for t in outputs.attentions
                    ]

                    # Assert the attention_map dimension matches the number tokens
                    for attn in attentions:
                        assert attn.shape[-1] == len(
                            encoding.tokens()
                        ), f"{attn.shape} != {len(encoding.tokens())}"

                    attentions = tuple(attentions)
                    attn_maps[key].append(attentions)

            # Save the attention maps
            for key in attn_maps:
                attn = tuple(attn_maps[key])
                torch.save(attn, output_dir / f"attention_maps_{key}.pt")

            # Save the tokenization infos as Python pickle
            with open(output_dir / "tokenization_infos.pkl", "wb") as f:
                pickle.dump(tokenization_infos, f)

            # Create a metadata file containing the address of logged files
            if is_world_process_zero() and self.logger is not None:
                metadata = {
                    "category": cat,
                    "tokenization_infos": str(
                        (output_dir / "tokenization_infos.pkl").absolute()
                    ),
                    "dataset": str((output_dir / "dataset.parquet").absolute()),
                    **{
                        f"attention_maps_{key}": str(
                            (output_dir / f"attention_maps_{key}.pt").absolute()
                        )
                        for key in attn_maps
                    },
                }
                # Save the metadata file
                with open(metadata_file, "w") as f:
                    json.dump(metadata, f, indent=4)

                # Log the metadata file
                self.logger.save(str(metadata_file.absolute()))
