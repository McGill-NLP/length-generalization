import gc
import pickle
from collections import defaultdict
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, tqdm

from analyzers import Analyzer
from analyzers.attention_kl_analyzer import torch_compute_symmetric_kl
from common import ExperimentStage

logger = getLogger("app")


def compute_max_attention_indices(attn_map: torch.Tensor) -> List[int]:
    seq_len = attn_map.shape[0]

    target_indices = []
    for query_idx in range(seq_len):
        attn_scores_vector = attn_map[query_idx, : query_idx + 1]
        target_indices.append(attn_scores_vector.argmax().item())

    return target_indices


def compute_mean_kl_distance_two_score_matrix(
    score_matrix1: np.ndarray, score_matrix2: np.ndarray
):
    seq_len = score_matrix1.shape[0]
    assert score_matrix1.shape == score_matrix2.shape

    kls = []
    for l in range(seq_len):
        score_vec1 = score_matrix1[l, : l + 1]
        score_vec2 = score_matrix2[l, : l + 1]

        # Convert to probabilities by applying softmax
        score_vec1 = torch.softmax(torch.tensor(score_vec1), dim=0)
        score_vec2 = torch.softmax(torch.tensor(score_vec2), dim=0)

        sym_kl = torch_compute_symmetric_kl(score_vec1, score_vec2).numpy()
        kls.append(sym_kl)

    mean_kl = np.mean(kls)
    return mean_kl


@Analyzer.register("attention_aggregate")
class AttentionAggregateAnalyzer(Analyzer):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def analyze(self):
        self._compute_aggregate_measures()

    def _compute_aggregate_measures(self):
        def get_ds(split: str) -> Dataset:
            stage = ExperimentStage.TRAINING
            ds_path = self.dl_factory.get_ds_file_path(
                ExperimentStage.from_split(split)
            )
            ds = self.dl_factory.get_dataset(path=ds_path, stage=stage)
            return ds

        def load_saved_dataset(path: Path) -> Dataset:
            if (path / "dataset.parquet").exists():
                return Dataset.from_parquet(str(path / "dataset.parquet"))
            elif (path / "dataset.jsonl").exists():
                return Dataset.from_json(str(path / "dataset.jsonl"))
            else:
                raise ValueError(f"Could not find dataset in {path}")

        validation_ds = get_ds("validation")
        test_ds = get_ds("test")
        validation_categories = validation_ds.unique("category")
        test_categories = test_ds.unique("category")

        all_categories = list(set(validation_categories + test_categories))
        all_categories.sort()

        df_data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        for cat in tqdm(all_categories, desc="Computing Aggregate Measures"):
            attn_analysis_dir = (
                self.exp_root
                / "analysis"
                / "AttentionAnalyzer__test"
                / f"category__{cat}"
            )

            all_attention_map_names = [
                "scores",
                "scores_before",
                "scores_pass",
                "scores_rot",
            ]

            attention_map_names = [
                name
                for name in all_attention_map_names
                if (attn_analysis_dir / f"attention_maps_{name}.pt").exists()
            ]

            if (
                not (attn_analysis_dir / "dataset.jsonl").exists()
                and not (attn_analysis_dir / "dataset.parquet").exists()
            ) or len(attention_map_names) == 0:
                logger.warning(f"Skipping {cat} because it does not exist")
                continue

            # Load the dataset
            ds = load_saved_dataset(attn_analysis_dir)
            # Load tokenization_infos.pkl
            with open(attn_analysis_dir / "tokenization_infos.pkl", "rb") as f:
                tokenization_infos = pickle.load(f)

            self._compute_max_attention_index_measures(
                attention_map_names, attn_analysis_dir, cat, df_data, ds, tokenization_infos
            )

            if (
                "scores" in attention_map_names
                and "scores_before" in attention_map_names
            ):
                self._compute_score_before_after_measures(
                    attn_analysis_dir, cat, df_data, ds
                )

        # Save dataframes
        for name, data in df_data.items():
            df = pd.DataFrame(data)
            csv_file = self.analysis_root / f"{name}.csv"
            df.to_csv(csv_file, index=False)
            if self.logger is not None:
                self.logger.save(str(csv_file.absolute()))

    @staticmethod
    def _compute_score_before_after_measures(attn_analysis_dir, cat, df_data, ds):
        # Compute the KL distance between the two attention maps
        attn_map_scores = torch.load(attn_analysis_dir / f"attention_maps_scores.pt")
        attn_map_scores_before = torch.load(
            attn_analysis_dir / f"attention_maps_scores_before.pt"
        )
        # Make sure the attention maps are in the same order as the dataset
        assert len(attn_map_scores) == len(ds)
        assert len(attn_map_scores_before) == len(ds)
        num_layers = len(attn_map_scores[0])
        num_heads = attn_map_scores[0][0].shape[1]
        for sample_idx in range(len(ds)):
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    attn_matrix_scores = attn_map_scores[sample_idx][layer_idx][
                        0, head_idx
                    ]
                    attn_matrix_scores_before = attn_map_scores_before[sample_idx][
                        layer_idx
                    ][0, head_idx]

                    kl = compute_mean_kl_distance_two_score_matrix(
                        attn_matrix_scores, attn_matrix_scores_before
                    )

                    df_data["before_after_kl"].append(
                        {
                            "category": cat,
                            "sample_idx": sample_idx,
                            "layer_idx": layer_idx,
                            "kl": kl,
                        }
                    )
        del attn_map_scores
        del attn_map_scores_before
        gc.collect()

    @staticmethod
    def _compute_max_attention_index_measures(
        attention_map_names, attn_analysis_dir, cat, df_data, ds, tokenization_infos
    ):
        for attn_map_name in attention_map_names:
            # Load the huggingface attention maps
            # format:
            # *attn_map = [ sample_1_attn_map, sample_2_attn_map, ... ]
            # sample_i_attn_map = [ layer_1_attn_map, layer_2_attn_map, ... ]
            # layer_i_attn_map = tensor of shape (1, num_heads, seq_len, seq_len)
            attn_map_path = attn_analysis_dir / f"attention_maps_{attn_map_name}.pt"
            if not attn_map_path.exists():
                logger.warning(f"Skipping {attn_map_name} because it does not exist")
                continue

            attn_map = torch.load(attn_map_path)

            # Make sure the attention maps are in the same order as the dataset
            assert len(attn_map) == len(ds)

            num_layers = len(attn_map[0])
            num_heads = attn_map[0][0].shape[1]

            for sample_idx in range(len(ds)):
                tok_info = tokenization_infos[sample_idx]
                for layer_idx in range(num_layers):
                    for head_idx in range(num_heads):
                        attn_matrix = attn_map[sample_idx][layer_idx][0, head_idx]

                        # Assert attn_matrix dimensions matches number of tokens
                        assert attn_matrix.shape[0] == len(tok_info["encoding"].tokens())

                        max_attn_idx = compute_max_attention_indices(attn_matrix)
                        max_attn_idx = "-".join([str(d) for d in max_attn_idx])

                        df_data[f"max_attn_idx__{attn_map_name}"].append(
                            {
                                "category": cat,
                                "sample_idx": sample_idx,
                                "head_idx": head_idx,
                                "layer_idx": layer_idx,
                                "max_attn_idx": max_attn_idx,
                            }
                        )

            del attn_map
            gc.collect()
