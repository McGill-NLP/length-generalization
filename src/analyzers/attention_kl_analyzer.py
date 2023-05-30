import itertools
from logging import getLogger
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, tqdm

from analyzers import Analyzer
from common import ExperimentStage

logger = getLogger("app")


def torch_compute_symmetric_kl(p, q):
    assert p.shape == q.shape
    p = torch.tensor(p)
    q = torch.tensor(q)
    kl = torch.nn.KLDivLoss(reduction="sum", log_target=False)
    return kl(p.log(), q) + kl(q.log(), p)


def torch_compute_jensen_shannon_divergence(p, q):
    assert p.shape == q.shape
    p = torch.tensor(p)
    q = torch.tensor(q)
    m = 0.5 * (p + q)
    kl = torch.nn.KLDivLoss(reduction="sum", log_target=False)
    return 0.5 * (kl(p.log(), m) + kl(q.log(), m))


def compute_distance_two_matrix(m1: torch.Tensor, m2: torch.Tensor):
    seq_len = m1.shape[0]
    assert m1.shape == m2.shape

    kls = []
    for l in range(seq_len):
        v1 = m1[l, : l + 1]
        v2 = m2[l, : l + 1]
        sym_kl = torch_compute_symmetric_kl(v1, v2).numpy()
        kls.append(sym_kl)
    return np.mean(kls)


def compute_jsd_two_matrix(m1: torch.Tensor, m2: torch.Tensor):
    seq_len = m1.shape[0]
    assert m1.shape == m2.shape

    kls = []
    for l in range(seq_len):
        v1 = m1[l, : l + 1]
        v2 = m2[l, : l + 1]
        sym_kl = torch_compute_jensen_shannon_divergence(v1, v2).numpy()
        kls.append(sym_kl)
    return np.mean(kls)


@Analyzer.register("attention_kl")
class AttentionKLAnalyzer(Analyzer):
    def __init__(
        self,
        no_pe_run_ids: Dict[str, str],
        seed: int,
        attention_analysis_root_dir: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # { "283984": "...id...", "284000": "...id..." }
        self.no_pe_seed_to_run_id = no_pe_run_ids
        self.seed = seed



        # Make sure seeds are ints
        self.no_pe_seed_to_run_id = {
            int(k): v for k, v in self.no_pe_seed_to_run_id.items()
        }

        if seed not in self.no_pe_seed_to_run_id:
            logger.info(f"Seed {seed} not found in no_pe_run_ids")
            self.no_pe_run_id = None
            return

        current_pe_type = self.model.config.position_encoding_type
        assert current_pe_type is not None

        if current_pe_type == "none":
            # We randomly sample from the no_pe_run_ids with different seeds
            # to get a more robust estimate of the KL
            other_seeds = sorted(list(self.no_pe_seed_to_run_id.keys()))
            other_seeds.remove(seed)
            if len(other_seeds) == 0:
                logger.info("Only one NoPE run found, skipping KL analysis")
                self.no_pe_run_id = None
                return

            other_seed = np.random.choice(other_seeds)
            logger.info(
                f"Using seed {other_seed} for NoPE run instead of {seed}"
            )
            self.no_pe_run_id = self.no_pe_seed_to_run_id[other_seed]
        else:
            self.no_pe_run_id = self.no_pe_seed_to_run_id[seed]

        self.attention_analysis_root_dir = Path(attention_analysis_root_dir)
        self.no_pe_analysis_root_dir = (
            self.attention_analysis_root_dir / self.no_pe_run_id
        )

        assert (
            self.no_pe_analysis_root_dir.exists()
        ), f"No analysis found for NoPE ({self.no_pe_run_id})"

    def analyze(self):
        if self.no_pe_run_id is None:
            logger.info("No NoPE run found, skipping KL analysis")
            return

        self._compute_kl()

    def _compute_kl(self):
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

        df_data: List[Dict[str, Any]] = []

        for cat in tqdm(all_categories, desc="Computing attentions KL"):
            attn_analysis_dir = (
                self.exp_root
                / "analysis"
                / "AttentionAnalyzer__test"
                / f"category__{cat}"
            )
            no_pe_attn_analysis_dir = self.no_pe_analysis_root_dir / f"{cat}"

            if (
                (
                    not (attn_analysis_dir / "dataset.jsonl").exists()
                    and not (attn_analysis_dir / "dataset.parquet").exists()
                )
                or not (attn_analysis_dir / "attention_maps_probs.pt").exists()
                or (
                    not (no_pe_attn_analysis_dir / "dataset.jsonl").exists()
                    and not (no_pe_attn_analysis_dir / "dataset.parquet").exists()
                )
            ):
                logger.warning(f"Skipping {cat} because it does not exist")
                continue

            # Load the dataset
            ds = load_saved_dataset(attn_analysis_dir)
            no_pe_ds = load_saved_dataset(no_pe_attn_analysis_dir)

            # Make sure the datasets are the same
            assert len(ds) == len(no_pe_ds)
            for idx in range(len(ds)):
                assert (
                    ds[idx] == no_pe_ds[idx]
                ), f"Datasets are not the same at index {idx}"

            # Load the huggingface attention maps
            # format:
            # *attn_map = [ sample_1_attn_map, sample_2_attn_map, ... ]
            # sample_i_attn_map = [ layer_1_attn_map, layer_2_attn_map, ... ]
            # layer_i_attn_map = tensor of shape (1, num_heads, seq_len, seq_len)
            attn_map = torch.load(attn_analysis_dir / "attention_maps_probs.pt")
            no_pe_attn_map = torch.load(
                no_pe_attn_analysis_dir / "attention_maps_probs.pt"
            )

            # Make sure the attention maps are the same
            assert len(attn_map) == len(no_pe_attn_map)

            num_layers = len(attn_map[0])
            num_heads = attn_map[0][0].shape[1]

            for sample_idx in range(len(ds)):
                for layer_idx in range(num_layers):
                    # We compute the KL divergence between every pair of heads in the same layer
                    head_combinations = list(
                        itertools.combinations(list(range(num_heads)), 2)
                    )
                    for src_head_idx, tgt_head_idx in head_combinations:
                        attn_matrix = attn_map[sample_idx][layer_idx][0, src_head_idx]
                        no_pe_attn_matrix = no_pe_attn_map[sample_idx][layer_idx][
                            0, tgt_head_idx
                        ]
                        kl = compute_distance_two_matrix(no_pe_attn_matrix, attn_matrix)
                        jsd = compute_jsd_two_matrix(no_pe_attn_matrix, attn_matrix)
                        df_data.append(
                            {
                                "category": cat,
                                "sample_idx": sample_idx,
                                "layer_idx": layer_idx,
                                "src_head_idx": src_head_idx,
                                "tgt_head_idx": tgt_head_idx,
                                "head_category": f"{src_head_idx}__{tgt_head_idx}",
                                "kl": kl,
                                "jsd": jsd,
                            }
                        )

        df = pd.DataFrame(df_data)
        csv_file = self.analysis_root / "attention_kl.csv"
        df.to_csv(csv_file, index=False)
        if self.logger is not None:
            self.logger.save(str(csv_file.absolute()))
