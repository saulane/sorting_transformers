import random
from dataclasses import dataclass
from typing import List, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class DatasetSpec:
    lengths: List[int]
    value_min: int
    value_max: int
    allow_duplicates: bool


class SortingDataset(Dataset):
    """Deterministic sorting dataset generated from (seed, index)."""

    def __init__(
        self,
        size: int,
        lengths: Sequence[int],
        value_min: int,
        value_max: int,
        allow_duplicates: bool,
        vocab_offset: int,
        bos_token_id: int,
        eos_token_id: int,
        seed: int,
        index_offset: int = 0,
    ) -> None:
        if size < 1:
            raise ValueError("size must be >= 1")
        if not lengths:
            raise ValueError("lengths must be non-empty")
        if value_min < 0 or value_max < value_min:
            raise ValueError("value range invalid")
        self.size = size
        self.lengths = list(lengths)
        self.value_min = value_min
        self.value_max = value_max
        self.allow_duplicates = allow_duplicates
        self.vocab_offset = vocab_offset
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.seed = seed
        self.index_offset = index_offset

    def __len__(self) -> int:
        return self.size

    def _sample_values(self, rng: random.Random, length: int) -> List[int]:
        if self.allow_duplicates:
            return [rng.randint(self.value_min, self.value_max) for _ in range(length)]
        population = list(range(self.value_min, self.value_max + 1))
        if length > len(population):
            raise ValueError(
                "length exceeds unique values in range; enable duplicates or widen range"
            )
        return rng.sample(population, length)

    def __getitem__(self, idx: int) -> dict:
        rng = random.Random(self.seed + self.index_offset + idx)
        length = rng.choice(self.lengths)
        values = self._sample_values(rng, length)
        sorted_values = sorted(values)

        input_ids = [self.bos_token_id]
        input_ids += [self.vocab_offset + value for value in values]
        input_ids.append(self.eos_token_id)

        target_ids = [self.bos_token_id]
        target_ids += [self.vocab_offset + value for value in sorted_values]
        target_ids.append(self.eos_token_id)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "target_ids": torch.tensor(target_ids, dtype=torch.long),
            "values": torch.tensor(values, dtype=torch.long),
            "sorted_values": torch.tensor(sorted_values, dtype=torch.long),
            "length": length,
        }


def build_dataset_from_config(cfg: dict, split: str, seed_offset: int = 0) -> SortingDataset:
    dataset_cfg = cfg["dataset"]
    in_dist = split in {"train", "val"}
    lengths_key = "train_lengths" if in_dist else "test_lengths"
    value_min = dataset_cfg["train_value_min"] if in_dist else dataset_cfg["test_value_min"]
    value_max = dataset_cfg["train_value_max"] if in_dist else dataset_cfg["test_value_max"]
    n_examples_key = {
        "train": "n_train_examples",
        "val": "n_val_examples",
        "test": "n_test_examples",
    }[split]
    size = int(dataset_cfg.get(n_examples_key, 0))
    if size < 1:
        raise ValueError(f"dataset.{n_examples_key} must be >= 1")

    train_size = int(dataset_cfg.get("n_train_examples", 0))
    val_size = int(dataset_cfg.get("n_val_examples", 0))
    if split == "train":
        index_offset = 0
    elif split == "val":
        index_offset = train_size
    else:
        index_offset = train_size + val_size
    return SortingDataset(
        size=size,
        lengths=dataset_cfg[lengths_key],
        value_min=value_min,
        value_max=value_max,
        allow_duplicates=dataset_cfg["allow_duplicates"],
        vocab_offset=dataset_cfg["vocab_offset"],
        bos_token_id=cfg["bos_token_id"],
        eos_token_id=cfg["eos_token_id"],
        seed=cfg["seed"] + seed_offset,
        index_offset=index_offset,
    )


def infer_vocab_size(cfg: dict) -> int:
    dataset_cfg = cfg["dataset"]
    max_value = max(dataset_cfg["train_value_max"], dataset_cfg["test_value_max"])
    return dataset_cfg["vocab_offset"] + max_value + 1
