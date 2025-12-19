from __future__ import annotations

from typing import List, Tuple

import torch


def values_to_tokens(values: List[int], cfg: dict) -> torch.Tensor:
    tokens = [cfg["bos_token_id"]]
    tokens += [cfg["dataset"]["vocab_offset"] + value for value in values]
    tokens.append(cfg["eos_token_id"])
    return torch.tensor(tokens, dtype=torch.long)


def batch_from_values(values_batch: List[List[int]], cfg: dict, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = [values_to_tokens(values, cfg) for values in values_batch]
    max_len = max(seq.size(0) for seq in input_ids)
    pad_token_id = cfg["pad_token_id"]
    batch = torch.full((len(input_ids), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(input_ids):
        batch[i, : seq.size(0)] = seq
    attention_mask = batch.ne(pad_token_id)
    return batch.to(device), attention_mask.to(device)


def sorted_tokens(values: List[int]) -> List[int]:
    return sorted(values)
