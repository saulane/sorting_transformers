from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from sorting_transformers.data.collate import collate_batch
from sorting_transformers.data.sorting_dataset import SortingDataset
from sorting_transformers.train.metrics import compute_batch_metrics, aggregate_metrics


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    *,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    metrics: List[Dict[str, float]] = []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        decoder_target = batch["decoder_target"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        src_key_padding_mask = ~attention_mask
        max_len = decoder_target.size(1)

        preds = model.greedy_decode(
            input_ids,
            src_key_padding_mask=src_key_padding_mask,
            max_len=max_len,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
        )

        metrics.append(
            compute_batch_metrics(
                preds,
                decoder_target,
                pad_token_id=pad_token_id,
                bos_token_id=bos_token_id,
                eos_token_id=eos_token_id,
            )
        )

    return aggregate_metrics(metrics)


def evaluate_lengths(
    model: torch.nn.Module,
    lengths: List[int],
    *,
    base_dataset: SortingDataset,
    batch_size: int,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
    device: torch.device,
) -> Dict[int, Dict[str, float]]:
    results: Dict[int, Dict[str, float]] = {}
    for length in lengths:
        dataset = SortingDataset(
            size=base_dataset.size,
            lengths=[length],
            value_min=base_dataset.value_min,
            value_max=base_dataset.value_max,
            allow_duplicates=base_dataset.allow_duplicates,
            vocab_offset=base_dataset.vocab_offset,
            bos_token_id=base_dataset.bos_token_id,
            eos_token_id=base_dataset.eos_token_id,
            seed=base_dataset.seed + length,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, pad_token_id),
        )
        results[length] = evaluate_model(
            model,
            loader,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            device=device,
        )
    return results
