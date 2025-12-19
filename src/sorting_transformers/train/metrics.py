from __future__ import annotations

from typing import Dict, List, Tuple

import torch


def _strip_special(
    seq: List[int], pad_token_id: int, bos_token_id: int, eos_token_id: int
) -> List[int]:
    return [
        token
        for token in seq
        if token not in (pad_token_id, bos_token_id, eos_token_id)
    ]


def _stable_rank_indices(values: List[int], reference: List[int]) -> List[int]:
    positions: Dict[int, List[int]] = {}
    for idx, val in enumerate(reference):
        positions.setdefault(val, []).append(idx)
    ranks = []
    for val in values:
        if val not in positions or not positions[val]:
            ranks.append(len(reference))
        else:
            ranks.append(positions[val].pop(0))
    return ranks


def _inversion_count(ranks: List[int]) -> int:
    count = 0
    for i in range(len(ranks)):
        for j in range(i + 1, len(ranks)):
            if ranks[i] > ranks[j]:
                count += 1
    return count


def kendall_tau(pred_values: List[int], target_values: List[int]) -> float:
    n = len(target_values)
    if n < 2:
        return 1.0
    ranks = _stable_rank_indices(pred_values, target_values)
    inversions = _inversion_count(ranks)
    total_pairs = n * (n - 1) / 2
    return 1.0 - (2.0 * inversions / total_pairs)


def prefix_sorted_ratio(pred_values: List[int]) -> float:
    if not pred_values:
        return 0.0
    length = 1
    for idx in range(1, len(pred_values)):
        if pred_values[idx - 1] <= pred_values[idx]:
            length += 1
        else:
            break
    return length / len(pred_values)


def unique_ratio(values: List[int]) -> float:
    if not values:
        return 0.0
    return len(set(values)) / len(values)


def inversion_ratio(pred_values: List[int], target_values: List[int]) -> float:
    n = len(target_values)
    if n < 2:
        return 0.0
    ranks = _stable_rank_indices(pred_values, target_values)
    inversions = _inversion_count(ranks)
    total_pairs = n * (n - 1) / 2
    return inversions / total_pairs


def classify_failure(
    pred_values: List[int],
    target_values: List[int],
    exact_match: bool,
) -> str:
    if exact_match:
        return "correct"
    if unique_ratio(pred_values) < 0.5:
        return "collapse"
    if inversion_ratio(pred_values, target_values) < 0.1:
        return "local_swaps"
    if prefix_sorted_ratio(pred_values) >= 0.5:
        return "prefix_sorted"
    return "other"


def compute_batch_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
    *,
    pad_token_id: int,
    bos_token_id: int,
    eos_token_id: int,
) -> Dict[str, float]:
    mask = targets.ne(pad_token_id)
    correct_tokens = (preds.eq(targets) & mask).sum().item()
    total_tokens = mask.sum().item()
    token_acc = correct_tokens / total_tokens if total_tokens else 0.0

    matches = preds.eq(targets) | ~mask
    exact_match = matches.all(dim=1).float().mean().item()

    tau_scores: List[float] = []
    prefix_scores: List[float] = []
    inversion_scores: List[float] = []
    labels = {
        "correct": 0,
        "collapse": 0,
        "local_swaps": 0,
        "prefix_sorted": 0,
        "other": 0,
    }

    preds_list = preds.detach().cpu().tolist()
    targets_list = targets.detach().cpu().tolist()

    for pred_seq, target_seq in zip(preds_list, targets_list, strict=True):
        pred_values = _strip_special(pred_seq, pad_token_id, bos_token_id, eos_token_id)
        target_values = _strip_special(target_seq, pad_token_id, bos_token_id, eos_token_id)
        if len(target_values) == 0:
            continue
        tau_scores.append(kendall_tau(pred_values, target_values))
        prefix_scores.append(prefix_sorted_ratio(pred_values))
        inversion_scores.append(inversion_ratio(pred_values, target_values))
        label = classify_failure(
            pred_values,
            target_values,
            pred_values == target_values,
        )
        labels[label] += 1

    mean_tau = sum(tau_scores) / len(tau_scores) if tau_scores else 0.0
    mean_prefix = sum(prefix_scores) / len(prefix_scores) if prefix_scores else 0.0
    mean_inversion = sum(inversion_scores) / len(inversion_scores) if inversion_scores else 0.0

    return {
        "token_accuracy": token_acc,
        "exact_match": exact_match,
        "kendall_tau": mean_tau,
        "prefix_sorted_ratio": mean_prefix,
        "inversion_ratio": mean_inversion,
        "count_correct": labels["correct"],
        "count_collapse": labels["collapse"],
        "count_local_swaps": labels["local_swaps"],
        "count_prefix_sorted": labels["prefix_sorted"],
        "count_other": labels["other"],
    }


def aggregate_metrics(metrics: List[Dict[str, float]]) -> Dict[str, float]:
    if not metrics:
        return {}
    totals = {key: 0.0 for key in metrics[0]}
    for item in metrics:
        for key, value in item.items():
            totals[key] += value
    aggregated = {}
    for key, value in totals.items():
        if key.startswith("count_"):
            aggregated[key] = value
        else:
            aggregated[key] = value / len(metrics)
    return aggregated
