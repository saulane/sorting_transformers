from typing import List

import torch


def _pad_sequences(seqs: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    max_len = max(seq.size(0) for seq in seqs)
    batch = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        batch[i, : seq.size(0)] = seq
    return batch


def collate_batch(batch: List[dict], pad_token_id: int) -> dict:
    input_ids = [item["input_ids"] for item in batch]
    target_ids = [item["target_ids"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    input_ids_padded = _pad_sequences(input_ids, pad_token_id)
    target_ids_padded = _pad_sequences(target_ids, pad_token_id)

    decoder_input = target_ids_padded[:, :-1]
    decoder_target = target_ids_padded[:, 1:]

    attention_mask = input_ids_padded.ne(pad_token_id)
    decoder_attention_mask = decoder_input.ne(pad_token_id)

    return {
        "input_ids": input_ids_padded,
        "target_ids": target_ids_padded,
        "decoder_input": decoder_input,
        "decoder_target": decoder_target,
        "attention_mask": attention_mask,
        "decoder_attention_mask": decoder_attention_mask,
        "lengths": lengths,
    }
