import random
from typing import Dict, List

import torch
from torch.utils.data import Dataset


class SortDataset(Dataset):
    """Generates deterministic sorting examples on the fly for reproducibility."""

    def __init__(
        self,
        size: int,
        vocab_size: int,
        min_len: int,
        max_len: int,
        int_token_offset: int,
        seed: int,
    ) -> None:
        if min_len < 1:
            raise ValueError("min_len must be >= 1")
        if min_len > max_len:
            raise ValueError("min_len must be <= max_len")
        self.size = size
        self.vocab_size = vocab_size
        self.min_len = min_len
        self.max_len = max_len
        self.int_token_offset = int_token_offset
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | int]:
        rng = random.Random(self.seed + idx)
        length = rng.randint(self.min_len, self.max_len)
        seq = [rng.randrange(self.vocab_size) for _ in range(length)]
        sorted_seq = sorted(seq)

        src = torch.tensor(
            [value + self.int_token_offset for value in seq], dtype=torch.long
        )
        tgt = torch.tensor(
            [value + self.int_token_offset for value in sorted_seq], dtype=torch.long
        )

        return {"src": src, "tgt": tgt, "length": length}


def _pad_sequences(seqs: List[torch.Tensor], pad_token_id: int) -> torch.Tensor:
    max_len = max(seq.size(0) for seq in seqs)
    batch = torch.full((len(seqs), max_len), pad_token_id, dtype=torch.long)
    for i, seq in enumerate(seqs):
        batch[i, : seq.size(0)] = seq
    return batch


def collate_batch(
    batch: List[Dict[str, torch.Tensor | int]],
    pad_token_id: int,
    bos_token_id: int,
) -> Dict[str, torch.Tensor]:
    src_seqs = [item["src"] for item in batch]
    tgt_seqs = [item["tgt"] for item in batch]
    lengths = torch.tensor([item["length"] for item in batch], dtype=torch.long)

    src = _pad_sequences(src_seqs, pad_token_id)
    tgt = _pad_sequences(tgt_seqs, pad_token_id)

    decoder_in = torch.full_like(tgt, pad_token_id)
    for i, seq in enumerate(tgt_seqs):
        shifted = torch.cat([torch.tensor([bos_token_id], dtype=torch.long), seq[:-1]])
        decoder_in[i, : shifted.size(0)] = shifted

    return {
        "src": src,
        "tgt": tgt,
        "decoder_in": decoder_in,
        "lengths": lengths,
    }
