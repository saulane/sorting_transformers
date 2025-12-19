import argparse

import torch
from torch.utils.data import DataLoader

from config import config_from_dict, get_config, vocab_size_total
from data import SortDataset, collate_batch
from model import TransformerSeq2Seq


IN_DIST_SEED_OFFSET = 20_000  # Offset to avoid overlapping sequences with training.
EXTRA_SEED_OFFSET = 30_000  # Separate RNG stream for extra-length eval.


def make_dataloader(
    dataset: SortDataset, batch_size: int, pad_token_id: int, bos_token_id: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_batch(batch, pad_token_id, bos_token_id),
    )


@torch.no_grad()
def evaluate(
    model: TransformerSeq2Seq,
    dataloader: DataLoader,
    pad_token_id: int,
    bos_token_id: int,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_tokens = 0
    correct_tokens = 0
    total_sequences = 0
    correct_sequences = 0

    for batch in dataloader:
        src = batch["src"].to(device)
        tgt = batch["tgt"].to(device)
        src_key_padding_mask = src.eq(pad_token_id)
        max_len = tgt.size(1)

        preds = model.greedy_decode(
            src,
            src_key_padding_mask=src_key_padding_mask,
            max_len=max_len,
            bos_token_id=bos_token_id,
            pad_token_id=pad_token_id,
        )

        mask = tgt.ne(pad_token_id)
        correct_tokens += (preds.eq(tgt) & mask).sum().item()
        total_tokens += mask.sum().item()

        matches = preds.eq(tgt) | ~mask
        correct_sequences += matches.all(dim=1).sum().item()
        total_sequences += tgt.size(0)

    token_acc = correct_tokens / total_tokens if total_tokens else 0.0
    exact_match = correct_sequences / total_sequences if total_sequences else 0.0
    return exact_match, token_acc


def main() -> None:
    default_cfg = get_config()
    parser = argparse.ArgumentParser(description="Evaluate a sorting transformer.")
    parser.add_argument(
        "--checkpoint",
        default=default_cfg.checkpoint_path,
        help="Path to a saved checkpoint.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = config_from_dict(checkpoint["config"])

    if cfg.model_max_seq_len < cfg.train_max_seq_len:
        raise ValueError("model_max_seq_len must be >= train_max_seq_len")
    for length in cfg.eval_extra_lengths:
        if length > cfg.model_max_seq_len:
            raise ValueError("eval_extra_lengths must be <= model_max_seq_len")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerSeq2Seq(
        vocab_size=vocab_size_total(cfg),
        max_seq_len=cfg.model_max_seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pad_token_id=cfg.pad_token_id,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    in_dist_dataset = SortDataset(
        size=cfg.val_size,
        vocab_size=cfg.vocab_size,
        min_len=cfg.train_min_seq_len,
        max_len=cfg.train_max_seq_len,
        int_token_offset=cfg.int_token_offset,
        seed=cfg.seed + IN_DIST_SEED_OFFSET,
    )
    in_dist_loader = make_dataloader(
        in_dist_dataset, cfg.batch_size, cfg.pad_token_id, cfg.bos_token_id
    )

    exact, token = evaluate(
        model,
        in_dist_loader,
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        device=device,
    )
    print(
        f"In-distribution lengths {cfg.train_min_seq_len}-{cfg.train_max_seq_len} | "
        f"exact {exact:.4f} | token {token:.4f}"
    )

    for length in cfg.eval_extra_lengths:
        extra_dataset = SortDataset(
            size=cfg.val_size,
            vocab_size=cfg.vocab_size,
            min_len=length,
            max_len=length,
            int_token_offset=cfg.int_token_offset,
            seed=cfg.seed + EXTRA_SEED_OFFSET + length,
        )
        extra_loader = make_dataloader(
            extra_dataset, cfg.batch_size, cfg.pad_token_id, cfg.bos_token_id
        )

        exact, token = evaluate(
            model,
            extra_loader,
            pad_token_id=cfg.pad_token_id,
            bos_token_id=cfg.bos_token_id,
            device=device,
        )
        print(f"Length {length} | exact {exact:.4f} | token {token:.4f}")


if __name__ == "__main__":
    main()
