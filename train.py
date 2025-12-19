import os
import random

import torch
from torch.utils.data import DataLoader

from config import Config, config_to_dict, get_config, vocab_size_total
from data import SortDataset, collate_batch
from model import TransformerSeq2Seq


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


VAL_SEED_OFFSET = 10_000  # Offset to keep train/val RNG streams disjoint.


def make_dataloader(
    dataset: SortDataset,
    batch_size: int,
    pad_token_id: int,
    bos_token_id: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator if shuffle else None,
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


def validate_config(cfg: Config) -> None:
    if cfg.model_max_seq_len < cfg.train_max_seq_len:
        raise ValueError("model_max_seq_len must be >= train_max_seq_len")
    for length in cfg.eval_extra_lengths:
        if length > cfg.model_max_seq_len:
            raise ValueError("eval_extra_lengths must be <= model_max_seq_len")


def main() -> None:
    cfg = get_config()
    validate_config(cfg)
    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else device
    total_vocab = vocab_size_total(cfg)

    train_dataset = SortDataset(
        size=cfg.train_size,
        vocab_size=cfg.vocab_size,
        min_len=cfg.train_min_seq_len,
        max_len=cfg.train_max_seq_len,
        int_token_offset=cfg.int_token_offset,
        seed=cfg.seed,
    )
    val_dataset = SortDataset(
        size=cfg.val_size,
        vocab_size=cfg.vocab_size,
        min_len=cfg.train_min_seq_len,
        max_len=cfg.train_max_seq_len,
        int_token_offset=cfg.int_token_offset,
        seed=cfg.seed + VAL_SEED_OFFSET,
    )

    train_loader = make_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        shuffle=True,
        seed=cfg.seed,
    )
    val_loader = make_dataloader(
        val_dataset,
        batch_size=cfg.batch_size,
        pad_token_id=cfg.pad_token_id,
        bos_token_id=cfg.bos_token_id,
        shuffle=False,
        seed=cfg.seed,
    )

    model = TransformerSeq2Seq(
        vocab_size=total_vocab,
        max_seq_len=cfg.model_max_seq_len,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        d_ff=cfg.d_ff,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        pad_token_id=cfg.pad_token_id,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.pad_token_id)

    train_losses = []
    val_exact_matches = []
    val_token_accs = []

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_tokens = 0

        for batch in train_loader:
            src = batch["src"].to(device)
            tgt = batch["tgt"].to(device)
            decoder_in = batch["decoder_in"].to(device)

            src_key_padding_mask = src.eq(cfg.pad_token_id)
            tgt_key_padding_mask = decoder_in.eq(cfg.pad_token_id)

            logits = model(
                src,
                decoder_in,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
            )

            loss = criterion(logits.reshape(-1, total_vocab), tgt.reshape(-1))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                token_count = tgt.ne(cfg.pad_token_id).sum().item()
                total_loss += loss.item() * token_count
                total_tokens += token_count

        train_loss = total_loss / total_tokens if total_tokens else 0.0
        train_losses.append(train_loss)

        val_exact, val_token = evaluate(
            model,
            val_loader,
            pad_token_id=cfg.pad_token_id,
            bos_token_id=cfg.bos_token_id,
            device=device,
        )
        val_exact_matches.append(val_exact)
        val_token_accs.append(val_token)

        print(
            f"Epoch {epoch:02d} | loss {train_loss:.4f} | "
            f"val exact {val_exact:.4f} | val token {val_token:.4f}"
        )

    os.makedirs(os.path.dirname(cfg.checkpoint_path), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config_to_dict(cfg),
        "metadata": {
            "epochs": cfg.epochs,
            "train_loss": train_losses,
            "val_exact_match": val_exact_matches,
            "val_token_accuracy": val_token_accs,
        },
    }
    torch.save(checkpoint, cfg.checkpoint_path)
    print(f"Saved checkpoint to {cfg.checkpoint_path}")


if __name__ == "__main__":
    main()
