import argparse
import random
import sys
import warnings

warnings.filterwarnings("ignore", message="Failed to initialize NumPy:*")

import torch

from config import config_from_dict, get_config, vocab_size_total
from model import TransformerSeq2Seq


RED = "\033[31m"
RESET = "\033[0m"


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise ValueError(f"Unknown device: {device_arg}")


def _token_to_display(token_id: int, int_token_offset: int, pad_token_id: int, bos_token_id: int) -> str:
    if token_id == pad_token_id:
        return "PAD"
    if token_id == bos_token_id:
        return "BOS"
    if token_id >= int_token_offset:
        return str(token_id - int_token_offset)
    return f"T{token_id}"


def _format_predicted_sequence(
    pred_token_ids: list[int],
    expected_token_ids: list[int],
    *,
    use_color: bool,
    int_token_offset: int,
    pad_token_id: int,
    bos_token_id: int,
) -> str:
    parts: list[str] = []
    for pred_id, exp_id in zip(pred_token_ids, expected_token_ids, strict=True):
        text = _token_to_display(pred_id, int_token_offset, pad_token_id, bos_token_id)
        if pred_id != exp_id and use_color:
            text = f"{RED}{text}{RESET}"
        parts.append(text)
    return "[" + " ".join(parts) + "]"


@torch.no_grad()
def main() -> None:
    default_cfg = get_config()
    parser = argparse.ArgumentParser(
        description=(
            "Load a saved sorting transformer checkpoint, generate random sequences, "
            "and print greedy-decoded predictions with mistakes highlighted."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=default_cfg.checkpoint_path,
        help="Path to a saved checkpoint.",
    )
    parser.add_argument(
        "-n",
        "--num-seqs",
        type=int,
        default=10,
        help="Number of random sequences to generate.",
    )
    parser.add_argument(
        "-k",
        "--seq-len",
        type=int,
        default=None,
        help="Sequence length (defaults to checkpoint train_max_seq_len).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed for generated sequences.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="Device to run on.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors even when stdout is a TTY.",
    )
    parser.add_argument(
        "--hide-target",
        action="store_true",
        help="Do not print the expected sorted sequence.",
    )
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    cfg = config_from_dict(checkpoint["config"])

    seq_len = args.seq_len if args.seq_len is not None else cfg.train_max_seq_len
    if seq_len < 1:
        raise ValueError("--seq-len must be >= 1")
    if seq_len > cfg.model_max_seq_len:
        raise ValueError(
            f"--seq-len ({seq_len}) exceeds model_max_seq_len ({cfg.model_max_seq_len})"
        )
    if args.num_seqs < 1:
        raise ValueError("--num-seqs must be >= 1")

    device = _resolve_device(args.device)
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
    model.eval()

    rng = random.Random(args.seed)
    sequences: list[list[int]] = [
        [rng.randrange(cfg.vocab_size) for _ in range(seq_len)]
        for _ in range(args.num_seqs)
    ]
    targets: list[list[int]] = [sorted(seq) for seq in sequences]

    src = torch.tensor(
        [[value + cfg.int_token_offset for value in seq] for seq in sequences],
        dtype=torch.long,
        device=device,
    )
    tgt = torch.tensor(
        [[value + cfg.int_token_offset for value in seq] for seq in targets],
        dtype=torch.long,
        device=device,
    )

    src_key_padding_mask = src.eq(cfg.pad_token_id)
    preds = model.greedy_decode(
        src,
        src_key_padding_mask=src_key_padding_mask,
        max_len=seq_len,
        bos_token_id=cfg.bos_token_id,
        pad_token_id=cfg.pad_token_id,
    )

    correct = preds.eq(tgt)
    mean_token_accuracy = correct.float().mean().item()
    exact_match = correct.all(dim=1).float().mean().item()

    use_color = (not args.no_color) and sys.stdout.isatty()
    preds_list = preds.detach().to("cpu").tolist()
    tgt_list = tgt.detach().to("cpu").tolist()

    for i, (seq, target, pred_token_ids, expected_token_ids) in enumerate(
        zip(sequences, targets, preds_list, tgt_list, strict=True),
        start=1,
    ):
        per_seq_acc = sum(p == e for p, e in zip(pred_token_ids, expected_token_ids, strict=True)) / seq_len
        print(f"{i:03d} input : [{ ' '.join(map(str, seq)) }]")
        if not args.hide_target:
            print(f"{i:03d} target: [{ ' '.join(map(str, target)) }]")
        print(
            f"{i:03d} pred  : "
            + _format_predicted_sequence(
                pred_token_ids,
                expected_token_ids,
                use_color=use_color,
                int_token_offset=cfg.int_token_offset,
                pad_token_id=cfg.pad_token_id,
                bos_token_id=cfg.bos_token_id,
            )
            + f"  acc={per_seq_acc:.3f}"
        )
        print()

    print(f"Mean token accuracy: {mean_token_accuracy:.4f}")
    print(f"Exact match rate:   {exact_match:.4f}")


if __name__ == "__main__":
    main()
