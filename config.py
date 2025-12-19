from dataclasses import asdict, dataclass
from typing import Tuple


@dataclass
class Config:
    seed: int = 1337

    # Data
    vocab_size: int = 100
    train_min_seq_len: int = 5
    train_max_seq_len: int = 30
    model_max_seq_len: int = 100  # Must cover train and eval lengths.
    train_size: int = 100000
    val_size: int = 1000
    batch_size: int = 128

    # Tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    int_token_offset: int = 2  # Integer tokens start at this offset.

    # Model
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 256
    num_layers: int = 3
    dropout: float = 0.1

    # Training
    epochs: int = 20
    lr: float = 3e-4
    weight_decay: float = 0.0

    # Evaluation
    eval_extra_lengths: Tuple[int, ...] = ()

    # Saving
    checkpoint_path: str = "checkpoints/sort_transformer.pt"


def get_config() -> Config:
    return Config()


def config_to_dict(cfg: Config) -> dict:
    return asdict(cfg)


def config_from_dict(data: dict) -> Config:
    payload = dict(data)
    if isinstance(payload.get("eval_extra_lengths"), list):
        payload["eval_extra_lengths"] = tuple(payload["eval_extra_lengths"])
    return Config(**payload)


def vocab_size_total(cfg: Config) -> int:
    return cfg.vocab_size + cfg.int_token_offset
