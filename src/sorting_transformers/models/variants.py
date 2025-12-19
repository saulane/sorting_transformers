from __future__ import annotations

from typing import Any, Dict

from sorting_transformers.data.sorting_dataset import infer_vocab_size
from sorting_transformers.models.transformer_base import TransformerSeq2Seq


VARIANT_PRESETS = {
    "baseline_absolute": {"positional": "absolute"},
    "rel_bias": {"positional": "rel_bias"},
    "rope": {"positional": "rope"},
    "alibi": {"positional": "alibi"},
    "comparison_normalized": {"comparison_normalize": True},
    "tied_attention": {"tie_attention": True},
    "tiny": {"d_model": 64, "n_layers": 2, "n_heads": 2, "d_ff": 128},
    "wide_shallow": {"d_model": 256, "n_layers": 2, "n_heads": 8, "d_ff": 512},
}


def build_model(cfg: Dict[str, Any]) -> TransformerSeq2Seq:
    model_cfg = dict(cfg["model"])
    variant = model_cfg.get("variant", "baseline_absolute")
    preset = VARIANT_PRESETS.get(variant, {})

    for key, value in preset.items():
        model_cfg.setdefault(key, value)

    if variant in ("baseline_absolute", "rel_bias", "rope", "alibi"):
        model_cfg["positional"] = preset["positional"]

    if variant == "comparison_normalized":
        model_cfg["comparison_normalize"] = True
    if variant == "tied_attention":
        model_cfg["tie_attention"] = True

    lengths = cfg["dataset"]["train_lengths"] + cfg["dataset"]["test_lengths"]
    max_seq_len = model_cfg.get("max_seq_len")
    if max_seq_len is None:
        max_seq_len = max(lengths) + 2
    if max_seq_len < max(lengths) + 2:
        raise ValueError("model.max_seq_len must cover max length + BOS/EOS")

    return TransformerSeq2Seq(
        vocab_size=infer_vocab_size(cfg),
        max_seq_len=max_seq_len,
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        d_ff=model_cfg["d_ff"],
        num_layers=model_cfg["n_layers"],
        dropout=model_cfg["dropout"],
        pad_token_id=cfg["pad_token_id"],
        positional=model_cfg.get("positional", "absolute"),
        tie_attention=model_cfg.get("tie_attention", False),
        comparison_normalize=model_cfg.get("comparison_normalize", False),
    )
