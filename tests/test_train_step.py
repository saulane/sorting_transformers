import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sorting_transformers.models.variants import build_model


VARIANTS = [
    "baseline_absolute",
    "rel_bias",
    "rope",
    "alibi",
    "comparison_normalized",
    "tied_attention",
    "tiny",
    "wide_shallow",
]


class TrainStepTest(unittest.TestCase):
    def test_one_step_per_variant(self):
        for variant in VARIANTS:
            cfg = {
                "pad_token_id": 0,
                "bos_token_id": 1,
                "eos_token_id": 2,
                "device": "cpu",
                "dataset": {
                    "train_lengths": [4],
                    "test_lengths": [4],
                    "train_value_min": 0,
                    "train_value_max": 7,
                    "test_value_min": 0,
                    "test_value_max": 7,
                    "allow_duplicates": True,
                    "vocab_offset": 3,
                },
                "model": {
                    "variant": variant,
                    "d_model": 32,
                    "n_layers": 2,
                    "n_heads": 4,
                    "d_ff": 64,
                    "dropout": 0.0,
                    "positional": "absolute",
                    "tie_attention": False,
                    "comparison_normalize": False,
                },
            }
            model = build_model(cfg)
            vocab_size = model.vocab_size
            input_ids = torch.randint(0, vocab_size, (2, 6))
            decoder_input = torch.randint(0, vocab_size, (2, 5))
            logits = model(input_ids, decoder_input)
            loss = logits.mean()
            loss.backward()


if __name__ == "__main__":
    unittest.main()
