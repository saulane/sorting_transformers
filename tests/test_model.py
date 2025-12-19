import sys
from pathlib import Path
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sorting_transformers.models.variants import build_model


class ModelForwardTest(unittest.TestCase):
    def test_forward_shapes(self):
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
                "variant": "baseline_absolute",
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
        batch_size = 2
        seq_len = 6
        vocab_size = model.vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        decoder_input = torch.randint(0, vocab_size, (batch_size, seq_len - 1))
        logits = model(input_ids, decoder_input)
        self.assertEqual(logits.shape, (batch_size, seq_len - 1, vocab_size))


if __name__ == "__main__":
    unittest.main()
