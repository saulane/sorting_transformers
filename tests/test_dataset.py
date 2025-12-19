import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from sorting_transformers.data.sorting_dataset import SortingDataset


class SortingDatasetTest(unittest.TestCase):
    def test_targets_are_sorted(self):
        dataset = SortingDataset(
            size=5,
            lengths=[6],
            value_min=0,
            value_max=9,
            allow_duplicates=True,
            vocab_offset=3,
            bos_token_id=1,
            eos_token_id=2,
            seed=123,
        )
        sample = dataset[0]
        values = sample["values"].tolist()
        target_tokens = sample["target_ids"].tolist()
        target_values = [token - 3 for token in target_tokens[1:-1]]
        self.assertEqual(sorted(values), target_values)


if __name__ == "__main__":
    unittest.main()
