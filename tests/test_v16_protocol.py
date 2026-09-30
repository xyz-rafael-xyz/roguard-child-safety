import unittest
import runpy
from pathlib import Path

from roguard.review import ReviewError, load_train_dev

ROOT = Path(__file__).resolve().parents[1]


class AdvisoryRepairGateTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "models/ro-mmbert-v16-abstract/adapter_model.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_development_selected_adapter_matches_frozen_record(self):
        verify = runpy.run_path(str(ROOT / "eval/verify_committed_v16_selection.py"))["verify"]
        result = verify(ROOT)
        self.assertEqual(result["selected_epoch"], 1)
        self.assertEqual(result["development_exact_pairs"], 24)

    def test_narrow_task_requires_declared_category_coverage(self):
        train, dev = load_train_dev(
            ROOT, ["batch-0028", "batch-0029"], "ro", required_categories=("D1", "S1"))
        self.assertEqual((len(train), len(dev)), (384, 48))
        with self.assertRaisesRegex(ReviewError, "lacks positive ro/R1"):
            load_train_dev(ROOT, ["batch-0028", "batch-0029"], "ro")

    def test_sealed_test_cannot_enter_narrow_training(self):
        with self.assertRaisesRegex(ReviewError, "held-out test"):
            load_train_dev(ROOT, ["batch-0028", "batch-0029", "batch-0030"], "ro",
                           required_categories=("D1", "S1"))


if __name__ == "__main__":
    unittest.main()
