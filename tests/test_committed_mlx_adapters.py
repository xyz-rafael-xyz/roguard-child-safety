import runpy
import unittest
from pathlib import Path

from roguard.model import MLXClassifier

ROOT = Path(__file__).resolve().parents[1]


class CommittedMLXAdaptersTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "models/ro-v4-abstract/adapters.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_four_selected_failures_match_frozen_records(self):
        verify = runpy.run_path(str(ROOT / "eval/verify_committed_mlx_adapters.py"))["verify"]
        self.assertEqual(set(verify(ROOT)), {"ro-v4", "ro-v6", "ro-qwen-v7", "ro-qwen-v8"})

    @unittest.skipUnless((ROOT / "models/ro-v4-abstract/adapters.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_portable_adapter_needs_an_explicit_local_base(self):
        with self.assertRaisesRegex(ValueError, "local pinned base_model_path"):
            MLXClassifier(ROOT / "models/ro-v4-abstract")

    @unittest.skipUnless((ROOT / "models/ro-v1-abstract/adapters.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_three_historical_pilots_match_frozen_records(self):
        verify = runpy.run_path(str(ROOT / "eval/verify_committed_pilot_adapters.py"))["verify"]
        self.assertEqual(set(verify(ROOT)), {"ro-v1", "ro-v2", "ro-v3"})
        with self.assertRaisesRegex(ValueError, "local pinned base_model_path"):
            MLXClassifier(ROOT / "models/ro-v1-abstract")


if __name__ == "__main__":
    unittest.main()
