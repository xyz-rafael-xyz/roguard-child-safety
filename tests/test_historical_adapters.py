import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class HistoricalAdaptersTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "models/ro-mmbert-v9a-abstract/adapter_model.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_selected_artifacts_match_frozen_records(self):
        verify = runpy.run_path(str(ROOT / "eval/verify_committed_historical_adapters.py"))["verify"]
        self.assertEqual(set(verify(ROOT)), {"v9a", "v10"})


if __name__ == "__main__":
    unittest.main()
