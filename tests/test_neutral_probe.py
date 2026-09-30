import runpy
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NeutralProbeTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "models/ro-mmbert-v11-abstract/adapter_model.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_frozen_negative_probe_recomputes(self):
        verify = runpy.run_path(str(ROOT / "eval/verify_mmbert_neutral.py"))["verify"]
        report = verify(ROOT)
        self.assertEqual(report["requests"], 48)
        self.assertTrue(report["preregistered_success"])


if __name__ == "__main__":
    unittest.main()
