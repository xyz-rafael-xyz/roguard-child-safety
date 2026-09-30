"""Check that every committed mmBERT adapter includes its fitted output head."""

from pathlib import Path
import json
import struct
import tempfile
import unittest

from roguard.mmbert_study import verify_adapter_head

ROOT = Path(__file__).resolve().parents[1]


class AdapterHeadTests(unittest.TestCase):
    @unittest.skipUnless((ROOT / "models/ro-mmbert-v11-abstract/adapter_model.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_committed_encoder_adapters_include_fitted_heads(self):
        for name in ("v9a", "v10", "v11", "v16", "v18-factor", "v22", "v23"):
            with self.subTest(name=name):
                path = ROOT / "models" / f"ro-mmbert-{name}-abstract"
                verify_adapter_head(path)

    @unittest.skipUnless((ROOT / "models/ro-mmbert-v19-joint-abstract/adapter_model.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_joint_four_output_adapter_has_fitted_head(self):
        path = ROOT / "models/ro-mmbert-v19-joint-abstract"
        verify_adapter_head(path, num_labels=4)

    @unittest.skipUnless((ROOT / "models/ro-mmbert-v20-balanced-joint-abstract/adapter_model.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_balanced_joint_adapter_has_fitted_head(self):
        path = ROOT / "models/ro-mmbert-v20-balanced-joint-abstract"
        verify_adapter_head(path, num_labels=4)

    @unittest.skipUnless((ROOT / "models/ro-nli-v21-abstract/adapter_model.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_nli_adapter_has_three_class_head_and_pooler(self):
        path = ROOT / "models/ro-nli-v21-abstract"
        verify_adapter_head(path, num_labels=3, require_pooler=True)

    def test_missing_or_truncated_head_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            header = {"unrelated.tensor": {"dtype": "F32", "shape": [2],
                                           "data_offsets": [0, 8]}}
            encoded = json.dumps(header).encode("utf-8")
            (path / "adapter_model.safetensors").write_bytes(
                struct.pack("<Q", len(encoded)) + encoded + bytes(8))
            with self.assertRaisesRegex(ValueError, "fitted classifier"):
                verify_adapter_head(path)
            (path / "adapter_model.safetensors").write_bytes(b"short")
            with self.assertRaisesRegex(ValueError, "header"):
                verify_adapter_head(path)


if __name__ == "__main__":
    unittest.main()
