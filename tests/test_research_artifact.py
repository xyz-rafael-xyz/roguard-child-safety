import json
import unittest
from pathlib import Path

from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "models/ro-mmbert-v11-abstract"


class ResearchArtifactTests(unittest.TestCase):
    @unittest.skipUnless((ADAPTER / "adapter_model.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_private_research_adapter_matches_frozen_selection(self):
        manifest = json.loads((ADAPTER / "research.json").read_text(encoding="utf-8"))
        choice = json.loads((ROOT / "eval/runs/ro-mmbert-v11-dev-selection.json").read_text(encoding="utf-8"))
        config = json.loads((ADAPTER / "adapter_config.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["adapter_weight_sha256"],
                         sha256(ADAPTER / "adapter_model.safetensors"))
        self.assertEqual(manifest["adapter_weight_sha256"], choice["selected_weight_sha256"])
        self.assertEqual(manifest["adapter_config_sha256"], sha256(ADAPTER / "adapter_config.json"))
        self.assertEqual(manifest["v11_selection_sha256"],
                         sha256(ROOT / "eval/runs/ro-mmbert-v11-dev-selection.json"))
        self.assertEqual(manifest["prompt_sha256"], sha256(ROOT / "src/roguard/prompt_v4.py"))
        self.assertEqual(manifest["shared_threshold"], choice["selected_threshold"])
        self.assertEqual(manifest["base_revision"], choice["revision"])
        self.assertEqual(config["base_model_name_or_path"], choice["model"])
        self.assertEqual(manifest["evaluation_status"], "failed_registered_synthetic_target")


if __name__ == "__main__":
    unittest.main()
