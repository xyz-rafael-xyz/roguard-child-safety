"""Bilingual blind-packet prediction and cell-wrapper binding tests."""

import json
import tempfile
import unittest
from pathlib import Path

from roguard.blind_packets import build_blind_packets
from roguard.human_eval import _read_blind_predictions
from roguard.prospective_predict_bilingual import (
    check_cell_wrapper, freeze_bilingual_packet_predictions)
from roguard.review import sha256
from tests.test_independent_batch_intake import BATCH, fixture


class BilingualPredictionTests(unittest.TestCase):
    def test_ukrainian_packet_scores_without_owner_labels(self):
        class Backend:
            thresholds = {"uk:S1": 0.5}

            def score_cards(self, cards):
                self.cards = cards
                return [0.9 if "наявний" in card["text"] else 0.1 for card in cards]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, language="uk", category="S1")
            packets = root / "packets"
            build_blind_packets(root, BATCH, "S1", packets)
            packet = packets / "reviewer-a.jsonl"
            output = root / "predictions.json"
            backend = Backend()
            result = freeze_bilingual_packet_predictions(packet, backend,
                                                         "fixture_only", output)
            self.assertEqual(len(backend.cards), 96)
            self.assertEqual(set(backend.cards[0]), {"language", "source_kind", "text"})
            self.assertFalse(result["source_batch_or_author_labels_read"])
            payload = json.loads(output.read_text())
            self.assertEqual(payload["language"], "uk")
            self.assertEqual(payload["category"], "S1")
            self.assertNotIn('"text"', output.read_text())
            self.assertNotIn('"labels"', output.read_text())
            owner = json.loads((packets / "owner-map.json").read_text())
            _, aligned = _read_blind_predictions(
                output, packet, sha256(packet), "uk", "S1",
                {item["item_id"]: item["source_id"] for item in owner["id_map"]})
            self.assertEqual(sum(aligned.values()), 48)
            with self.assertRaises(FileExistsError):
                freeze_bilingual_packet_predictions(packet, backend,
                                                     "fixture_only", output)

    def test_cell_wrapper_matches_original_weight_and_cutoff(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "models/bi-mmbert-v25-abstract"
            wrapper = root / "models/v25-uk-s1"
            original.mkdir(parents=True)
            wrapper.mkdir(parents=True)
            for directory in (original, wrapper):
                (directory / "adapter_model.safetensors").write_bytes(b"fixture weight")
                (directory / "adapter_config.json").write_bytes(b"fixture config")
            (original / "research.json").write_text(json.dumps({
                "weight_sha256": sha256(original / "adapter_model.safetensors"),
                "adapter_config_sha256": sha256(original / "adapter_config.json"),
                "thresholds": {"uk:S1": 0.5}}))
            record = {"schema_version": 1,
                      "study": "v25_independent_abstract_comparator",
                      "status": "source_artifact_frozen_for_independent_study",
                      "input_scope": "independently_authored_abstract_cards_only",
                      "eligible_for_live_child_message_screening": False,
                      "trained_language": "uk", "category": "S1",
                      "source_adapter_dir": "models/bi-mmbert-v25-abstract",
                      "shared_threshold": 0.5,
                      "adapter_weight_sha256": sha256(wrapper / "adapter_model.safetensors"),
                      "adapter_config_sha256": sha256(wrapper / "adapter_config.json"),
                      "source_manifest_sha256": sha256(original / "research.json")}
            (wrapper / "research.json").write_text(json.dumps(record))
            source, cutoff, model_id = check_cell_wrapper(root, wrapper, "uk", "S1")
            self.assertEqual((source, cutoff), (original.resolve(), 0.5))
            self.assertEqual(model_id, "mmbert_" + record["adapter_weight_sha256"])
            record["shared_threshold"] = 0.4
            (wrapper / "research.json").write_text(json.dumps(record))
            with self.assertRaisesRegex(ValueError, "differs"):
                check_cell_wrapper(root, wrapper, "uk", "S1")


if __name__ == "__main__":
    unittest.main()
