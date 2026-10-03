"""Replay must reject plausible prediction edits, not just a wrong model ID."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roguard.blind_packets import build_blind_packets
from roguard.prospective_predict_bilingual import freeze_bilingual_packet_predictions
from roguard.prospective_replay_bilingual import replay_bilingual_predictions
from tests.test_independent_batch_intake import BATCH, fixture


class BilingualReplayTests(unittest.TestCase):
    def test_replays_ukrainian_decisions_and_rejects_edited_output(self):
        class Backend:
            thresholds = {"uk:D1": 0.1}

            def score_cards(self, cards):
                return [0.9 if "наявний" in card["text"] else 0.05 for card in cards]

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture(root, language="uk", category="D1")
            packets = root / "packets"
            build_blind_packets(root, BATCH, "D1", packets)
            packet = packets / "reviewer-a.jsonl"
            predictions = root / "predictions.json"
            backend = Backend()
            freeze_bilingual_packet_predictions(packet, backend, "frozen_model", predictions)
            record_path = root / "eval/prospective/freeze.json"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(json.dumps({
                "adapter_dir": "models/wrapper", "cutoff": 0.1}), encoding="utf-8")
            wrapper = root / "models/wrapper"
            wrapper.mkdir(parents=True)
            source = root / "models/source"
            with (patch("roguard.prospective_replay_bilingual.check_cell_wrapper",
                        return_value=(source, 0.1, "frozen_model")),
                  patch("roguard.prospective_replay_bilingual.verify_study_freeze",
                        return_value={"path": "eval/prospective/freeze.json",
                                      "prediction_model_id": "frozen_model",
                                      "sha256": "f" * 64})):
                report = replay_bilingual_predictions(root, packet, predictions,
                                                      wrapper, record_path, backend)
                self.assertTrue(report["all_decisions_reproduced_from_registered_adapter"])
                self.assertEqual(report["cards"], 96)
                self.assertFalse(report["real_child_language_accuracy_established"])
                payload = json.loads(predictions.read_text(encoding="utf-8"))
                first = payload["predictions"][0]
                first["predicted"] = [] if first["predicted"] else ["D1"]
                predictions.write_text(json.dumps(payload), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "do not reproduce"):
                    replay_bilingual_predictions(root, packet, predictions,
                                                wrapper, record_path, backend)


if __name__ == "__main__":
    unittest.main()
