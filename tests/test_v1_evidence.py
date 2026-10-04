"""The aggregate gate must not confuse declared status with checked evidence."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roguard.v1_evidence import CELLS, audit_v1_evidence


ROOT = Path(__file__).resolve().parents[1]


class V1EvidenceTests(unittest.TestCase):
    def test_current_repository_reports_open_external_gates(self):
        report = audit_v1_evidence(ROOT)
        self.assertFalse(report["machine_abstract_gates_passed"])
        self.assertFalse(report["candidate_v1_release_gates_passed"])
        self.assertFalse(report["candidate_v1_study_pipeline_bound"])
        self.assertEqual({item["status"] for item in report["candidate_v1_taxonomy"]["taxonomy"].values()},
                         {"pending"})
        self.assertEqual({item["status"] for item in report["taxonomy"].values()},
                         {"pending_or_invalid_record"})
        self.assertEqual({item["status"] for item in report["studies"].values()},
                         {"blocked_by_taxonomy_review"})
        self.assertFalse(report["real_child_message_accuracy_established"])

    def test_manifest_rejects_path_escape_and_fake_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "studies.json"
            path.write_text(json.dumps({"schema_version": 1, "studies": {
                "ro:D1": {"packet_dir": "../outside", "reviewer_a": "a",
                          "reviewer_b": "b", "adjudicated": "c", "predictions": "d"}}}))
            with self.assertRaisesRegex(ValueError, "escapes"):
                audit_v1_evidence(root, path)
            path.write_text(json.dumps({"schema_version": 1, "studies": {
                "ro:D1": {"status": "meets_numeric_target"}}}))
            with self.assertRaisesRegex(ValueError, "file entry"):
                audit_v1_evidence(root, path)

    def test_numeric_pass_requires_all_four_matching_verified_studies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "studies.json"
            files = {name: "evidence/" + name for name in
                     ("packet_dir", "reviewer_a", "reviewer_b", "adjudicated", "predictions")}
            path.write_text(json.dumps({"schema_version": 1, "studies": {
                cell: files for cell in CELLS}}))

            def report_for_cell(*_args):
                cell = report_for_cell.cells.pop(0)
                language, category = cell.split(":")
                return {"language": language, "category": category,
                        "prediction_model_id_matches_freeze": True,
                        "annotation_packet_binding_verified": True,
                        "packet_integrity_verified": True,
                        "answer_id_alignment_verified": True,
                        "adjudication_id_alignment_verified": True,
                        "prediction_source_alignment_verified": True,
                        "abstract_numeric_target": {"status": "meets_numeric_target"},
                        "batch": "batch-9999", "pairs": {"adjudicated_valid_one_fact_pairs": 48},
                        "model_exact_valid_one_fact_pair_rate": 1.0,
                        "model_precision": 1.0, "model_recall_with_abstentions_as_misses": 1.0,
                        "model_specificity_with_abstentions_as_misses": 1.0,
                        "model_parse_coverage_all_cards": 1.0}

            report_for_cell.cells = list(CELLS)
            with patch("roguard.v1_evidence.verify_independent_reviews", return_value="digest"), \
                 patch("roguard.v1_evidence.evaluate_adjudicated_registered",
                       side_effect=report_for_cell):
                report = audit_v1_evidence(root, path)
            self.assertTrue(report["machine_abstract_gates_passed"])
            self.assertFalse(report["human_identity_and_independence_verified_by_software"])
            self.assertFalse(report["model_execution_verified_by_software"])
            self.assertFalse(report["operational_child_safety_release_authorized"])

            with patch("roguard.v1_evidence.verify_independent_reviews", return_value="digest"), \
                 patch("roguard.v1_evidence.evaluate_adjudicated_registered",
                       return_value={"language": "ro", "category": "D1"}):
                incomplete = audit_v1_evidence(root, path)
            self.assertFalse(incomplete["machine_abstract_gates_passed"])
            self.assertEqual({item["status"] for item in incomplete["studies"].values()},
                             {"invalid_study_evidence"})


if __name__ == "__main__":
    unittest.main()
