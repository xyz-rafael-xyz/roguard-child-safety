"""Audit binds two annotation streams to blinded, attested abstract cards."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from roguard.agreement_batch import compare_evidence_batches
from roguard.blind_packets import build_blind_packets
from roguard.packet_binding import binding_path
from roguard.review import sha256
from roguard.review_audit import audit_review_session

ROOT = Path(__file__).resolve().parents[1]
CARD = {"language": "ro", "disclosure": {
    "source_role": "minor", "safety_or_support_anchor": False,
    "indirect_or_repeated_support_pattern": False,
    "explicit_support_request": False,
}}


def _answers(packet: Path, output: Path) -> None:
    rows = [json.loads(line) for line in packet.read_text(encoding="utf-8").splitlines()]
    output.write_text("".join(json.dumps({"item_id": item["item_id"], "card": CARD}) + "\n"
                              for item in rows), encoding="utf-8")


class ReviewAuditTests(unittest.TestCase):
    def test_packet_bindings_checked_and_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            packets = path / "packets"
            build_blind_packets(ROOT, "batch-0035", "D1", packets)
            left, right = path / "left.jsonl", path / "right.jsonl"
            _answers(packets / "reviewer-a.jsonl", left)
            _answers(packets / "reviewer-b.jsonl", right)
            for answer, packet in ((left, packets / "reviewer-a.jsonl"),
                                   (right, packets / "reviewer-b.jsonl")):
                sidecar = binding_path(answer)
                sidecar.write_text(json.dumps({"schema_version": 1,
                                               "packet_sha256": sha256(packet)}))
                sidecar.chmod(0o600)
            self.assertTrue(audit_review_session(ROOT, packets, left, right)[
                "annotation_packet_binding_verified"])
            binding_path(left).write_text(json.dumps({"schema_version": 1,
                                                      "packet_sha256": "0" * 64}))
            with self.assertRaisesRegex(ValueError, "binding differs"):
                audit_review_session(ROOT, packets, left, right)

    def test_independent_mode_requires_both_packet_bindings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            packets = path / "packets"
            build_blind_packets(ROOT, "batch-0035", "D1", packets)
            left, right = path / "left.jsonl", path / "right.jsonl"
            _answers(packets / "reviewer-a.jsonl", left)
            _answers(packets / "reviewer-b.jsonl", right)
            with patch("roguard.review_audit.INDEPENDENT_ORIGIN",
                       "abstract_joint_factor_holdout_v1"):
                with self.assertRaisesRegex(ValueError, "requires both private packet bindings"):
                    audit_review_session(ROOT, packets, left, right)

    def test_answer_change_during_audit_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            packets = path / "packets"
            build_blind_packets(ROOT, "batch-0035", "D1", packets)
            left, right = path / "left.jsonl", path / "right.jsonl"
            _answers(packets / "reviewer-a.jsonl", left)
            _answers(packets / "reviewer-b.jsonl", right)
            def mutate_after_comparison(a, b):
                report = compare_evidence_batches(a, b)
                left.write_text(left.read_text() + "\n")
                return report
            with patch("roguard.review_audit.compare_evidence_batches",
                       side_effect=mutate_after_comparison):
                with self.assertRaisesRegex(ValueError, "changed during audit"):
                    audit_review_session(ROOT, packets, left, right)

    def test_packet_and_answers_align_without_claiming_reviewer_validity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            packets = path / "packets"
            build_blind_packets(ROOT, "batch-0035", "D1", packets)
            left, right = path / "left.jsonl", path / "right.jsonl"
            _answers(packets / "reviewer-a.jsonl", left)
            _answers(packets / "reviewer-b.jsonl", right)
            report = audit_review_session(ROOT, packets, left, right)
            self.assertEqual(report["rows"], 48)
            self.assertTrue(report["packet_integrity_verified"])
            self.assertTrue(report["source_alignment_verified"])
            self.assertEqual(report["agreement"]["per_category"]["D1"]["complete_pairs"], 48)
            self.assertFalse(report["reviewer_independence_verified"])
            self.assertFalse(report["label_validity_verified"])
            self.assertNotIn("abstract_card", json.dumps(report))

    def test_tampered_packet_rejected_even_if_owner_hash_is_rewritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            packets = path / "packets"
            build_blind_packets(ROOT, "batch-0035", "D1", packets)
            left, right = path / "left.jsonl", path / "right.jsonl"
            _answers(packets / "reviewer-a.jsonl", left)
            _answers(packets / "reviewer-b.jsonl", right)
            packet = packets / "reviewer-a.jsonl"
            rows = [json.loads(line) for line in packet.read_text().splitlines()]
            rows[0]["abstract_card"] = "Altered abstract card"
            packet.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows))
            with self.assertRaisesRegex(ValueError, "packet bytes"):
                audit_review_session(ROOT, packets, left, right)
            owner_path = packets / "owner-map.json"
            owner = json.loads(owner_path.read_text())
            owner["packet_sha256"]["reviewer-a.jsonl"] = sha256(packet)
            owner_path.write_text(json.dumps(owner))
            with self.assertRaisesRegex(ValueError, "attested abstract card"):
                audit_review_session(ROOT, packets, left, right)

    def test_answer_from_wrong_item_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            packets = path / "packets"
            build_blind_packets(ROOT, "batch-0035", "D1", packets)
            left, right = path / "left.jsonl", path / "right.jsonl"
            _answers(packets / "reviewer-a.jsonl", left)
            _answers(packets / "reviewer-b.jsonl", right)
            rows = [json.loads(line) for line in left.read_text().splitlines()]
            rows[0]["item_id"] = "unregistered"
            left.write_text("".join(json.dumps(item) + "\n" for item in rows))
            with self.assertRaisesRegex(ValueError, "Answer stream"):
                audit_review_session(ROOT, packets, left, right)

    def test_answer_order_must_match_each_distinct_blinded_packet(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            packets = path / "packets"
            build_blind_packets(ROOT, "batch-0035", "D1", packets)
            left, right = path / "left.jsonl", path / "right.jsonl"
            _answers(packets / "reviewer-a.jsonl", left)
            _answers(packets / "reviewer-b.jsonl", right)
            with self.assertRaisesRegex(ValueError, "assigned blinded packet"):
                audit_review_session(ROOT, packets, right, left)


if __name__ == "__main__":
    unittest.main()
