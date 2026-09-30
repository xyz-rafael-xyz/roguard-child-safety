"""Blinded packets omit reference labels and keep the ID map with the owner."""

import json
from pathlib import Path
import tempfile
import unittest

from roguard.blind_packets import build_blind_packets
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]


class BlindPacketTests(unittest.TestCase):
    def test_two_orders_share_opaque_ids_without_reference_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "review"
            result = build_blind_packets(ROOT, "batch-0035", "D1", target)
            self.assertEqual(result["cards"], 48)
            left = [json.loads(line) for line in (target / "reviewer-a.jsonl").read_text().splitlines()]
            right = [json.loads(line) for line in (target / "reviewer-b.jsonl").read_text().splitlines()]
            owner = json.loads((target / "owner-map.json").read_text())
            left_ids = [item["item_id"] for item in left]
            right_ids = [item["item_id"] for item in right]
            self.assertEqual(len(set(left_ids)), 48)
            self.assertEqual(set(left_ids), set(right_ids))
            self.assertNotEqual(left_ids, right_ids)
            self.assertTrue(all(set(item) == {"item_id", "language", "category",
                                                   "source_kind", "abstract_card"}
                                for item in left + right))
            self.assertEqual({item["source_id"] for item in owner["id_map"]},
                             {row["id"] for row in load_approved(ROOT, ["batch-0035"])})
            self.assertNotIn("labels", (target / "reviewer-a.jsonl").read_text())
            self.assertEqual(owner["packet_sha256"]["reviewer-a.jsonl"],
                             sha256(target / "reviewer-a.jsonl"))
            self.assertFalse(owner["reviewer_independence_verified"])
            with self.assertRaises(FileExistsError):
                build_blind_packets(ROOT, "batch-0035", "D1", target)

    def test_invalid_category_does_not_create_output(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "review"
            with self.assertRaises(ValueError):
                build_blind_packets(ROOT, "batch-0035", "R1", target)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
