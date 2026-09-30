import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from roguard.review import (AUTO_VALIDATOR, CATEGORIES, VALIDATED_GENERATORS, ReviewError,
                            candidate_manifest, load_approved, sha256,
                            taxonomy_sha256, verify_independent_reviews, verify_taxonomy)


class ReviewGateTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "taxonomy").mkdir()
        (self.root / "data" / "synthetic").mkdir(parents=True)
        for name in ("taxonomy_ro.md", "taxonomy_uk.md"):
            (self.root / "taxonomy" / name).write_text("Draft taxonomy\n", encoding="utf-8")
        (self.root / "taxonomy" / "review_ro.json").write_text(json.dumps({
            "status": "approved", "reviewer": "fixture reviewer", "reviewed_at": "2026-09-29",
            "taxonomy_sha256": taxonomy_sha256(self.root, "ro"),
        }), encoding="utf-8")
        self.independent_path = self.root / "taxonomy" / "review_uk.json"
        self.independent_path.write_text(json.dumps({
            "status": "pending", "taxonomy_sha256": taxonomy_sha256(self.root, "uk"),
            "reviews": [],
        }), encoding="utf-8")
        self.batch = self.root / "data" / "synthetic" / "batch-0001.jsonl"
        self.batch.write_text(json.dumps({
            "id": "batch-0001-ro-01", "language": "ro", "source_kind": "message",
            "split": "train", "text": "Abstract category card A", "labels": ["D1"],
            "origin": "abstract_template_v3",
        }) + "\n", encoding="utf-8")
        self.batch.with_suffix(".preview.md").write_text(
            "| batch-0001-ro-01 | ro | message | D1 | Abstract category card A |\n", encoding="utf-8")
        self.record = self.batch.with_suffix(".review.json")
        self.record.write_text(json.dumps(candidate_manifest(self.root, "batch-0001")), encoding="utf-8")

    def approve(self):
        record = json.loads(self.record.read_text(encoding="utf-8"))
        record.update(status="approved", reviewer="fixture reviewer", reviewed_at="2026-09-29",
                      reviewed_ids=["batch-0001-ro-01"])
        self.record.write_text(json.dumps(record), encoding="utf-8")

    def test_pending_batch_cannot_enter_training(self):
        with self.assertRaisesRegex(ReviewError, "Pending batch validation"):
            load_approved(self.root, ["batch-0001"])

    def test_reproducible_batch_attestation_binds_generator_and_audit(self):
        generator = self.root / "training" / "test_generator.py"
        generator.parent.mkdir()
        source_row = json.loads(self.batch.read_text(encoding="utf-8"))
        generator.write_text(f"def build_rows():\n    return [{source_row!r}]\n", encoding="utf-8")
        audit = self.batch.with_suffix(".audit.json")
        audit.write_text('{"rows": 1}\n', encoding="utf-8")
        record = json.loads(self.record.read_text(encoding="utf-8"))
        record.update(status="validated", validator=AUTO_VALIDATOR,
                      validated_at="2026-09-29", validated_ids=["batch-0001-ro-01"],
                      generator_path="training/test_generator.py",
                      generator_sha256=sha256(generator), audit_sha256=sha256(audit))
        self.record.write_text(json.dumps(record), encoding="utf-8")
        with patch.dict(VALIDATED_GENERATORS, {"batch-0001": "training/test_generator.py"}):
            self.assertEqual(len(load_approved(self.root, ["batch-0001"])), 1)
            generator.write_text("# changed generator\n", encoding="utf-8")
            with self.assertRaisesRegex(ReviewError, "inputs changed"):
                load_approved(self.root, ["batch-0001"])

    def test_exact_bytes_and_taxonomy_are_bound(self):
        self.approve()
        self.assertEqual(len(load_approved(self.root, ["batch-0001"])), 1)
        self.batch.write_text(self.batch.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "digest differs"):
            load_approved(self.root, ["batch-0001"])

    def test_taxonomy_edit_invalidates_approval(self):
        self.approve()
        (self.root / "taxonomy" / "taxonomy_ro.md").write_text("Changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "digest differs"):
            load_approved(self.root, ["batch-0001"])

    def test_batch_must_be_explicit(self):
        self.approve()
        with self.assertRaises(ReviewError):
            load_approved(self.root, [])

    def test_review_must_enumerate_every_row(self):
        self.approve()
        record = json.loads(self.record.read_text(encoding="utf-8"))
        record["reviewed_ids"] = []
        self.record.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "enumerate every row"):
            load_approved(self.root, ["batch-0001"])

    def test_reviewed_preview_cannot_change(self):
        self.approve()
        self.batch.with_suffix(".preview.md").write_text("Changed preview\n", encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "preview differs"):
            load_approved(self.root, ["batch-0001"])

    def test_preview_cannot_hide_an_approved_row(self):
        self.batch.with_suffix(".preview.md").write_text(
            "| batch-0001-ro-01 | ro | message | none | Abstract category card A |\n", encoding="utf-8")
        record = json.loads(self.record.read_text(encoding="utf-8"))
        from roguard.review import sha256
        record.update(status="approved", reviewer="fixture reviewer", reviewed_at="2026-09-29",
                      reviewed_ids=["batch-0001-ro-01"],
                      preview_sha256=sha256(self.batch.with_suffix(".preview.md")))
        self.record.write_text(json.dumps(record), encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "misstates row"):
            load_approved(self.root, ["batch-0001"])

    def test_ukrainian_taxonomy_is_deferred(self):
        with self.assertRaisesRegex(ReviewError, "Pending independent"):
            verify_taxonomy(self.root, "uk")

    def approved_uk_reviews(self):
        return {
            "status": "approved", "taxonomy_sha256": taxonomy_sha256(self.root, "uk"),
            "reviews": [{
                "kind": kind, "reviewer": reviewer, "reviewed_at": "2026-09-29",
                "independent_of_author": True, "summary": "Reviewed every category in fixture.",
                "category_decisions": {code: "accept" for code in CATEGORIES},
            } for kind, reviewer in (("language", "fictional language reviewer"),
                                      ("child_safety", "fictional safety reviewer"))],
        }

    def write_uk_reviews(self, record):
        self.independent_path.write_text(json.dumps(record), encoding="utf-8")

    def test_two_complete_independent_reviews_open_uk_training_gate(self):
        self.write_uk_reviews(self.approved_uk_reviews())
        self.assertEqual(verify_taxonomy(self.root, "uk"), taxonomy_sha256(self.root, "uk"))

    def test_independent_review_rejects_revision_request(self):
        record = self.approved_uk_reviews()
        record["reviews"][1]["category_decisions"]["R1"] = "revise"
        self.write_uk_reviews(record)
        with self.assertRaisesRegex(ReviewError, "incomplete"):
            verify_taxonomy(self.root, "uk")

    def test_independent_review_rejects_same_reviewer(self):
        record = self.approved_uk_reviews()
        record["reviews"][1]["reviewer"] = "FICTIONAL LANGUAGE REVIEWER"
        self.write_uk_reviews(record)
        with self.assertRaisesRegex(ReviewError, "different reviewers"):
            verify_taxonomy(self.root, "uk")

    def test_independent_review_rejects_changed_taxonomy(self):
        self.write_uk_reviews(self.approved_uk_reviews())
        (self.root / "taxonomy" / "taxonomy_uk.md").write_text("Changed\n", encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "digest differs"):
            verify_taxonomy(self.root, "uk")

    def test_independent_review_rejects_missing_discipline_and_invalid_date(self):
        record = self.approved_uk_reviews()
        record["reviews"][1]["kind"] = "language"
        self.write_uk_reviews(record)
        with self.assertRaisesRegex(ReviewError, "both distinct disciplines"):
            verify_taxonomy(self.root, "uk")
        record = self.approved_uk_reviews()
        record["reviews"][1]["reviewed_at"] = "yesterday"
        self.write_uk_reviews(record)
        with self.assertRaisesRegex(ReviewError, "invalid date"):
            verify_taxonomy(self.root, "uk")

    def test_independent_review_rejects_malformed_record(self):
        self.write_uk_reviews(["not an object"])
        with self.assertRaisesRegex(ReviewError, "Pending independent"):
            verify_taxonomy(self.root, "uk")
        record = self.approved_uk_reviews()
        record["reviews"][0]["kind"] = []
        self.write_uk_reviews(record)
        with self.assertRaisesRegex(ReviewError, "both distinct disciplines"):
            verify_taxonomy(self.root, "uk")

    def test_independent_ro_release_review_is_still_pending(self):
        path = self.root / "taxonomy" / "review_ro_independent.json"
        path.write_text(json.dumps({
            "status": "pending", "taxonomy_sha256": taxonomy_sha256(self.root, "ro"),
            "reviews": [],
        }), encoding="utf-8")
        with self.assertRaisesRegex(ReviewError, "Pending independent"):
            verify_independent_reviews(self.root, "ro")


if __name__ == "__main__":
    unittest.main()
