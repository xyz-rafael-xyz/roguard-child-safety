"""Exact-byte offline review handoff; no human review is simulated."""

import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from roguard.review import taxonomy_sha256
from roguard.review_bundle import create_review_bundle


class ReviewBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repo"
        (self.root / "taxonomy").mkdir(parents=True)
        (self.root / "docs").mkdir()
        (self.root / "taxonomy/taxonomy_ro.md").write_text("Definiție românească.\n", encoding="utf-8")
        digest = taxonomy_sha256(self.root, "ro")
        (self.root / "taxonomy/review_ro_independent.json").write_text(json.dumps({
            "status": "pending", "taxonomy_sha256": digest, "reviews": []}))
        (self.root / "docs/RO_REVIEW_PACKET.md").write_text(f"Review {digest}\n")
        (self.root / "docs/ROLE_TERMINOLOGY_AUDIT.md").write_text("Role note\n")
        (self.root / "docs/REVIEW_GATE.md").write_text("Record instructions\n")
        self.output = Path(self.temp.name) / "review.zip"

    def test_exact_taxonomy_and_manifest_without_cards(self):
        result = create_review_bundle(self.root, "ro", self.output)
        with zipfile.ZipFile(self.output) as archive:
            self.assertEqual(set(archive.namelist()), set(result["files"]))
            self.assertIn("RO_REVIEW_PACKET.md", archive.read("README.md").decode("utf-8"))
            taxonomy = archive.read("taxonomy/taxonomy_ro.md")
            manifest = json.loads(archive.read("manifest.json"))
            self.assertEqual(taxonomy, (self.root / "taxonomy/taxonomy_ro.md").read_bytes())
            self.assertEqual(manifest["taxonomy_sha256"], taxonomy_sha256(self.root, "ro"))
            for name, expected in manifest["files_sha256"].items():
                self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), expected)
            self.assertFalse(manifest["model_scores_or_labeled_cards_included"])
        with self.assertRaises(FileExistsError):
            create_review_bundle(self.root, "ro", self.output)

    def test_taxonomy_drift_blocks_export(self):
        (self.root / "taxonomy/taxonomy_ro.md").write_text("Changed taxonomy\n")
        with self.assertRaisesRegex(ValueError, "differs"):
            create_review_bundle(self.root, "ro", self.output)
        self.assertFalse(self.output.exists())

    def test_unreviewed_bundle_stays_outside_tracked_repo_paths(self):
        with self.assertRaisesRegex(ValueError, "outside Git"):
            create_review_bundle(self.root, "ro", self.root / "docs/review.zip")

    def test_destination_symlink_is_rejected(self):
        self.output.symlink_to(Path(self.temp.name) / "redirected.zip")
        with self.assertRaisesRegex(ValueError, "symlink"):
            create_review_bundle(self.root, "ro", self.output)


if __name__ == "__main__":
    unittest.main()
