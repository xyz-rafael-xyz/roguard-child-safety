import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from roguard.review_bundle import create_review_bundle
from roguard.v1_review import candidate_digest

ROOT = Path(__file__).resolve().parents[1]


class V1ReviewBundleTests(unittest.TestCase):
    def test_exact_candidate_bundle_without_cards_or_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            for language in ("ro", "uk"):
                target = Path(directory) / f"candidate-{language}.zip"
                result = create_review_bundle(ROOT, language, target, candidate_v1=True)
                self.assertEqual(result["taxonomy_sha256"], candidate_digest(ROOT, language))
                with zipfile.ZipFile(target) as archive:
                    manifest = json.loads(archive.read("manifest.json"))
                    self.assertEqual(manifest["taxonomy_version"], "v1_candidate")
                    self.assertFalse(manifest["model_scores_or_labeled_cards_included"])
                    for name, digest in manifest["files_sha256"].items():
                        self.assertEqual(hashlib.sha256(archive.read(name)).hexdigest(), digest)
                    self.assertIn(f"taxonomy/taxonomy_{language}_v1_candidate.md", archive.namelist())
                    self.assertFalse(any(name.startswith("data/") or name.startswith("models/")
                                         for name in archive.namelist()))
                with self.assertRaises(FileExistsError):
                    create_review_bundle(ROOT, language, target, candidate_v1=True)


if __name__ == "__main__":
    unittest.main()
