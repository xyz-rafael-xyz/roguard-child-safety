"""A pre-author freeze covers the source tree that existed at that commit."""

import subprocess
import tempfile
import unittest
from pathlib import Path

from eval.verify_registered_study_freeze import _source_files_at_commit


class RegisteredSourceInventoryTests(unittest.TestCase):
    def test_later_unrelated_module_does_not_change_frozen_source_set(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            source = root / "src/roguard"
            source.mkdir(parents=True)
            (source / "predict.py").write_text("# frozen\n")
            subprocess.run(["git", "add", "src/roguard/predict.py"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                            "commit", "-qm", "freeze source"], cwd=root, check=True)
            frozen = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                             text=True).strip()
            (source / "review_bundle.py").write_text("# added later\n")
            subprocess.run(["git", "add", "src/roguard/review_bundle.py"], cwd=root, check=True)
            subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                            "commit", "-qm", "later review tool"], cwd=root, check=True)
            self.assertEqual(_source_files_at_commit(root, frozen), {"src/roguard/predict.py"})


if __name__ == "__main__":
    unittest.main()
