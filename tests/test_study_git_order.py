"""Mechanical Git-history fixtures; no human study is represented here."""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eval import verify_independent_study_git_order as chronology

BATCH = "batch-0099"
FREEZE = "eval/prospective/fixture-freeze.json"
PREDICTION = "eval/prospective/fixture-predictions.json"
SUFFIXES = (".jsonl", ".preview.md", ".provenance.json", ".review.json")


def _commit(root: Path, message: str) -> str:
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(["git", "-c", "user.name=Fixture", "-c", "user.email=fixture@example.test",
                    "commit", "-qm", message], cwd=root, check=True)
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                   text=True).strip()


def _fixture(root: Path, *, batch_first: bool = False) -> tuple[str, str, str]:
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "eval/prospective").mkdir(parents=True)
    (root / "data/synthetic").mkdir(parents=True)
    (root / FREEZE).write_text('{"fixture":true}\n')
    freeze_commit = _commit(root, "fixture freeze")
    prediction = {"schema_version": 2, "packet_sha256": "0" * 64,
                  "language": "ro", "category": "D1", "model_id": "fixture",
                  "predictions": [{"item_id": f"opaque_{number}",
                                   "predicted": [], "strict_parse": True}
                                  for number in range(96)]}
    provenance = {"batch": BATCH, "language": "ro", "category": "D1",
                  "label_status": "author_intended_not_adjudicated",
                  "model_freeze_path": FREEZE, "model_freeze_sha256": "a" * 64}

    def add_prediction():
        (root / PREDICTION).write_text(json.dumps(prediction) + "\n")
        return _commit(root, "fixture blind prediction")

    def add_batch():
        for suffix in SUFFIXES:
            path = root / f"data/synthetic/{BATCH}{suffix}"
            path.write_text(json.dumps(provenance) + "\n" if suffix == ".provenance.json"
                            else "fixture approved abstract batch\n")
        return _commit(root, "fixture labeled batch")

    if batch_first:
        batch_commit = add_batch()
        prediction_commit = add_prediction()
    else:
        prediction_commit = add_prediction()
        batch_commit = add_batch()
    return freeze_commit, prediction_commit, batch_commit


class StudyGitOrderTests(unittest.TestCase):
    def test_sealed_prediction_precedes_labeled_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            freeze_commit, prediction_commit, batch_commit = _fixture(root)
            with patch.object(chronology, "verify_study_freeze", return_value={
                    "path": FREEZE, "sha256": "a" * 64,
                    "prediction_model_id": "fixture"}):
                result = chronology.verify(root, BATCH, Path(PREDICTION))
            self.assertEqual([result[key] for key in (
                "freeze_commit", "prediction_commit", "labeled_batch_commit")],
                [freeze_commit, prediction_commit, batch_commit])
            self.assertTrue(result["git_commit_order_verified"])
            self.assertFalse(result["external_push_order_verified"])

    def test_labeled_batch_committed_first_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root, batch_first=True)
            with patch.object(chronology, "verify_study_freeze", return_value={
                    "path": FREEZE, "sha256": "a" * 64,
                    "prediction_model_id": "fixture"}):
                with self.assertRaisesRegex(ValueError, "prediction commit before labeled"):
                    chronology.verify(root, BATCH, Path(PREDICTION))

    def test_changed_prediction_after_seal_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _fixture(root)
            path = root / PREDICTION
            path.write_text(path.read_text() + "\n")
            _commit(root, "fixture changed prediction")
            with patch.object(chronology, "verify_study_freeze", return_value={
                    "path": FREEZE, "sha256": "a" * 64,
                    "prediction_model_id": "fixture"}):
                with self.assertRaisesRegex(ValueError, "one immutable addition"):
                    chronology.verify(root, BATCH, Path(PREDICTION))


if __name__ == "__main__":
    unittest.main()
