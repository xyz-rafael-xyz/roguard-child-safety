"""Replay frozen synthetic reports without loading optional model runtimes."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from eval.v25_metrics import evaluate as v25_evaluate
from eval.v25_metrics import target_passed
from eval.v26_metrics import evaluate as fact_evaluate
from roguard.review import sha256
from roguard.v26_facts import decision_score

ROOT = Path(__file__).resolve().parents[1]


def _rows(relative: str) -> list[dict]:
    return [json.loads(line) for line in (ROOT / relative).read_text(encoding="utf-8").splitlines()]


class FrozenSyntheticRecordsTests(unittest.TestCase):
    def test_v25_sealed_result_recomputes_from_saved_scores(self):
        rows = _rows("data/synthetic/v25/test.jsonl")
        predictions = _rows("eval/prospective/v25-synthetic-predictions.jsonl")
        result = json.loads((ROOT / "eval/prospective/v25-synthetic-result.json").read_text())
        manifest = json.loads((ROOT / "models/bi-mmbert-v25-abstract/research.json").read_text())
        self.assertEqual(len(rows), len(predictions))
        self.assertEqual([row["id"] for row in rows], [row["id"] for row in predictions])
        self.assertEqual([row["labels"] for row in rows],
                         [row["expected"] for row in predictions])
        self.assertEqual(result["test_sha256"], sha256(ROOT / "data/synthetic/v25/test.jsonl"))
        self.assertEqual(result["predictions_sha256"], sha256(
            ROOT / "eval/prospective/v25-synthetic-predictions.jsonl"))
        self.assertEqual(result["selected_model_manifest_sha256"], sha256(
            ROOT / "models/bi-mmbert-v25-abstract/research.json"))
        model_scores = [item["model_score"] for item in predictions]
        baseline_scores = [item["char_baseline_score"] for item in predictions]
        self.assertEqual(result["model"], v25_evaluate(rows, model_scores,
                                                        manifest["thresholds"]))
        baseline_cutoffs = json.loads((
            ROOT / "models/bi-mmbert-v25-abstract/development-selection.json"
        ).read_text())["char_baseline_dev"]["thresholds"]
        self.assertEqual(result["char_ngram_baseline"],
                         v25_evaluate(rows, baseline_scores, baseline_cutoffs))
        self.assertFalse(result["fixed_target_passed"])
        self.assertFalse(target_passed(result["model"]))
        self.assertEqual(result["model"]["exact_pairs"], 165)

    def test_v26_to_v29_failed_development_selections_recompute(self):
        for version, directory in (
                (26, "bi-mmbert-v26-facts"), (27, "bi-mmbert-v27-balanced"),
                (28, "bi-v28-hybrid"), (29, "bi-v29-hybrid")):
            with self.subTest(version=version):
                rows = _rows(f"data/synthetic/v{version}/dev.jsonl")
                artifact = ROOT / "models" / directory
                selection_path = artifact / "development-selection.json"
                selection = json.loads(selection_path.read_text(encoding="utf-8"))
                manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
                self.assertFalse((ROOT / f"data/synthetic/v{version}/test.jsonl").exists())
                self.assertEqual(manifest["development_selection_sha256"], sha256(selection_path))
                self.assertFalse(selection["development_eligible_for_test_reveal"])
                self.assertFalse(manifest["development_eligible_for_test_reveal"])
                if version in (26, 27):
                    candidate = next(item for item in selection["candidates"]
                                     if item["epoch"] == selection["selected_epoch"])
                    scores = candidate["scores"]
                    report = candidate["report"]
                else:
                    scores = selection["scores"]
                    report = selection["report"]
                    self.assertEqual(scores, [decision_score(
                        item["composed_fact_scores"], row["source_kind"])
                        for row, item in zip(rows, selection["evidence"])])
                self.assertEqual(report, fact_evaluate(rows, scores,
                                                        manifest["thresholds"]))
                self.assertFalse(target_passed(report))

    @unittest.skipUnless((ROOT / "models/bi-mmbert-v25-abstract/adapter_model.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_v25_private_weight_matches_manifest(self):
        artifact = ROOT / "models/bi-mmbert-v25-abstract"
        manifest = json.loads((artifact / "research.json").read_text())
        self.assertEqual(manifest["weight_sha256"], sha256(artifact / "adapter_model.safetensors"))

    @unittest.skipUnless((ROOT / "models/bi-mmbert-v26-facts/adapter_model.safetensors").exists(),
                         "research adapter weights are held in the private archive")
    def test_v26_v27_private_weights_match_manifests(self):
        for directory in ("bi-mmbert-v26-facts", "bi-mmbert-v27-balanced"):
            with self.subTest(directory=directory):
                artifact = ROOT / "models" / directory
                manifest = json.loads((artifact / "research.json").read_text())
                self.assertEqual(manifest["weight_sha256"], sha256(artifact / "adapter_model.safetensors"))


if __name__ == "__main__":
    unittest.main()
