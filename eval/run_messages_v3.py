"""Score the once-opened fictional-message holdout without text in reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.message_model import SyntheticMessageModel
from training.generate_messages_v3 import generate
from training.fit_messages_v3 import digest, load_exact, metrics

TARGET_PAIRS = 85
TARGET_RECALL = 90
MAX_FALSE_REVIEWS = 10


def evaluate(selection_dir: Path, test_path: Path, output_dir: Path,
             generator=generate, target_pairs: int = TARGET_PAIRS,
             target_recall: int = TARGET_RECALL,
             max_false_reviews: int = MAX_FALSE_REVIEWS,
             study: str = "v3_fictional_message_holdout") -> dict:
    if output_dir.exists():
        raise FileExistsError("Test output already exists")
    record = json.loads((selection_dir / "selection.json").read_text(encoding="utf-8"))
    if record.get("status") != "selected_before_test" or set(record["models"]) != {"ro", "uk"}:
        raise ValueError("Missing frozen bilingual selection")
    rows = load_exact(test_path, "test", generator)
    predictions = []
    cells = {}
    for language in ("ro", "uk"):
        model_record = record["models"][language]
        path = selection_dir / model_record["filename"]
        if digest(path) != model_record["sha256"]:
            raise ValueError("Selected model changed")
        model = SyntheticMessageModel(path.read_bytes())
        if model.language != language or abs(
                model.cutoff - record["selected"]["cells"][language]["cutoff"]) > 1e-12:
            raise ValueError("Selected model identity or cutoff changed")
        subset = [row for row in rows if row["language"] == language]
        scores = [model.score(row["text"]) for row in subset]
        report = metrics(subset, scores, model.cutoff)
        report["cutoff"] = model.cutoff
        report["model_sha256"] = model.sha256
        report["high_confidence_false_negatives"] = sum(
            row["label"] == 1 and score < 0.1 for row, score in zip(subset, scores))
        report["high_confidence_false_reviews"] = sum(
            row["label"] == 0 and score >= 0.9 for row, score in zip(subset, scores))
        report["target_passed"] = (report["complete_pairs"] >= target_pairs and
                                   report["tp"] >= target_recall and
                                   report["fp"] <= max_false_reviews)
        cells[language] = report
        predictions.extend({"id": row["id"], "pair_id": row["pair_id"],
                            "language": language, "synthetic_score": score,
                            "predicted": int(score >= model.cutoff)}
                           for row, score in zip(subset, scores))
    result = {"study": study, "status": "once_scored",
              "test_sha256": digest(test_path),
              "selection_sha256": digest(selection_dir / "selection.json"),
              "test_rows": len(rows), "cells": cells,
              "target": {"complete_pairs_per_language": target_pairs,
                         "positive_recall_per_language": target_recall,
                         "maximum_false_reviews_per_language": max_false_reviews},
              "target_passed": all(cell["target_passed"] for cell in cells.values()),
              "real_child_language_accuracy_established": False,
              "external_action_taken": False}
    output_dir.mkdir(parents=True)
    (output_dir / "result.json").write_text(json.dumps(result, ensure_ascii=False,
        indent=2) + "\n", encoding="utf-8")
    (output_dir / "predictions.jsonl").write_text("".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
        for row in sorted(predictions, key=lambda row: row["id"])), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection-dir", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.selection_dir, args.test, args.output_dir),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
