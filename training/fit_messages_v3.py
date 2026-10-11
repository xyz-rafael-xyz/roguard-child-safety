"""Fit a small bilingual character model on fictional messages only."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from roguard.message_model import SyntheticMessageModel
from training.generate_messages_v3 import generate

CONFIGS = ((2, 5, 0.3), (2, 5, 1.0), (2, 5, 3.0),
           (3, 5, 0.3), (3, 5, 1.0), (3, 5, 3.0))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_exact(path: Path, split: str, generator=generate) -> list[dict]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    if rows != generator(split):
        raise ValueError(f"{split} does not match the registered fictional generator")
    return rows


def metrics(rows: list[dict], scores: list[float], cutoff: float) -> dict:
    if len(rows) != len(scores):
        raise ValueError("Score count differs from row count")
    by_pair: dict[str, dict[int, bool]] = {}
    counts = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    family_errors: dict[str, int] = {}
    for row, score in zip(rows, scores):
        truth, predicted = row["label"], score >= cutoff
        counts[("tp" if predicted else "fn") if truth else
               ("fp" if predicted else "tn")] += 1
        by_pair.setdefault(row["pair_id"], {})[truth] = predicted == bool(truth)
        if predicted != bool(truth):
            family_errors[row["family"]] = family_errors.get(row["family"], 0) + 1
    if any(set(pair) != {0, 1} for pair in by_pair.values()):
        raise ValueError("Incomplete contrast pair")
    return {**counts, "pairs": len(by_pair),
            "complete_pairs": sum(all(pair.values()) for pair in by_pair.values()),
            "family_errors": dict(sorted(family_errors.items()))}


def select_cutoff(rows: list[dict], scores: list[float]) -> tuple[float, dict]:
    sorted_scores = sorted(set(scores))
    cutoffs = [0.0, 1.0] + [(a + b) / 2 for a, b in zip(sorted_scores, sorted_scores[1:])]
    choices = [(metrics(rows, scores, cutoff), cutoff) for cutoff in cutoffs]
    report, cutoff = max(choices, key=lambda item: (
        item[0]["complete_pairs"], item[0]["tp"], -item[0]["fp"],
        -abs(item[1] - 0.5), -item[1]))
    return cutoff, report


def export_model(vectorizer: TfidfVectorizer, model: LogisticRegression,
                 cutoff: float, language: str, path: Path) -> str:
    names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    payload = {"format": "roguard-character-tfidf-v1", "language": language,
               "ngram_range": list(vectorizer.ngram_range),
               "cutoff": cutoff, "intercept": float(model.intercept_[0]),
               "features": [[name, float(vectorizer.idf_[index]),
                             float(coefficients[index])]
                            for index, name in enumerate(names)]}
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    path.write_bytes(gzip.compress(data, compresslevel=9, mtime=0))
    return digest(path)


def fit(train_path: Path, dev_path: Path, output: Path,
        generator=generate, study: str = "v3") -> dict:
    if output.exists():
        raise FileExistsError("Selection output already exists")
    train = load_exact(train_path, "train", generator)
    dev = load_exact(dev_path, "dev", generator)
    output.mkdir(parents=True)
    candidates = []
    for low, high, c in CONFIGS:
        cells = {}
        fitted = {}
        for language in ("ro", "uk"):
            train_rows = [row for row in train if row["language"] == language]
            dev_rows = [row for row in dev if row["language"] == language]
            vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(low, high),
                                         min_df=2, max_features=60000,
                                         sublinear_tf=True, lowercase=True, norm="l2")
            features = vectorizer.fit_transform([row["text"] for row in train_rows])
            model = LogisticRegression(C=c, max_iter=1000, solver="lbfgs",
                                       class_weight="balanced", random_state=3100)
            model.fit(features, [row["label"] for row in train_rows])
            scores = model.predict_proba(vectorizer.transform(
                [row["text"] for row in dev_rows]))[:, 1].tolist()
            cutoff, report = select_cutoff(dev_rows, scores)
            cells[language] = {"cutoff": cutoff, "metrics": report,
                               "features": len(vectorizer.vocabulary_)}
            fitted[language] = (vectorizer, model, cutoff)
        candidate = {"ngram_range": [low, high], "C": c, "cells": cells}
        print(json.dumps(candidate, ensure_ascii=False), flush=True)
        candidates.append((candidate, fitted))
    selected, models = max(candidates, key=lambda item: (
        min(item[0]["cells"][language]["metrics"]["complete_pairs"]
            for language in ("ro", "uk")),
        sum(item[0]["cells"][language]["metrics"]["complete_pairs"]
            for language in ("ro", "uk")),
        min(item[0]["cells"][language]["metrics"]["tp"]
            for language in ("ro", "uk")),
        -sum(item[0]["cells"][language]["metrics"]["fp"]
             for language in ("ro", "uk")),
        -item[0]["ngram_range"][1], -item[0]["C"]))
    files = {}
    for language, (vectorizer, model, cutoff) in models.items():
        path = output / f"message-model-{language}.json.gz"
        files[language] = {"filename": path.name,
                           "sha256": export_model(vectorizer, model, cutoff, language, path)}
        subset = [row["text"] for row in dev if row["language"] == language]
        reference = model.predict_proba(vectorizer.transform(subset))[:, 1]
        standalone = SyntheticMessageModel(path.read_bytes())
        maximum_difference = max(abs(float(expected) - standalone.score(text))
                                 for text, expected in zip(subset, reference))
        if maximum_difference > 1e-10:
            raise AssertionError("Exported scorer differs from fitted estimator")
        files[language]["maximum_score_difference"] = maximum_difference
    record = {"status": "selected_before_test", "study": study,
              "task": "fictional_clear_support_signal",
              "source": "locally_authored_synthetic_only", "train_sha256": digest(train_path),
              "dev_sha256": digest(dev_path), "selected": selected,
              "models": files,
              "candidates": [candidate for candidate, _ in candidates],
              "real_child_language_accuracy_established": False,
              "external_action_taken": False}
    (output / "selection.json").write_text(json.dumps(record, ensure_ascii=False,
        indent=2) + "\n", encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--dev", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(fit(args.train, args.dev, args.output)["selected"],
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
