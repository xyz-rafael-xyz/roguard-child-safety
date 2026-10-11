"""Descriptive out-of-family probe for the frozen v3 message model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from roguard.message_model import SyntheticMessageModel


def evaluate(bank_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError("Probe output already exists")
    data = bank_path.read_bytes()
    bank = json.loads(data)
    if set(bank) != {"ro", "uk"}:
        raise ValueError("Expected two language banks")
    cells = {}
    for language in ("ro", "uk"):
        model = SyntheticMessageModel.packaged(language)
        tp = fp = fn = tn = complete = 0
        errors = []
        for index, pair in enumerate(bank[language]):
            if (not isinstance(pair, list) or len(pair) != 2 or
                    not all(isinstance(text, str) and text.strip() for text in pair)):
                raise ValueError("Expected positive/negative text pair")
            predicted = [model.score(text) >= model.cutoff for text in pair]
            tp += predicted[0]
            fn += not predicted[0]
            fp += predicted[1]
            tn += not predicted[1]
            complete += predicted == [True, False]
            if predicted != [True, False]:
                errors.append({"pair_index": index, "positive_reviewed": predicted[0],
                               "negative_reviewed": predicted[1]})
        cells[language] = {"pairs": len(bank[language]), "complete_pairs": complete,
                           "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                           "errors": errors, "model_sha256": model.sha256}
    report = {"study": "v3_colloquial_probe", "status": "exploratory_diagnostic",
              "bank_sha256": hashlib.sha256(data).hexdigest(), "cells": cells,
              "real_child_language_accuracy_established": False}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.bank, args.output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
