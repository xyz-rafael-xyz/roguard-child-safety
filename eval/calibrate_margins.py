"""Fit score thresholds on Romanian development pairs without opening a test."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

from roguard.metrics import metrics
from roguard.review import APPLIES_TO, CATEGORIES, load_approved, sha256
from roguard.token_margin import MLXTokenMarginClassifier

GROUPS = (("D1",), ("R1",), ("A1", "S1"), ("P1",), ("G1",))


def candidates(values: list[float]) -> tuple[float, ...]:
    ordered = sorted(set(values))
    if not ordered or any(not math.isfinite(value) or not 0 <= value <= 1 for value in ordered):
        raise ValueError("Choice scores must be finite values in [0, 1]")
    return tuple(sorted({0.0, 1.0, *((a + b) / 2 for a, b in zip(ordered, ordered[1:]))}))


def predictions_for(rows: list[dict], scores: dict[str, dict[str, float]], thresholds: dict[str, float]) -> dict[str, tuple[str, ...]]:
    return {
        row["id"]: tuple(code for code in CATEGORIES
                         if code in thresholds and row["source_kind"] in APPLIES_TO[code]
                         and scores[row["id"]][code] >= thresholds[code])
        for row in rows
    }


def counts(rows: list[dict], predictions: dict[str, tuple[str, ...]]) -> dict[str, int]:
    if len(rows) % 2:
        raise ValueError("Expected complete adjacent development pairs")
    for left, right in zip(rows[::2], rows[1::2]):
        if (left["labels"] or len(right["labels"]) != 1 or
                left["source_kind"] != right["source_kind"]):
            raise ValueError("Expected negative/positive one-factor pairs")
    return {
        "exact_rows": sum(predictions[row["id"]] == tuple(row["labels"]) for row in rows),
        "exact_pairs": sum(predictions[left["id"]] == () and
                           predictions[right["id"]] == tuple(right["labels"])
                           for left, right in zip(rows[::2], rows[1::2])),
        "true_positives": sum(len(set(predictions[row["id"]]) & set(row["labels"])) for row in rows),
        "false_positives": sum(len(set(predictions[row["id"]]) - set(row["labels"])) for row in rows),
        "false_review_on_negatives": sum(bool(predictions[row["id"]]) for row in rows[::2]),
    }


def select_thresholds(rows: list[dict], scores: dict[str, dict[str, float]]) -> dict[str, float]:
    selected = {}
    for group in GROUPS:
        applicable_rows = [row for row in rows if any(row["source_kind"] in APPLIES_TO[code] for code in group)]
        if not applicable_rows or any(row["labels"] and row["labels"][0] not in group for row in applicable_rows):
            raise ValueError("Development group lacks complete category pairs")
        grids = [candidates([scores[row["id"]][code] for row in applicable_rows
                             if row["source_kind"] in APPLIES_TO[code]]) for code in group]
        best_key = None
        best_thresholds = None
        for trial in itertools.product(*grids):
            thresholds = dict(zip(group, trial))
            result = counts(applicable_rows, predictions_for(applicable_rows, scores, thresholds))
            key = (result["exact_pairs"], result["exact_rows"], result["true_positives"],
                   -result["false_positives"], -sum(abs(value - 0.5) for value in trial),
                   tuple(-value for value in trial))
            if best_key is None or key > best_key:
                best_key, best_thresholds = key, thresholds
        assert best_thresholds is not None
        selected.update(best_thresholds)
    return {code: selected[code] for code in CATEGORIES}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    rows = load_approved(root, ["batch-0010"])
    if len(rows) != 36 or any(row["split"] != "dev" for row in rows):
        raise ValueError("Expected the frozen v4 development pairs")
    adapter = root / "checkpoints" / "ro-v4" / "selected-adapter"
    metadata = json.loads((adapter / "roguard_metadata.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "eval" / "runs" / "ro-v4-dev-selection.json").read_text(encoding="utf-8"))
    if (metadata.get("selected_step") != 500 or selection.get("selected_step") != 500 or
            sha256(adapter / "adapters.safetensors") != selection["selected_weight_sha256"]):
        raise ValueError("Selected adapter differs from the committed development choice")
    model = MLXTokenMarginClassifier(adapter)
    scores = {row["id"]: {code: score for code, score in
                          model.score(row["text"], row["language"], row["source_kind"]).items()
                          if row["source_kind"] in APPLIES_TO[code]} for row in rows}
    thresholds = select_thresholds(rows, scores)
    predictions = predictions_for(rows, scores, thresholds)
    cv_predictions = {}
    for left, right in zip(rows[::2], rows[1::2]):
        train_rows = [row for row in rows if row["id"] not in {left["id"], right["id"]}]
        cv_thresholds = select_thresholds(train_rows, scores)
        cv_predictions.update(predictions_for([left, right], scores, cv_thresholds))
    result = {
        "status": "development_only_token_margin_calibration",
        "development_batch": "batch-0010",
        "development_sha256": sha256(root / "data" / "synthetic" / "batch-0010.jsonl"),
        "adapter_weight_sha256": sha256(adapter / "adapters.safetensors"),
        "selection_record_sha256": sha256(root / "eval" / "runs" / "ro-v4-dev-selection.json"),
        "prompt_sha256": sha256(root / "src" / "roguard" / "prompt_v4.py"),
        "scorer_sha256": sha256(root / "src" / "roguard" / "token_margin.py"),
        "calibration_code_sha256": sha256(root / "eval" / "calibrate_margins.py"),
        "score_semantics": "conditional_da_nu_next_token_score_not_calibrated_risk_probability",
        "selection_rule": "max_exact_pairs_then_exact_rows_then_true_positives_then_fewer_false_positives_then_closest_to_half_then_lower_threshold",
        "thresholds": thresholds,
        "development_counts": counts(rows, predictions),
        "development_metrics": metrics(rows, predictions),
        "leave_one_pair_out_counts": counts(rows, cv_predictions),
        "scores": [{"id": row["id"], "expected": row["labels"], "scores": scores[row["id"]],
                    "predicted": predictions[row["id"]]} for row in rows],
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError("Development calibration already exists")
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
