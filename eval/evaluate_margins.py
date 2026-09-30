"""Open a previously attested test once with frozen development thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.metrics import metrics
from roguard.review import CATEGORIES, ReviewError, load_approved, sha256, taxonomy_sha256
from roguard.screen import screen
from roguard.token_margin import MLXTokenMarginClassifier

from calibrate_margins import counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Do not overwrite a held-out evaluation")
    calibration_path = root / "eval" / "runs" / "ro-v5-dev-thresholds.json"
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    adapter = root / "checkpoints" / "ro-v4" / "selected-adapter"
    metadata = json.loads((adapter / "roguard_metadata.json").read_text(encoding="utf-8"))
    selection_path = root / "eval" / "runs" / "ro-v4-dev-selection.json"
    if (calibration.get("status") != "development_only_token_margin_calibration" or
            calibration.get("development_batch") != "batch-0010" or
            calibration.get("development_sha256") != sha256(root / "data" / "synthetic" / "batch-0010.jsonl") or
            calibration.get("adapter_weight_sha256") != sha256(adapter / "adapters.safetensors") or
            calibration.get("selection_record_sha256") != sha256(selection_path) or
            calibration.get("prompt_sha256") != sha256(root / "src" / "roguard" / "prompt_v4.py") or
            calibration.get("scorer_sha256") != sha256(root / "src" / "roguard" / "token_margin.py") or
            calibration.get("calibration_code_sha256") != sha256(root / "eval" / "calibrate_margins.py") or
            set(calibration.get("thresholds", {})) != set(CATEGORIES) or
            metadata.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            set(metadata.get("batches", ())) != {"batch-0009", "batch-0010"}):
        raise ReviewError("Development threshold or adapter provenance differs")
    rows = load_approved(root, ["batch-0013"])
    if len(rows) != 96 or any(row["split"] != "test" or row["language"] != "ro" for row in rows):
        raise ReviewError("Expected the frozen Romanian batch-0013 test")
    if any(left["labels"] or len(right["labels"]) != 1 or left["source_kind"] != right["source_kind"]
           for left, right in zip(rows[::2], rows[1::2])):
        raise ReviewError("Expected adjacent one-condition test pairs")
    model = MLXTokenMarginClassifier(adapter)
    predictions = {}
    details = []
    for row in rows:
        result = screen(row["text"], language="ro", source_kind=row["source_kind"],
                        backend=model, thresholds=calibration["thresholds"])
        predictions[row["id"]] = result.labels
        details.append({"id": row["id"], "expected": row["labels"],
                        "predicted": result.labels, "scores": result.scores, "strict_parse": True})
    summary = counts(rows, predictions)
    summary["correct_direction_flips"] = sum(
        right["labels"][0] not in predictions[left["id"]] and
        right["labels"][0] in predictions[right["id"]]
        for left, right in zip(rows[::2], rows[1::2]))
    result = {
        "status": "frozen_v5_token_margin_test", "batches": ["batch-0013"],
        "test_sha256": sha256(root / "data" / "synthetic" / "batch-0013.jsonl"),
        "attestation_sha256": sha256(root / "data" / "synthetic" / "batch-0013.review.json"),
        "threshold_record_sha256": sha256(calibration_path),
        "adapter_weight_sha256": sha256(adapter / "adapters.safetensors"),
        "prompt_sha256": sha256(root / "src" / "roguard" / "prompt_v4.py"),
        "score_semantics": calibration["score_semantics"],
        "thresholds": calibration["thresholds"], "rows": len(rows),
        "counts": summary, "per_category": metrics(rows, predictions), "predictions": details,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
