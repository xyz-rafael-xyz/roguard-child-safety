"""Evaluate an attested held-out batch with a local MLX QLoRA adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.metrics import metrics
from roguard.model import MLXClassifier
from roguard.qwen_output import PARSER_PROFILE, parse_qwen_binary
from roguard.review import CATEGORIES, ReviewError, load_approved, taxonomy_sha256
from roguard.screen import screen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", action="append", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qwen-empty-think-wrapper", action="store_true")
    args = parser.parse_args()
    rows = load_approved(args.root, args.batch)
    if not rows or any(row["split"] != "test" for row in rows):
        raise ReviewError("Evaluator accepts attested held-out test batches only")
    metadata = json.loads((args.adapter / "roguard_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("backend") != "mlx_lm_codes":
        raise ReviewError("Adapter is not a RoGuard MLX code model")
    if metadata["taxonomy_sha256"] != taxonomy_sha256(args.root, metadata["language"]):
        raise ReviewError("Adapter taxonomy differs from test taxonomy")
    if set(args.batch) & set(metadata["batches"]):
        raise ReviewError("Test batches overlap training or development batches")
    if any(row["language"] != metadata["language"] for row in rows):
        raise ReviewError("This adapter evaluates only its trained language")
    if args.qwen_empty_think_wrapper and metadata.get("base_model_id") != "mlx-community/Qwen3-4B-Instruct-2507-4bit":
        raise ReviewError("Qwen wrapper parser requires the pinned Qwen adapter")
    backend = MLXClassifier(args.adapter, binary_parser=parse_qwen_binary if args.qwen_empty_think_wrapper else None)
    predictions = {}
    details = []
    thresholds = {code: 0.5 for code in CATEGORIES}
    for row in rows:
        try:
            result = screen(row["text"], language=row["language"], source_kind=row["source_kind"],
                            backend=backend, thresholds=thresholds)
            predicted = result.labels
            parsed = True
        except ValueError as exc:
            if "Model output could not be parsed" not in str(exc):
                raise
            predicted = ()
            parsed = False
        predictions[row["id"]] = predicted
        details.append({"id": row["id"], "expected": row["labels"], "predicted": predicted,
                        "strict_parse": parsed})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Do not overwrite an evaluation run")
    args.output.write_text(json.dumps({
        "adapter": str(args.adapter), "batches": args.batch, "rows": len(rows),
        "parsed_rows": sum(item["strict_parse"] for item in details),
        "metrics": metrics(rows, predictions), "predictions": details,
        "score_semantics": "hard_code_output_not_calibrated_probability",
        "parser_profile": PARSER_PROFILE if args.qwen_empty_think_wrapper else "literal_da_nu_v1",
        "unparsed_policy": "count_as_no_predicted_labels_and_report_separately",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
