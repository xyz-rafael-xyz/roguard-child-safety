"""Model-agnostic evaluation on attested abstract one-condition pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import metrics
from .review import APPLIES_TO, CATEGORIES, INDEPENDENT_ORIGIN, load_approved, sha256


def evaluate_pairs(rows: list[dict], items: list[dict]) -> dict:
    """Score strict predictions without trusting supplied expected labels."""
    if not rows or len(rows) % 2 or len(items) != len(rows):
        raise ValueError("Expected one prediction per row in complete adjacent pairs")
    if any(row.get("origin") == INDEPENDENT_ORIGIN for row in rows):
        raise ValueError("Author-intended labels are unadjudicated; use roguard-human-eval")
    predictions: dict[str, tuple[str, ...]] = {}
    parsed = 0
    for row, item in zip(rows, items):
        if item.get("id") != row["id"]:
            raise ValueError("Prediction IDs must match the attested row order")
        if "expected" in item and item["expected"] != row["labels"]:
            raise ValueError("Prediction file changed an expected label")
        raw = item.get("predicted")
        if not isinstance(raw, list) or len(raw) != len(set(raw)) or any(
            code not in CATEGORIES or row["source_kind"] not in APPLIES_TO[code] for code in raw
        ):
            raise ValueError("Predicted categories must be unique and applicable")
        if type(item.get("strict_parse")) is not bool:
            raise ValueError("Each prediction needs a boolean strict_parse")
        if not item["strict_parse"] and raw:
            raise ValueError("Unparsed output must abstain from category labels")
        parsed += item["strict_parse"]
        predictions[row["id"]] = tuple(code for code in CATEGORIES if code in raw)
    for left, right in zip(rows[::2], rows[1::2]):
        if (left["labels"] or len(right["labels"]) != 1 or
                left["source_kind"] != right["source_kind"]):
            raise ValueError("Batch is not an adjacent one-condition pair test")
    exact_rows = sum(predictions[row["id"]] == tuple(row["labels"]) for row in rows)
    exact_pairs = sum(predictions[left["id"]] == () and
                      predictions[right["id"]] == tuple(right["labels"])
                      for left, right in zip(rows[::2], rows[1::2]))
    flips = sum(right["labels"][0] not in predictions[left["id"]] and
                right["labels"][0] in predictions[right["id"]]
                for left, right in zip(rows[::2], rows[1::2]))
    return {
        "cards": len(rows), "pairs": len(rows) // 2, "parsed_cards": parsed,
        "exact_cards": exact_rows, "exact_pairs": exact_pairs,
        "correct_direction_flips": flips,
        "false_review_on_negatives": sum(bool(predictions[row["id"]]) for row in rows[::2]),
        "per_category": metrics(rows, predictions),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model labels on attested abstract contrast pairs")
    parser.add_argument("--batch", required=True, help="Registered batch ID")
    parser.add_argument("--predictions", type=Path, required=True, help="JSON file with predictions or outputs list")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, help="Optional new JSON report path; stdout otherwise")
    args = parser.parse_args()
    try:
        root = args.root.resolve()
        rows = load_approved(root, [args.batch])
        if any(row["origin"] == INDEPENDENT_ORIGIN for row in rows):
            raise ValueError("Author-intended labels are unadjudicated; use roguard-human-eval")
        payload = json.loads(args.predictions.read_text(encoding="utf-8"))
        if payload.get("batches") != [args.batch]:
            raise ValueError("Prediction file must name exactly the selected batch")
        items = payload.get("predictions", payload.get("outputs"))
        if not isinstance(items, list):
            raise ValueError("Prediction file needs a predictions or outputs list")
        result = {"batch": args.batch,
                  "batch_sha256": sha256(root / "data" / "synthetic" / f"{args.batch}.jsonl"),
                  "attestation_sha256": sha256(root / "data" / "synthetic" / f"{args.batch}.review.json"),
                  "predictions_sha256": sha256(args.predictions),
                  "scope": "abstract_symbolic_pairs_only", **evaluate_pairs(rows, items)}
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            target = args.output.resolve()
            if target.exists():
                raise FileExistsError("Evaluation report already exists")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
