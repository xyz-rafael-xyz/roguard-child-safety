"""Score an attested, held-out synthetic batch with a trained adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from roguard.model import HFClassifier
from roguard.metrics import metrics
from roguard.review import ReviewError, load_approved, taxonomy_sha256
from roguard.screen import screen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", action="append", required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = load_approved(args.root, args.batch)
    if not rows or any(row["split"] != "test" for row in rows):
        raise ReviewError("Evaluator accepts attested held-out test batches only")
    metadata = json.loads((args.adapter / "roguard_metadata.json").read_text(encoding="utf-8"))
    if metadata["taxonomy_sha256"] != taxonomy_sha256(args.root, metadata["language"]):
        raise ReviewError("Adapter taxonomy differs from test taxonomy")
    if set(args.batch) & set(metadata["batches"]):
        raise ReviewError("Test batches overlap the training/development batches")
    if any(row["language"] != metadata["language"] for row in rows):
        raise ReviewError("This adapter evaluates only its trained language")
    backend = HFClassifier(args.adapter)
    predictions = {}
    details = []
    for row in rows:
        thresholds = metadata["thresholds"].get(row["language"])
        if thresholds is None:
            raise ReviewError(f"No development calibration for {row['language']}")
        result = screen(row["text"], language=row["language"], source_kind=row["source_kind"], backend=backend, thresholds=thresholds)
        predictions[row["id"]] = result.labels
        details.append({"id": row["id"], "scores": result.scores, "predicted": result.labels, "expected": row["labels"]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError("Do not overwrite an evaluation run")
    args.output.write_text(json.dumps({"batches": args.batch, "adapter": str(args.adapter), "rows": len(rows), "metrics": metrics(rows, predictions), "predictions": details}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
