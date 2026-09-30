"""Public human evaluator with independent-study model-freeze ID binding."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .human_eval_bound import evaluate_adjudicated_bound
from .review import sha256
from .study_freeze import verify_study_freeze


def evaluate_adjudicated_registered(root: Path, packet_dir: Path,
                                    reviewer_a: Path, reviewer_b: Path,
                                    adjudicated: Path, predictions_path: Path) -> dict:
    """Reject an independent prediction identifier unlike its pre-author freeze."""
    root = root.resolve()
    report = evaluate_adjudicated_bound(root, packet_dir, reviewer_a, reviewer_b,
                                        adjudicated, predictions_path)
    if "source_provenance" not in report["input_sha256"]:
        return report  # Historical generator-origin demonstrations have no author freeze.
    path = root / "data/synthetic" / f"{report['batch']}.provenance.json"
    original = report["input_sha256"]["source_provenance"]
    if sha256(path) != original:
        raise ValueError("Source provenance changed after the packet-bound evaluation")
    provenance = json.loads(path.read_text(encoding="utf-8"))
    freeze = verify_study_freeze(root, Path(provenance["model_freeze_path"]),
                                 report["language"], report["category"])
    if (sha256(path) != original or
            freeze["sha256"] != provenance["model_freeze_sha256"] or
            report["model_id"] != freeze["prediction_model_id"]):
        raise ValueError("Prediction identifier differs from the pre-author model freeze")
    report["model_freeze_sha256"] = freeze["sha256"]
    report["prediction_model_id_matches_freeze"] = True
    report["model_execution_attested"] = False
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score abstract predictions with packet and model-freeze binding")
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("reviewer_a", type=Path)
    parser.add_argument("reviewer_b", type=Path)
    parser.add_argument("adjudicated", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_adjudicated_registered(
            args.root, args.packet_dir, args.reviewer_a,
            args.reviewer_b, args.adjudicated, args.predictions)
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(rendered)
        else:
            print(rendered, end="")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
