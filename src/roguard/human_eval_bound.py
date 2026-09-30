"""Public evaluation entry point with end-to-end packet-binding integrity."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .human_eval import evaluate_adjudicated
from .packet_binding import binding_path, read_binding
from .review import sha256


def _binding_digest(answers: Path) -> str | None:
    path = binding_path(answers)
    return sha256(path) if path.exists() or path.is_symlink() else None


def evaluate_adjudicated_bound(
    root: Path,
    packet_dir: Path,
    reviewer_a: Path,
    reviewer_b: Path,
    adjudicated: Path,
    predictions_path: Path,
) -> dict:
    """Keep both reviewers' packet bindings stable through final scoring."""
    original = (_binding_digest(reviewer_a), _binding_digest(reviewer_b))
    report = evaluate_adjudicated(root, packet_dir, reviewer_a, reviewer_b,
                                  adjudicated, predictions_path)
    if original[0] is not None:
        if not read_binding(reviewer_a, report["packet_sha256"]["reviewer-a.jsonl"]):
            raise ValueError("Reviewer A packet binding disappeared during evaluation")
    if original[1] is not None:
        if not read_binding(reviewer_b, report["packet_sha256"]["reviewer-b.jsonl"]):
            raise ValueError("Reviewer B packet binding disappeared during evaluation")
    if (_binding_digest(reviewer_a), _binding_digest(reviewer_b)) != original:
        raise ValueError("Reviewer packet binding changed during evaluation")
    report["annotation_packet_binding_verified"] = all(original)
    if all(original):
        report["input_sha256"]["reviewer_a_packet_binding"] = original[0]
        report["input_sha256"]["reviewer_b_packet_binding"] = original[1]
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score frozen abstract predictions with packet-bound adjudications")
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("reviewer_a", type=Path)
    parser.add_argument("reviewer_b", type=Path)
    parser.add_argument("adjudicated", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_adjudicated_bound(
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
