"""Check a human-approved independent abstract batch without printing its cards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .review import INDEPENDENT_ORIGIN, load_approved, sha256
from .intake_overlap import audit_prior_overlap

ROOT = Path(__file__).resolve().parents[2]


def check_independent_batch(root: Path, batch: str) -> dict:
    root = root.resolve()
    rows = load_approved(root, [batch])
    if any(row["origin"] != INDEPENDENT_ORIGIN for row in rows):
        raise ValueError("Batch is not an independently authored abstract batch")
    provenance = json.loads((root / f"data/synthetic/{batch}.provenance.json").read_text())
    review = json.loads((root / f"data/synthetic/{batch}.review.json").read_text())
    return {
        "batch": batch, "language": rows[0]["language"],
        "category": provenance["category"], "cards": len(rows), "pairs": len(rows) // 2,
        "batch_sha256": sha256(root / f"data/synthetic/{batch}.jsonl"),
        "provenance_sha256": sha256(root / f"data/synthetic/{batch}.provenance.json"),
        "taxonomy_sha256": review["taxonomy_sha256"],
        "model_freeze_sha256": provenance["model_freeze_sha256"],
        "model_frozen_before_human_authorship_verified": False,
        "author_declarations_structurally_checked": True,
        "declared_factor_coverage_structurally_checked": True,
        "language_script_structurally_checked": True,
        "language_quality_verified": False,
        "safety_review_declaration_structurally_checked": True,
        "actual_author_independence_verified": False,
        "content_safety_judgment_verified_by_software": False,
        "intended_labels_adjudicated": False,
        "one_fact_contrast_verified_by_software": False,
        "prior_overlap_audit": audit_prior_overlap(root, rows, batch),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check independent abstract batch declarations")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        print(json.dumps(check_independent_batch(args.root, args.batch), indent=2))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
