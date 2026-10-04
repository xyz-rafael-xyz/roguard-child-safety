"""Write a content-free commitment to private v1 study packet bytes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .review import sha256
from .v1_study_kit import verify_v1_study_freeze

ROOT = Path(__file__).resolve().parents[2]


def create_v1_packet_seal(root: Path, owner_dir: Path, output: Path) -> dict:
    """Seal hashes before inference; the caller commits and pushes the file."""
    root, owner_dir = root.resolve(), owner_dir.expanduser().resolve()
    owner_path = owner_dir / "OWNER-MAP.json"
    if owner_path.is_symlink() or not owner_path.is_file():
        raise ValueError("Owner map must be a regular private file")
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if (not isinstance(owner, dict) or
            owner.get("study_kind") != "roguard_v1_candidate_independent_abstract" or
            owner.get("language") not in ("ro", "uk") or
            owner.get("category") not in ("D1", "S1")):
        raise ValueError("Owner map is not a corrected-v1 study")
    frozen = verify_v1_study_freeze(
        root, Path(owner["freeze_path"]), owner["language"], owner["category"])
    if (owner.get("freeze_sha256") != frozen["sha256"] or
            owner.get("candidate_taxonomy_sha256") != frozen["candidate_taxonomy_sha256"]):
        raise ValueError("Owner map differs from the pre-author freeze")
    packets = {}
    for key in ("reviewer-a", "reviewer-b"):
        path = owner_dir / f"{key}.jsonl"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Blind packet must be a regular file")
        packets[key] = sha256(path)
    if owner.get("packet_sha256") != packets:
        raise ValueError("Blind packets changed since assembly")
    source = owner_dir / "SOURCE-CARDS.jsonl"
    review = owner_dir / "CONTENT-REVIEW.json"
    if any(path.is_symlink() or not path.is_file() for path in (source, review)):
        raise ValueError("Sealed source and review must be regular private files")
    if owner.get("content_review_sha256") != sha256(review):
        raise ValueError("Content review changed since packet assembly")
    destination = output.expanduser().absolute()
    if (not destination.resolve().is_relative_to(root / "eval/prospective") or
            destination.suffix != ".json" or destination.exists() or
            destination.is_symlink()):
        raise ValueError("Packet seal needs a new JSON path under eval/prospective/")
    record = {
        "schema_version": 1,
        "study_kind": "roguard_v1_candidate_independent_abstract",
        "language": owner["language"], "category": owner["category"],
        "candidate_taxonomy_sha256": frozen["candidate_taxonomy_sha256"],
        "freeze_path": frozen["path"], "freeze_sha256": frozen["sha256"],
        "owner_map_sha256": sha256(owner_path),
        "packet_sha256": packets,
        "source_cards_sha256": sha256(source),
        "content_review_sha256": sha256(review),
        "no_card_text_or_intended_labels_in_seal": True,
        "human_chronology_verified_by_software": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    descriptor = os.open(destination, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(destination), "seal_sha256": sha256(destination),
            "language": owner["language"], "category": owner["category"],
            "contains_card_text_or_labels": False,
            "committed_and_pushed": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Commit to private v1 packet hashes before inference")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--owner-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = create_v1_packet_seal(args.root, args.owner_dir, args.output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
