"""Make two locally blinded abstract-card packets for independent D1/S1 review."""

from __future__ import annotations

import argparse
import json
import os
import secrets
import tempfile
from pathlib import Path

from .review import load_approved, sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[2]
KINDS = {"D1": "message", "S1": "response"}


def build_blind_packets(root: Path, batch: str, category: str,
                        output_dir: Path) -> dict:
    """Write both packets and an owner-only ID map after attestation passes."""
    if category not in KINDS:
        raise ValueError("Choose D1 or S1")
    root = root.resolve()
    target = output_dir.expanduser().resolve()
    if target.exists():
        raise FileExistsError("Review packet destination must not exist")
    rows = [row for row in load_approved(root, [batch])
            if row["source_kind"] == KINDS[category]]
    if len(rows) < 2:
        raise ValueError("Need at least two attested abstract cards for blinded ordering")
    languages = {row["language"] for row in rows}
    if len(languages) != 1:
        raise ValueError("A reviewer packet must contain exactly one language")
    language = languages.pop()
    ids = ["item_" + secrets.token_hex(8) for _ in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Opaque item ID collision")
    records = [
        {"item_id": opaque, "language": language,
         "category": category, "source_kind": row["source_kind"],
         "abstract_card": row["text"]}
        for opaque, row in zip(ids, rows)
    ]
    rng = secrets.SystemRandom()
    left, right = records.copy(), records.copy()
    rng.shuffle(left)
    rng.shuffle(right)
    if [item["item_id"] for item in left] == [item["item_id"] for item in right]:
        right[0], right[1] = right[1], right[0]
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".roguard-review-", dir=target.parent))
    try:
        for name, packet in (("reviewer-a.jsonl", left), ("reviewer-b.jsonl", right)):
            (temporary / name).write_text(
                "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n"
                        for item in packet), encoding="utf-8")
        owner = {
            "schema_version": 1, "batch": batch, "category": category,
            "language": language,
            "batch_sha256": sha256(root / f"data/synthetic/{batch}.jsonl"),
            "attestation_sha256": sha256(root / f"data/synthetic/{batch}.review.json"),
            "taxonomy_sha256": taxonomy_sha256(root, language),
            "packet_sha256": {
                name: sha256(temporary / name)
                for name in ("reviewer-a.jsonl", "reviewer-b.jsonl")},
            "id_map": [{"item_id": opaque, "source_id": row["id"]}
                       for opaque, row in zip(ids, rows)],
            "reviewer_independence_verified": False,
            "label_validity_verified": False,
        }
        (temporary / "owner-map.json").write_text(
            json.dumps(owner, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, target)
    except BaseException:
        for path in temporary.iterdir():
            path.unlink()
        temporary.rmdir()
        raise
    return {"output_dir": str(target), "language": language,
            "category": category, "cards": len(rows),
            "packet_a": "reviewer-a.jsonl", "packet_b": "reviewer-b.jsonl",
            "owner_map": "owner-map.json", "reference_labels_in_packets": False,
            "reviewer_independence_verified": False}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write two local blinded packets from one attested abstract batch")
    parser.add_argument("--batch", required=True)
    parser.add_argument("--category", choices=tuple(KINDS), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build_blind_packets(ROOT, args.batch, args.category, args.output_dir)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
