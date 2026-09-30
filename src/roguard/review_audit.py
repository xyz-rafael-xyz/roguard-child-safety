"""Bind two content-free annotation streams to exact blinded abstract packets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .agreement_batch import _read_jsonl, compare_evidence_batches
from .blind_packets import KINDS
from .packet_binding import binding_path, read_binding
from .review import INDEPENDENT_ORIGIN, load_approved, sha256, taxonomy_sha256


def _read_packet(path: Path) -> list[dict]:
    rows = _read_jsonl(path)
    if any(not isinstance(row, dict) or set(row) != {
            "item_id", "language", "category", "source_kind", "abstract_card"}
           for row in rows):
        raise ValueError("Reviewer packet has invalid fields")
    return rows


def audit_review_session(root: Path, packet_dir: Path,
                         left_answers: Path, right_answers: Path) -> dict:
    """Verify packet and answer alignment, then report agreement without content."""
    root, packet_dir = root.resolve(), packet_dir.resolve()
    owner_path = packet_dir / "owner-map.json"
    owner_digest = sha256(owner_path)
    owner = json.loads(owner_path.read_text(encoding="utf-8"))
    if (not isinstance(owner, dict) or set(owner) != {
            "schema_version", "batch", "category", "language", "batch_sha256",
            "attestation_sha256", "taxonomy_sha256", "packet_sha256", "id_map",
            "reviewer_independence_verified", "label_validity_verified"} or
            owner["schema_version"] != 1 or owner["category"] not in KINDS or
            owner["reviewer_independence_verified"] is not False or
            owner["label_validity_verified"] is not False):
        raise ValueError("Owner map has invalid review metadata")
    batch, category, language = owner["batch"], owner["category"], owner["language"]
    rows = [row for row in load_approved(root, [batch])
            if row["source_kind"] == KINDS[category]]
    independent = all(row["origin"] == INDEPENDENT_ORIGIN for row in rows)
    if (len(rows) < 2 or {row["language"] for row in rows} != {language} or
            owner["batch_sha256"] != sha256(root / f"data/synthetic/{batch}.jsonl") or
            owner["attestation_sha256"] != sha256(root / f"data/synthetic/{batch}.review.json") or
            owner["taxonomy_sha256"] != taxonomy_sha256(root, language) or
            not isinstance(owner["id_map"], list) or len(owner["id_map"]) != len(rows) or
            not isinstance(owner["packet_sha256"], dict) or
            set(owner["packet_sha256"]) != {"reviewer-a.jsonl", "reviewer-b.jsonl"}):
        raise ValueError("Owner map differs from attested abstract source")
    mapping: dict[str, dict] = {}
    for entry, row in zip(owner["id_map"], rows):
        if (not isinstance(entry, dict) or set(entry) != {"item_id", "source_id"} or
                not isinstance(entry["item_id"], str) or
                entry["source_id"] != row["id"] or entry["item_id"] in mapping):
            raise ValueError("Owner map does not align opaque IDs to source rows")
        mapping[entry["item_id"]] = row
    packets = []
    for name in ("reviewer-a.jsonl", "reviewer-b.jsonl"):
        path = packet_dir / name
        if owner["packet_sha256"][name] != sha256(path):
            raise ValueError("Reviewer packet bytes differ from owner map")
        packet = _read_packet(path)
        if any(not isinstance(item["item_id"], str) for item in packet):
            raise ValueError("Reviewer packet has invalid opaque item IDs")
        ids = [item["item_id"] for item in packet]
        if len(ids) != len(rows) or len(set(ids)) != len(ids) or set(ids) != set(mapping):
            raise ValueError("Reviewer packet item IDs differ from owner map")
        for item in packet:
            source = mapping[item["item_id"]]
            if (item["language"] != language or item["category"] != category or
                    item["source_kind"] != source["source_kind"] or
                    item["abstract_card"] != source["text"]):
                raise ValueError("Reviewer packet differs from attested abstract card")
        packets.append(ids)
    if packets[0] == packets[1]:
        raise ValueError("Reviewer packets need distinct orders")
    answer_digests = (sha256(left_answers), sha256(right_answers))
    left = _read_jsonl(left_answers)
    right = _read_jsonl(right_answers)
    bindings = [read_binding(path, owner["packet_sha256"][name])
                for path, name in ((left_answers, "reviewer-a.jsonl"),
                                   (right_answers, "reviewer-b.jsonl"))]
    if independent and not all(bindings):
        raise ValueError("Independent review requires both private packet bindings")
    if any(bindings) and not all(bindings):
        raise ValueError("Review streams have incomplete packet bindings")
    binding_digests = ((sha256(binding_path(left_answers)),
                        sha256(binding_path(right_answers))) if all(bindings) else None)
    expected_card = {"language", "disclosure" if category == "D1" else "support"}
    for stream, packet_ids in zip((left, right), packets):
        if (len(stream) != len(rows) or any(
                not isinstance(item, dict) or set(item) != {"item_id", "card"} or
                not isinstance(item["item_id"], str) or
                item["item_id"] not in mapping or not isinstance(item["card"], dict) or
                set(item["card"]) != expected_card or item["card"]["language"] != language
                for item in stream)):
            raise ValueError("Answer stream differs from blinded packet IDs or category")
        if [item["item_id"] for item in stream] != packet_ids:
            raise ValueError("Answer stream order differs from its assigned blinded packet")
    agreement = compare_evidence_batches(left, right)
    if (sha256(owner_path) != owner_digest or
            sha256(root / f"data/synthetic/{batch}.jsonl") != owner["batch_sha256"] or
            sha256(root / f"data/synthetic/{batch}.review.json") != owner["attestation_sha256"] or
            taxonomy_sha256(root, language) != owner["taxonomy_sha256"] or
            any(sha256(packet_dir / name) != owner["packet_sha256"][name]
                for name in ("reviewer-a.jsonl", "reviewer-b.jsonl")) or
            (sha256(left_answers), sha256(right_answers)) != answer_digests or
            (binding_digests is not None and
             (sha256(binding_path(left_answers)),
              sha256(binding_path(right_answers))) != binding_digests) or
            (all(bindings) and not all(read_binding(path, owner["packet_sha256"][name])
                                       for path, name in ((left_answers, "reviewer-a.jsonl"),
                                                          (right_answers, "reviewer-b.jsonl"))))):
        raise ValueError("Review input changed during audit")
    return {
        "schema_version": 1, "batch": batch, "category": category,
        "language": language, "rows": len(rows),
        "packet_integrity_verified": True,
        "source_alignment_verified": True,
        "answer_id_alignment_verified": True,
        "annotation_packet_binding_verified": all(bindings),
        "reviewer_independence_verified": False,
        "label_validity_verified": False,
        "agreement": agreement,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit local blinded packet and D1/S1 answer alignment")
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("left_answers", type=Path)
    parser.add_argument("right_answers", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    try:
        result = audit_review_session(root, args.packet_dir,
                                      args.left_answers, args.right_answers)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
