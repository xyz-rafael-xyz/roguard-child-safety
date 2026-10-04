"""Exact-byte review gate for corrected v1 candidate taxonomies."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

CODES = frozenset({"D1", "R1", "A1", "P1", "G1", "S1"})


def candidate_digest(root: Path, language: str) -> str:
    if language not in {"ro", "uk"}:
        raise ValueError("Unsupported language")
    name = f"taxonomy_{language}_v1_candidate.md"
    path = root / "taxonomy" / name
    if path.is_symlink() or not path.is_file():
        raise ValueError("Candidate taxonomy must be a regular file")
    data = path.read_bytes()
    return hashlib.sha256(name.encode() + b"\0" + data + b"\0").hexdigest()


def audit_candidate_reviews(root: Path) -> dict:
    """Check record structure and digest; human identities remain unverified."""
    result = {}
    all_reviewer_ids = []
    for language in ("ro", "uk"):
        expected = candidate_digest(root, language)
        path = root / "taxonomy" / f"review_{language}_v1_candidate.json"
        if path.is_symlink() or not path.is_file():
            raise ValueError("Candidate review must be a regular file")
        record = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(record, dict) or record.get("taxonomy_sha256") != expected:
            result[language] = {"status": "invalid_digest", "taxonomy_sha256": expected}
            continue
        if record.get("status") != "approved":
            result[language] = {"status": "pending", "taxonomy_sha256": expected}
            continue
        reviews = record.get("reviews")
        attestation = record.get("owner_attestation")
        valid = (isinstance(reviews, list) and len(reviews) == 2 and
                 all(isinstance(item, dict) and isinstance(item.get("reviewer"), str) and
                     isinstance(item.get("kind"), str) for item in reviews) and
                 isinstance(attestation, dict) and
                 attestation.get("four_distinct_reviewers_identity_and_independence_verified_by_owner") is True and
                 attestation.get("names_and_contacts_withheld_from_git") is True and
                 isinstance(attestation.get("attested_in_conversation_on"), str))
        if valid:
            try:
                date.fromisoformat(attestation["attested_in_conversation_on"])
            except ValueError:
                valid = False
        if valid:
            valid = ({item.get("kind") for item in reviews} == {"language", "child_safety"} and
                     len({item.get("reviewer") for item in reviews}) == 2)
        if valid:
            for item in reviews:
                decisions = item.get("category_decisions")
                valid = (isinstance(item.get("reviewer"), str) and bool(item["reviewer"].strip()) and
                         item.get("independent_of_author") is True and
                         isinstance(item.get("professional_role"), str) and
                         bool(item["professional_role"].strip()) and
                         isinstance(item.get("summary"), str) and bool(item["summary"].strip()) and
                         isinstance(item.get("reviewed_at"), str) and
                         re.fullmatch(r"\d{4}-\d{2}-\d{2}", item["reviewed_at"]) is not None and
                         isinstance(item.get("source_document_sha256"), str) and
                         re.fullmatch(r"[0-9a-f]{64}", item["source_document_sha256"]) is not None and
                         item.get("name_and_contact_public") is False and
                         isinstance(decisions, dict) and set(decisions) == CODES and
                         all(decision == "accept" for decision in decisions.values()))
                if not valid:
                    break
                try:
                    date.fromisoformat(item["reviewed_at"])
                except ValueError:
                    valid = False
                    break
        if valid:
            valid = len({item["source_document_sha256"] for item in reviews}) == 1
        if valid:
            all_reviewer_ids.extend(item["reviewer"] for item in reviews)
        result[language] = {"status": "record_structure_verified" if valid else "invalid_review",
                            "taxonomy_sha256": expected}
    if len(all_reviewer_ids) == 4 and len(set(all_reviewer_ids)) != 4:
        for language in ("ro", "uk"):
            result[language]["status"] = "duplicate_reviewer_id"
    return {"schema_version": 1, "taxonomy": result,
            "machine_taxonomy_gate_passed": all(
                item["status"] == "record_structure_verified" for item in result.values()),
            "reviewer_identity_and_independence_verified_by_software": False,
            "model_accuracy_established": False,
            "operational_child_safety_release_authorized": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exact-byte v1 candidate taxonomy reviews")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    try:
        print(json.dumps(audit_candidate_reviews(args.root), ensure_ascii=False, indent=2))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
