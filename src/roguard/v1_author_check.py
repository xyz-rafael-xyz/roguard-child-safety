"""Check one completed v1 author workbook against its issued blank packet."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import stat
from datetime import date
from pathlib import Path
from zipfile import ZipFile

from .v1_study_packet import _card

MAX_JSON_BYTES = 1024 * 1024
EDITABLE = frozenset({
    "native_language_confirmed_by_author",
    "independent_of_model_work_confirmed_by_author",
    "authored_at", "pairs",
})


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("Duplicate JSON key in author workbook")
        value[key] = item
    return value


def _load_json(raw: bytes) -> dict:
    if len(raw) > MAX_JSON_BYTES:
        raise ValueError("Author workbook exceeds 1 MiB")
    value = json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def check_v1_author_workbook(issued_packet: Path, workbook: Path) -> dict:
    """Validate format, frozen assignment, date, and basic card shape only."""
    if issued_packet.is_symlink() or not issued_packet.is_file():
        raise ValueError("Issued packet must be a regular ZIP file")
    if workbook.is_symlink() or not workbook.is_file():
        raise ValueError("Completed workbook must be a regular JSON file")
    if issued_packet.stat().st_size > MAX_JSON_BYTES or workbook.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("Issued packet or workbook exceeds 1 MiB")
    completed = _load_json(workbook.read_bytes())
    role = completed.get("author_id")
    language = completed.get("language")
    category = completed.get("category")
    if (language not in ("ro", "uk") or category not in ("D1", "S1") or
            role not in {f"{language.upper()}-A1", f"{language.upper()}-A2"} or
            issued_packet.name != f"{role}-v1-author-packet.zip"):
        raise ValueError("Workbook role, language, category, or packet name is wrong")
    expected_names = {
        "START-HERE.txt", f"taxonomy/taxonomy_{language}_v1_candidate.md",
        *(f"{part}/{role}.{extension}" for part in ("D1", "S1")
          for extension in ("json", "txt")),
    }
    with ZipFile(issued_packet) as archive:
        infos = archive.infolist()
        if (len(infos) != len(expected_names) or
                {info.filename for info in infos} != expected_names or
                any(info.is_dir() or stat.S_ISLNK(info.external_attr >> 16)
                    or info.file_size > MAX_JSON_BYTES for info in infos)):
            raise ValueError("Issued packet has an unexpected file layout")
        template = _load_json(archive.read(f"{category}/{role}.json"))
        taxonomy_name = f"taxonomy/taxonomy_{language}_v1_candidate.md"
        taxonomy_digest = hashlib.sha256(
            Path(taxonomy_name).name.encode() + b"\0" + archive.read(taxonomy_name) + b"\0"
        ).hexdigest()
        packet_date = max(date(*info.date_time[:3]) for info in infos)
    if (set(completed) != set(template) or
            any(completed[key] != template[key] for key in template if key not in EDITABLE) or
            completed["candidate_taxonomy_sha256"] != taxonomy_digest or
            completed["native_language_confirmed_by_author"] is not True or
            completed["independent_of_model_work_confirmed_by_author"] is not True):
        raise ValueError("Workbook metadata differs from the issued assignment")
    authored_at = completed["authored_at"]
    if not isinstance(authored_at, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", authored_at):
        raise ValueError("authored_at must be an ISO date")
    if date.fromisoformat(authored_at) < packet_date:
        raise ValueError("Declared author date predates the issued packet")
    pairs = completed["pairs"]
    if not isinstance(pairs, list) or len(pairs) != 24:
        raise ValueError("Expected the 24 issued pairs")
    seen = set()
    for actual, original in zip(pairs, template["pairs"]):
        if (not isinstance(actual, dict) or set(actual) != set(original) or
                any(actual[key] != original[key] for key in ("pair_index", "factor"))):
            raise ValueError("Pair assignment differs from the issued packet")
        for key in ("negative_abstract_card", "positive_abstract_card"):
            text = _card(actual[key], language)
            normalized = " ".join(text.casefold().split())
            if normalized in seen:
                raise ValueError("Duplicate abstract card in author workbook")
            seen.add(normalized)
    return {
        "workbook_compatible": True,
        "author_role": role,
        "language": language,
        "category": category,
        "pairs": 24,
        "cards": 48,
        "issued_packet_sha256": hashlib.sha256(issued_packet.read_bytes()).hexdigest(),
        "card_text_in_report": False,
        "semantic_one_fact_validity_verified": False,
        "actual_authoring_chronology_verified": False,
        "author_identity_verified": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a completed v1 workbook against its issued author ZIP")
    parser.add_argument("issued_packet", type=Path)
    parser.add_argument("workbook", type=Path)
    args = parser.parse_args()
    try:
        result = check_v1_author_workbook(args.issued_packet, args.workbook)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
