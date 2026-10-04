"""Private v1 abstract-card intake and blinded packet construction."""

from __future__ import annotations

import argparse
import json
import re
import secrets
from datetime import date
from pathlib import Path

from .author_kit import _private_destination, _write_private_dir
from .intake_overlap import audit_prior_overlap
from .review import check_independent_card_script, sha256
from .v1_study_kit import FACTORS, verify_v1_study_freeze

ROOT = Path(__file__).resolve().parents[2]
NO_CARD_MARKS = ('"', "“", "”", "|", "\n", "\r")
CODE_TOKEN = re.compile(r"\b(?:D1|R1|A1|P1|G1|S1)\b")


def _load(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Expected a regular private JSON file")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _date(value: object) -> date:
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("Expected an ISO date")
    return date.fromisoformat(value)


def _card(text: object, language: str) -> str:
    if (not isinstance(text, str) or not 20 <= len(text) <= 1000 or
            text != text.strip() or any(mark in text for mark in NO_CARD_MARKS) or
            CODE_TOKEN.search(text)):
        raise ValueError("Card must be a single abstract description without quotes or labels")
    check_independent_card_script(text, language)
    return text


def _validate_workbooks(root: Path, kit: Path) -> tuple[dict, list[dict], dict]:
    owner_path = kit / "OWNER-ASSIGNMENTS.json"
    owner = _load(owner_path)
    if (set(owner) != {"schema_version", "language", "category",
                       "candidate_taxonomy_sha256", "freeze_path", "freeze_sha256",
                       "assignments", "author_ids_are_owner_supplied_opaque_roles",
                       "no_author_content_generated"} or
            owner["schema_version"] != 1 or owner["language"] not in ("ro", "uk") or
            owner["category"] not in FACTORS or
            owner["author_ids_are_owner_supplied_opaque_roles"] is not True or
            owner["no_author_content_generated"] is not True or
            not isinstance(owner["assignments"], dict)):
        raise ValueError("Owner assignments differ from the v1 kit")
    language, category = owner["language"], owner["category"]
    frozen = verify_v1_study_freeze(root, Path(owner["freeze_path"]), language, category)
    if (owner["freeze_sha256"] != frozen["sha256"] or
            owner["candidate_taxonomy_sha256"] != frozen["candidate_taxonomy_sha256"]):
        raise ValueError("Author kit differs from the candidate/model freeze")
    expected_authors = {f"{language.upper()}-A1", f"{language.upper()}-A2"}
    if set(owner["assignments"]) != expected_authors:
        raise ValueError("Author assignments need two distinct owner roles")
    rows, authored_dates, workbook_hashes = [], [], {}
    seen_text = set()
    for author_id in sorted(expected_authors):
        path = kit / f"{author_id}.json"
        book = _load(path)
        required = {"schema_version", "study_kind", "language", "category", "author_id",
                    "candidate_taxonomy_sha256", "freeze_path", "freeze_sha256",
                    "native_language_confirmed_by_author",
                    "independent_of_model_work_confirmed_by_author", "authored_at", "pairs"}
        if (set(book) != required or book["schema_version"] != 1 or
                book["study_kind"] != "roguard_v1_candidate_independent_abstract" or
                any(book[name] != owner[name] for name in (
                    "language", "category", "candidate_taxonomy_sha256",
                    "freeze_path", "freeze_sha256")) or
                book["author_id"] != author_id or
                book["native_language_confirmed_by_author"] is not True or
                book["independent_of_model_work_confirmed_by_author"] is not True or
                not isinstance(book["pairs"], list) or len(book["pairs"]) != 24 or
                not all(isinstance(pair, dict) for pair in book["pairs"]) or
                [pair.get("factor") for pair in book["pairs"]] != owner["assignments"][author_id]):
            raise ValueError("Author workbook differs from the frozen assignment")
        authored_dates.append(_date(book["authored_at"]))
        workbook_hashes[author_id] = sha256(path)
        for index, pair in enumerate(book["pairs"], 1):
            if (not isinstance(pair, dict) or set(pair) != {
                    "pair_index", "factor", "negative_abstract_card", "positive_abstract_card"} or
                    pair["pair_index"] != index or pair["factor"] not in FACTORS[category]):
                raise ValueError("Author changed the pair structure")
            negative = _card(pair["negative_abstract_card"], language)
            positive = _card(pair["positive_abstract_card"], language)
            for text in (negative, positive):
                normalized = " ".join(text.casefold().split())
                if normalized in seen_text:
                    raise ValueError("Abstract cards repeat exactly within the candidate")
                seen_text.add(normalized)
            rows.extend((
                {"language": language, "source_kind": "message" if category == "D1" else "response",
                 "author_id": author_id, "pair_index": index, "factor": pair["factor"],
                 "polarity": "negative", "text": negative},
                {"language": language, "source_kind": "message" if category == "D1" else "response",
                 "author_id": author_id, "pair_index": index, "factor": pair["factor"],
                 "polarity": "positive", "text": positive},
            ))
    return owner, rows, {"workbook_sha256": workbook_hashes,
                         "latest_authored_at": max(authored_dates).isoformat()}


def prepare_v1_content_review(root: Path, kit: Path, output_dir: Path) -> dict:
    """Assemble unapproved private cards and a content-review template."""
    root = root.resolve()
    kit = kit.expanduser().resolve()
    owner, rows, provenance = _validate_workbooks(root, kit)
    target = _private_destination(root, output_dir)
    overlap_rows = [{"language": row["language"], "source_kind": row["source_kind"],
                     "text": row["text"]} for row in rows]
    overlap = audit_prior_overlap(root, overlap_rows, "v1-candidate-private")
    cards = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    overlap_text = json.dumps(overlap, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    source = kit / "OWNER-ASSIGNMENTS.json"
    files = {"SOURCE-CARDS.jsonl": cards, "OVERLAP.json": overlap_text,
             "OWNER-ASSIGNMENTS.json": source.read_text(encoding="utf-8")}
    for author_id in sorted(provenance["workbook_sha256"]):
        files[f"{author_id}.json"] = (kit / f"{author_id}.json").read_text(encoding="utf-8")
    import hashlib
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    template = {
        "schema_version": 1, "language": owner["language"],
        "category": owner["category"],
        "reviewer_id": f"{owner['language'].upper()}-C",
        "reviewed_at": None,
        "source_cards_sha256": digest(cards),
        "overlap_sha256": digest(overlap_text),
        "owner_assignments_sha256": sha256(source),
        "author_workbook_sha256": provenance["workbook_sha256"],
        "abstract_only_checked": False,
        "no_realistic_content_checked": False,
        "language_checked": False,
        "one_fact_pair_logic_checked": False,
        "factor_balance_checked": False,
        "prior_and_cross_pair_overlap_checked": False,
    }
    files["CONTENT-REVIEW-TEMPLATE.json"] = json.dumps(template, ensure_ascii=False, indent=2) + "\n"
    _write_private_dir(target, files)
    return {"output_dir": str(target), "language": owner["language"],
            "category": owner["category"], "cards": 96, "pairs": 48,
            "overlap_warning_cards": overlap["near_overlap_cards"],
            "cross_pair_warning_cards": overlap["within_candidate_cross_pair"]["near_overlap_cards"],
            "approved_for_blind_packets": False}


def seal_v1_packets(root: Path, candidate_dir: Path, review_declaration: Path,
                    output_dir: Path) -> dict:
    """Create blind packets only after a distinct content reviewer signs hashes."""
    root = root.resolve()
    candidate_dir = candidate_dir.expanduser().resolve()
    owner, rows, provenance = _validate_workbooks(root, candidate_dir)
    declaration = _load(review_declaration)
    template = _load(candidate_dir / "CONTENT-REVIEW-TEMPLATE.json")
    required_flags = (
        "abstract_only_checked", "no_realistic_content_checked", "language_checked",
        "one_fact_pair_logic_checked", "factor_balance_checked",
        "prior_and_cross_pair_overlap_checked",
    )
    if (set(declaration) != set(template) or
            any(declaration[key] != template[key] for key in template
                if key not in {"reviewed_at", *required_flags}) or
            any(declaration[key] is not True for key in required_flags) or
            _date(declaration["reviewed_at"]) < _date(provenance["latest_authored_at"]) or
            declaration["reviewer_id"] in provenance["workbook_sha256"] or
            declaration["source_cards_sha256"] != sha256(candidate_dir / "SOURCE-CARDS.jsonl") or
            declaration["overlap_sha256"] != sha256(candidate_dir / "OVERLAP.json") or
            declaration["owner_assignments_sha256"] != sha256(candidate_dir / "OWNER-ASSIGNMENTS.json") or
            declaration["author_workbook_sha256"] != provenance["workbook_sha256"]):
        raise ValueError("Content-review declaration does not approve exact candidate bytes")
    expected_rows = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if (candidate_dir / "SOURCE-CARDS.jsonl").read_text(encoding="utf-8") != expected_rows:
        raise ValueError("Source cards changed since workbook validation")
    target = _private_destination(root, output_dir)
    ids = ["item_" + secrets.token_hex(8) for _ in rows]
    if len(set(ids)) != 96:
        raise ValueError("Opaque item ID collision")
    language, category = owner["language"], owner["category"]
    packets = [
        {"item_id": item_id, "language": language, "category": category,
         "source_kind": row["source_kind"], "abstract_card": row["text"]}
        for item_id, row in zip(ids, rows)
    ]
    rng = secrets.SystemRandom()
    left, right = packets.copy(), packets.copy()
    rng.shuffle(left)
    rng.shuffle(right)
    if [row["item_id"] for row in left] == [row["item_id"] for row in right]:
        right[0], right[1] = right[1], right[0]
    packet_text = {
        "reviewer-a.jsonl": "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in left),
        "reviewer-b.jsonl": "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in right),
    }
    import hashlib
    digest = lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()
    pair_map = []
    for index in range(0, 96, 2):
        row = rows[index]
        pair_map.append({"author_id": row["author_id"], "factor": row["factor"],
                         "negative_item_id": ids[index], "positive_item_id": ids[index + 1]})
    owner_map = {
        "schema_version": 1,
        "study_kind": "roguard_v1_candidate_independent_abstract",
        "language": language, "category": category,
        "candidate_taxonomy_sha256": owner["candidate_taxonomy_sha256"],
        "freeze_path": owner["freeze_path"], "freeze_sha256": owner["freeze_sha256"],
        "packet_sha256": {"reviewer-a": digest(packet_text["reviewer-a.jsonl"]),
                          "reviewer-b": digest(packet_text["reviewer-b.jsonl"])},
        "author_workbook_sha256": provenance["workbook_sha256"],
        "content_review_sha256": sha256(review_declaration),
        "pairs": pair_map,
    }
    files = {**packet_text, "OWNER-MAP.json": json.dumps(owner_map, ensure_ascii=False, indent=2) + "\n",
             "CONTENT-REVIEW.json": review_declaration.read_text(encoding="utf-8"),
             "OWNER-ASSIGNMENTS.json": (candidate_dir / "OWNER-ASSIGNMENTS.json").read_text(encoding="utf-8"),
             "SOURCE-CARDS.jsonl": expected_rows,
             "OVERLAP.json": (candidate_dir / "OVERLAP.json").read_text(encoding="utf-8")}
    for author_id in sorted(provenance["workbook_sha256"]):
        files[f"{author_id}.json"] = (candidate_dir / f"{author_id}.json").read_text(encoding="utf-8")
    _write_private_dir(target, files)
    return {"output_dir": str(target), "language": language, "category": category,
            "cards": 96, "pairs": 48, "reviewer_packets_have_labels": False,
            "owner_map_private": True, "ready_for_blind_prediction_and_annotation": True,
            "human_content_review_verified_by_software": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or seal private v1 abstract study packets")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--root", type=Path, default=ROOT)
    prepare.add_argument("--kit-dir", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    seal = sub.add_parser("seal")
    seal.add_argument("--root", type=Path, default=ROOT)
    seal.add_argument("--candidate-dir", type=Path, required=True)
    seal.add_argument("--review-declaration", type=Path, required=True)
    seal.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (prepare_v1_content_review(args.root, args.kit_dir, args.output_dir)
                  if args.command == "prepare" else
                  seal_v1_packets(args.root, args.candidate_dir,
                                  args.review_declaration, args.output_dir))
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
