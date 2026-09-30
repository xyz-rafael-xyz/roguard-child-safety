"""Offline workbooks for independently authored, abstract D1/S1 test pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import secrets
import shutil
import tempfile
from collections import Counter
from datetime import date
from pathlib import Path

from .review import (INDEPENDENT_FACTORS, INDEPENDENT_ORIGIN,
                     VALIDATED_GENERATORS, check_independent_card_script,
                     load_approved, sha256,
                     verify_independent_reviews)
from .intake_overlap import audit_prior_overlap
from .study_freeze import verify_study_freeze

ROOT = Path(__file__).resolve().parents[2]
BATCH_ID = re.compile(r"batch-[0-9]{4,8}\Z")
AUTHOR_ID = re.compile(r"[A-Za-z0-9_-]{3,40}\Z")
FORBIDDEN_CARD_MARKS = ('"', "“", "”", "|", "\n", "\r")


def _date(value: object) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _private_destination(root: Path, target: Path) -> Path:
    target = target.expanduser().resolve()
    if target.exists():
        raise FileExistsError("Author-kit destination must not exist")
    project = root.resolve()
    if target.is_relative_to(project) and not target.is_relative_to(project / "review_runs"):
        raise ValueError("Unreviewed workbooks and cards must stay outside Git or under review_runs/")
    return target


def _write_private_dir(target: Path, files: dict[str, str]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".roguard-author-", dir=target.parent))
    try:
        for name, content in files.items():
            descriptor = os.open(temporary / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(content)
        os.replace(temporary, target)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def _json(value: dict) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def new_author_kit(root: Path, batch: str, language: str, category: str,
                   output_dir: Path, *, freeze_record: Path) -> dict:
    """Create two separate blank, balanced tasks after taxonomy signoff."""
    root = root.resolve()
    if not BATCH_ID.fullmatch(batch) or batch in VALIDATED_GENERATORS or (
            root / f"data/synthetic/{batch}.jsonl").exists():
        raise ValueError("Choose an unused independent batch ID")
    if language not in ("ro", "uk") or category not in INDEPENDENT_FACTORS:
        raise ValueError("Choose Romanian or Ukrainian and D1 or S1")
    taxonomy_digest = verify_independent_reviews(root, language)
    model_freeze = verify_study_freeze(root, freeze_record, language, category)
    target = _private_destination(root, output_dir)
    factors = INDEPENDENT_FACTORS[category]
    files = {}
    assignments = {}
    for slot in ("a", "b"):
        author_id = "author_" + secrets.token_hex(6)
        assigned = list(factors) * (24 // len(factors))
        secrets.SystemRandom().shuffle(assigned)
        assignments[slot] = {"author_id": author_id, "factors": assigned}
        workbook = {
            "schema_version": 1, "batch": batch, "language": language,
            "category": category, "taxonomy_sha256": taxonomy_digest,
            "model_freeze_path": model_freeze["path"],
            "model_freeze_sha256": model_freeze["sha256"],
            "author_id": author_id, "native_language": None,
            "independent_of_project": None, "authored_at": None,
            "pairs": [{"pair_index": number, "factor": factor,
                       "negative_abstract_card": "", "positive_abstract_card": ""}
                      for number, factor in enumerate(assigned, 1)],
        }
        files[f"author-{slot}.json"] = _json(workbook)
    files["owner-assignments.json"] = _json({
        "schema_version": 1, "batch": batch, "language": language,
        "category": category, "taxonomy_sha256": taxonomy_digest,
        "model_freeze_path": model_freeze["path"],
        "model_freeze_sha256": model_freeze["sha256"],
        "assignments": assignments,
    })
    _write_private_dir(target, files)
    return {"output_dir": str(target), "batch": batch, "language": language,
            "category": category, "pairs_per_author": 24,
            "author_files": ["author-a.json", "author-b.json"],
            "owner_assignments": "owner-assignments.json",
            "human_authorship_or_independence_verified": False,
            "taxonomy_sha256": taxonomy_digest,
            "model_freeze_sha256": model_freeze["sha256"]}


def _read_workbook(path: Path, root: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Author workbook must be an ordinary file")
    book = json.loads(path.read_text(encoding="utf-8"))
    fields = {"schema_version", "batch", "language", "category", "taxonomy_sha256",
              "model_freeze_path", "model_freeze_sha256", "author_id",
              "native_language", "independent_of_project", "authored_at", "pairs"}
    if (not isinstance(book, dict) or set(book) != fields or book["schema_version"] != 1 or
            not isinstance(book["batch"], str) or not BATCH_ID.fullmatch(book["batch"]) or
            book["batch"] in VALIDATED_GENERATORS or
            book["language"] not in ("ro", "uk") or
            book["category"] not in INDEPENDENT_FACTORS or
            not isinstance(book["author_id"], str) or
            not AUTHOR_ID.fullmatch(book["author_id"]) or
            book["native_language"] != book["language"] or
            book["independent_of_project"] is not True or
            not _date(book["authored_at"]) or
            not isinstance(book["pairs"], list) or len(book["pairs"]) != 24):
        raise ValueError("Author workbook or personal declarations are incomplete")
    if book["taxonomy_sha256"] != verify_independent_reviews(root, book["language"]):
        raise ValueError("Author workbook differs from the reviewed taxonomy")
    if (not isinstance(book["model_freeze_path"], str) or
            not isinstance(book["model_freeze_sha256"], str) or
            verify_study_freeze(root, Path(book["model_freeze_path"]),
                                book["language"], book["category"])["sha256"] !=
            book["model_freeze_sha256"]):
        raise ValueError("Author workbook differs from the model freeze")
    factors = INDEPENDENT_FACTORS[book["category"]]
    seen = Counter()
    for number, pair in enumerate(book["pairs"], 1):
        if (not isinstance(pair, dict) or set(pair) != {
                "pair_index", "factor", "negative_abstract_card", "positive_abstract_card"} or
                type(pair["pair_index"]) is not int or pair["pair_index"] != number or
                pair["factor"] not in factors):
            raise ValueError("Author pair assignments differ from the balanced workbook")
        seen[pair["factor"]] += 1
        for key in ("negative_abstract_card", "positive_abstract_card"):
            card = pair[key]
            if (not isinstance(card, str) or len(card) < 20 or len(card) > 1000 or
                    card != card.strip() or any(mark in card for mark in FORBIDDEN_CARD_MARKS) or
                    re.search(r"\b(?:D1|R1|A1|P1|G1|S1)\b", card)):
                raise ValueError("Author card must be a single abstract description without quotes or labels")
            check_independent_card_script(card, book["language"])
        if pair["negative_abstract_card"].casefold() == pair["positive_abstract_card"].casefold():
            raise ValueError("A pair needs two different abstract descriptions")
    if any(seen[factor] != 24 // len(factors) for factor in factors):
        raise ValueError("Author workbook has unbalanced factor assignments")
    return book


def assemble_candidate(root: Path, author_a: Path, author_b: Path,
                       owner_assignments: Path, output_dir: Path) -> dict:
    """Assemble private review material; never approve it automatically."""
    root = root.resolve()
    if author_a.resolve() == author_b.resolve():
        raise ValueError("Supply two separate author workbooks")
    author_hashes = {"a": sha256(author_a), "b": sha256(author_b)}
    books = [_read_workbook(path, root) for path in (author_a, author_b)]
    a, b = books
    if owner_assignments.is_symlink() or not owner_assignments.is_file():
        raise ValueError("Owner assignment map must be an ordinary private file")
    assignment_digest = sha256(owner_assignments)
    owner = json.loads(owner_assignments.read_text(encoding="utf-8"))
    if (not isinstance(owner, dict) or set(owner) != {
            "schema_version", "batch", "language", "category",
            "taxonomy_sha256", "model_freeze_path",
            "model_freeze_sha256", "assignments"} or
            owner["schema_version"] != 1 or
            any(owner[key] != a[key] for key in (
                "batch", "language", "category", "taxonomy_sha256",
                "model_freeze_path", "model_freeze_sha256")) or
            not isinstance(owner["assignments"], dict) or
            set(owner["assignments"]) != {"a", "b"}):
        raise ValueError("Owner assignment map differs from author tasks")
    for slot, book in zip(("a", "b"), books):
        assigned = owner["assignments"][slot]
        if (not isinstance(assigned, dict) or set(assigned) != {"author_id", "factors"} or
                assigned["author_id"] != book["author_id"] or
                assigned["factors"] != [pair["factor"] for pair in book["pairs"]]):
            raise ValueError("Author changed an assigned factor or task order")
    if (any(a[key] != b[key] for key in ("batch", "language", "category", "taxonomy_sha256",
                                       "model_freeze_path", "model_freeze_sha256")) or
            a["author_id"] == b["author_id"] or
            (root / f"data/synthetic/{a['batch']}.jsonl").exists()):
        raise ValueError("Author workbooks need one unused batch and two distinct authors")
    if ({"a": sha256(author_a), "b": sha256(author_b)} != author_hashes or
            sha256(owner_assignments) != assignment_digest):
        raise ValueError("Author workbook changed during assembly")
    target = _private_destination(root, output_dir)
    batch, language, category = a["batch"], a["language"], a["category"]
    kind = "message" if category == "D1" else "response"
    rows = []
    pair_factors = []
    pair_authors = []
    for book in books:
        for pair in book["pairs"]:
            pair_factors.append(pair["factor"])
            pair_authors.append(book["author_id"])
            for key, labels in (("negative_abstract_card", []),
                                ("positive_abstract_card", [category])):
                rows.append({
                    "id": f"{batch}-{language}-{len(rows) + 1:05d}",
                    "language": language, "source_kind": kind, "split": "test",
                    "text": pair[key], "labels": labels, "origin": INDEPENDENT_ORIGIN,
                })
    if len({" ".join(row["text"].casefold().split()) for row in rows}) != 96:
        raise ValueError("The two author workbooks repeat abstract card text")
    overlap_audit = audit_prior_overlap(root, rows, batch)
    stem = batch
    data = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    preview = ("# Abstract candidate awaiting separate content review\n\n"
               "| ID | Limbă | Tip | Etichete | Text |\n"
               "|---|---|---|---|---|\n" +
               "".join(f"| {row['id']} | {language} | {kind} | "
                       f"{','.join(row['labels']) or 'none'} | {row['text']} |\n"
                       for row in rows))
    provenance = {
        "schema_version": 2, "batch": batch, "language": language,
        "category": category,
        "model_freeze_path": a["model_freeze_path"],
        "model_freeze_sha256": a["model_freeze_sha256"],
        "authors": [{"author_id": book["author_id"],
                     "native_language": book["native_language"],
                     "independent_of_project": book["independent_of_project"],
                     "authored_at": book["authored_at"]} for book in books],
        "pair_authors": pair_authors, "pair_factors": pair_factors,
        "safety_review": {"reviewer_id": "", "reviewed_at": "",
                          "abstract_only_checked": False,
                          "no_realistic_content_checked": False,
                          "pair_logic_checked": False,
                          "factor_balance_checked": False,
                          "prior_overlap_checked": False},
        "label_status": "author_intended_not_adjudicated",
    }
    provenance_text = _json(provenance)
    review = {
        "status": "pending", "reviewer": "", "reviewed_at": "",
        "reviewed_ids": [], "language": language,
        "batch_sha256": _text_sha256(data),
        "preview_sha256": _text_sha256(preview),
        "taxonomy_sha256": a["taxonomy_sha256"],
        "provenance_sha256": _text_sha256(provenance_text),
    }
    files = {f"{stem}.jsonl": data, f"{stem}.preview.md": preview,
             f"{stem}.provenance.json": provenance_text,
             f"{stem}.review.json": _json(review)}
    manifest = {"schema_version": 1, "status": "awaiting_separate_content_review",
                "batch": batch, "language": language, "category": category,
                "taxonomy_sha256": a["taxonomy_sha256"],
                "model_freeze_path": a["model_freeze_path"],
                "model_freeze_sha256": a["model_freeze_sha256"],
                "prior_overlap_audit": overlap_audit,
                "author_workbook_sha256": author_hashes,
                "owner_assignments_sha256": assignment_digest,
                "candidate_sha256": {name: _text_sha256(content)
                                     for name, content in files.items()},
                "actual_author_independence_verified_by_software": False,
                "content_safety_verified_by_software": False}
    files["assembly-manifest.json"] = _json(manifest)
    files["content-review-template.json"] = _json({
        "schema_version": 1, "batch": batch, "language": language,
        "category": category, "reviewer_id": "", "reviewed_at": "",
        "candidate_manifest_sha256": _text_sha256(files["assembly-manifest.json"]),
        "reviewed_batch_sha256": manifest["candidate_sha256"][f"{batch}.jsonl"],
        "reviewed_preview_sha256": manifest["candidate_sha256"][f"{batch}.preview.md"],
        "abstract_only_checked": False,
        "no_realistic_content_checked": False,
        "pair_logic_checked": False,
        "factor_balance_checked": False,
        "prior_overlap_checked": False,
    })
    _write_private_dir(target, files)
    return {"output_dir": str(target), "batch": batch, "language": language,
            "category": category, "cards": len(rows), "pairs": len(rows) // 2,
            "status": "awaiting_separate_content_review",
            "approved_for_repository": False}


def seal_candidate(root: Path, candidate_dir: Path,
                   review_declaration: Path) -> dict:
    """Place reviewed abstract cards in Git only after a separate declaration."""
    root, candidate_dir = root.resolve(), candidate_dir.expanduser().resolve()
    if review_declaration.is_symlink() or not review_declaration.is_file():
        raise ValueError("Content-review declaration must be an ordinary file")
    manifest_path = candidate_dir / "assembly-manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Candidate assembly manifest is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (not isinstance(manifest, dict) or set(manifest) != {
            "schema_version", "status", "batch", "language", "category",
            "taxonomy_sha256", "model_freeze_path", "model_freeze_sha256",
            "author_workbook_sha256", "owner_assignments_sha256",
            "candidate_sha256", "prior_overlap_audit",
            "actual_author_independence_verified_by_software",
            "content_safety_verified_by_software"} or
            manifest["schema_version"] != 1 or
            manifest["status"] != "awaiting_separate_content_review" or
            manifest["actual_author_independence_verified_by_software"] is not False or
            manifest["content_safety_verified_by_software"] is not False or
            not isinstance(manifest["batch"], str) or
            not BATCH_ID.fullmatch(manifest["batch"]) or
            manifest["batch"] in VALIDATED_GENERATORS or
            manifest["language"] not in ("ro", "uk") or
            manifest["category"] not in INDEPENDENT_FACTORS or
            not isinstance(manifest["author_workbook_sha256"], dict) or
            set(manifest["author_workbook_sha256"]) != {"a", "b"} or
            not isinstance(manifest["owner_assignments_sha256"], str) or
            not re.fullmatch(r"[0-9a-f]{64}", manifest["owner_assignments_sha256"]) or
            not isinstance(manifest["candidate_sha256"], dict)):
        raise ValueError("Candidate assembly manifest is invalid")
    batch, language, category = (manifest[key] for key in ("batch", "language", "category"))
    names = {f"{batch}.jsonl", f"{batch}.preview.md",
             f"{batch}.provenance.json", f"{batch}.review.json"}
    if set(manifest["candidate_sha256"]) != names:
        raise ValueError("Candidate assembly manifest omits a required file")
    for name in names:
        path = candidate_dir / name
        if path.is_symlink() or not path.is_file() or sha256(path) != manifest["candidate_sha256"][name]:
            raise ValueError("Candidate changed after assembly")
    if manifest["taxonomy_sha256"] != verify_independent_reviews(root, language):
        raise ValueError("Candidate differs from the reviewed taxonomy")
    if (not isinstance(manifest["model_freeze_path"], str) or
            not isinstance(manifest["model_freeze_sha256"], str) or
            verify_study_freeze(root, Path(manifest["model_freeze_path"]),
                                language, category)["sha256"] != manifest["model_freeze_sha256"]):
        raise ValueError("Candidate differs from the model freeze")
    declaration = json.loads(review_declaration.read_text(encoding="utf-8"))
    required = {"schema_version", "batch", "language", "category", "reviewer_id",
                "reviewed_at", "candidate_manifest_sha256", "reviewed_batch_sha256",
                "reviewed_preview_sha256", "abstract_only_checked",
                "no_realistic_content_checked", "pair_logic_checked",
                "factor_balance_checked", "prior_overlap_checked"}
    if (not isinstance(declaration, dict) or set(declaration) != required or
            declaration["schema_version"] != 1 or
            any(declaration[key] != manifest[key] for key in ("batch", "language", "category")) or
            not isinstance(declaration["reviewer_id"], str) or
            not AUTHOR_ID.fullmatch(declaration["reviewer_id"]) or
            not _date(declaration["reviewed_at"]) or
            declaration["candidate_manifest_sha256"] != sha256(manifest_path) or
            declaration["reviewed_batch_sha256"] != manifest["candidate_sha256"][f"{batch}.jsonl"] or
            declaration["reviewed_preview_sha256"] != manifest["candidate_sha256"][f"{batch}.preview.md"] or
            any(declaration[key] is not True for key in (
                "abstract_only_checked", "no_realistic_content_checked",
                "pair_logic_checked", "factor_balance_checked",
                "prior_overlap_checked"))):
        raise ValueError("Separate content-review declaration is incomplete or differs")
    provenance = json.loads((candidate_dir / f"{batch}.provenance.json").read_text(encoding="utf-8"))
    pending = json.loads((candidate_dir / f"{batch}.review.json").read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in
            (candidate_dir / f"{batch}.jsonl").read_text(encoding="utf-8").splitlines()]
    if manifest["prior_overlap_audit"] != audit_prior_overlap(root, rows, batch):
        raise ValueError("Prior-card overlap audit changed after assembly")
    if (provenance.get("model_freeze_path") != manifest["model_freeze_path"] or
            provenance.get("model_freeze_sha256") != manifest["model_freeze_sha256"]):
        raise ValueError("Candidate provenance differs from the model freeze")
    authors = provenance.get("authors") if isinstance(provenance, dict) else None
    if (not isinstance(authors, list) or len(authors) != 2 or
            any(not isinstance(author, dict) or
                author.get("author_id") == declaration["reviewer_id"] or
                not _date(author.get("authored_at")) or
                author["authored_at"] > declaration["reviewed_at"]
                for author in authors) or
            not isinstance(pending, dict) or pending.get("status") != "pending" or
            pending.get("batch_sha256") != declaration["reviewed_batch_sha256"] or
            pending.get("preview_sha256") != declaration["reviewed_preview_sha256"] or
            len(rows) != 96 or any(not isinstance(row, dict) or "id" not in row for row in rows)):
        raise ValueError("Candidate authors, rows, or pending review differ")
    provenance["safety_review"] = {
        "reviewer_id": declaration["reviewer_id"],
        "reviewed_at": declaration["reviewed_at"],
        **{key: declaration[key] for key in (
            "abstract_only_checked", "no_realistic_content_checked",
            "pair_logic_checked", "factor_balance_checked",
            "prior_overlap_checked")},
    }
    final_provenance = _json(provenance)
    pending.update({
        "status": "approved", "reviewer": declaration["reviewer_id"],
        "reviewed_at": declaration["reviewed_at"],
        "reviewed_ids": [row["id"] for row in rows],
        "provenance_sha256": _text_sha256(final_provenance),
    })
    final_files = {
        f"{batch}.jsonl": (candidate_dir / f"{batch}.jsonl").read_bytes(),
        f"{batch}.preview.md": (candidate_dir / f"{batch}.preview.md").read_bytes(),
        f"{batch}.provenance.json": final_provenance.encode("utf-8"),
        f"{batch}.review.json": _json(pending).encode("utf-8"),
    }
    destination = root / "data/synthetic"
    destination.mkdir(parents=True, exist_ok=True)
    if any((destination / name).exists() or (destination / name).is_symlink()
           for name in final_files):
        raise FileExistsError("Independent batch already exists in the repository")
    created = []
    try:
        for name, content in final_files.items():
            path = destination / name
            descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
            created.append(path)
            with os.fdopen(descriptor, "wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
        validated = load_approved(root, [batch])
        if len(validated) != 96:
            raise ValueError("Sealed independent batch has the wrong row count")
    except BaseException:
        for path in created:
            path.unlink(missing_ok=True)
        raise
    return {"batch": batch, "language": language, "category": category,
            "cards": len(validated), "pairs": len(validated) // 2,
            "batch_sha256": sha256(destination / f"{batch}.jsonl"),
            "provenance_sha256": sha256(destination / f"{batch}.provenance.json"),
            "actual_author_independence_verified_by_software": False,
            "content_safety_judgment_verified_by_software": False,
            "intended_labels_adjudicated": False}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare and seal private, independently authored abstract test pairs")
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("new", help="make two blank balanced author workbooks")
    create.add_argument("--batch", required=True)
    create.add_argument("--language", choices=("ro", "uk"), required=True)
    create.add_argument("--category", choices=("D1", "S1"), required=True)
    create.add_argument("--output-dir", type=Path, required=True)
    create.add_argument("--freeze-record", type=Path, required=True)
    create.add_argument("--root", type=Path, default=ROOT)
    assemble = commands.add_parser("assemble", help="make private, unapproved review material")
    assemble.add_argument("--author-a", type=Path, required=True)
    assemble.add_argument("--author-b", type=Path, required=True)
    assemble.add_argument("--owner-assignments", type=Path, required=True)
    assemble.add_argument("--output-dir", type=Path, required=True)
    assemble.add_argument("--root", type=Path, default=ROOT)
    seal = commands.add_parser("seal", help="intake after a separate content-review declaration")
    seal.add_argument("--candidate-dir", type=Path, required=True)
    seal.add_argument("--review-declaration", type=Path, required=True)
    seal.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        if args.command == "new":
            result = new_author_kit(args.root, args.batch, args.language,
                                    args.category, args.output_dir,
                                    freeze_record=args.freeze_record)
        elif args.command == "assemble":
            result = assemble_candidate(args.root, args.author_a, args.author_b,
                                        args.owner_assignments, args.output_dir)
        else:
            result = seal_candidate(args.root, args.candidate_dir,
                                    args.review_declaration)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
