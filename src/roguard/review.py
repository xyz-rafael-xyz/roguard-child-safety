"""Exact-byte provenance gates for taxonomy and synthetic batches."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import re
import unicodedata
from datetime import date
from pathlib import Path

from .policy import SUPPORT_FIELDS

CATEGORIES = ("D1", "R1", "A1", "P1", "G1", "S1")
LANGUAGES = ("ro", "uk")
ACTIVE_LANGUAGES = ("ro",)
TRAINING_LANGUAGES = ("ro", "uk")
KINDS = ("message", "routing_card", "response", "permission_card", "boundary_card", "gate_card")
SPLITS = ("train", "dev", "test")
APPLIES_TO = {
    "D1": ("message",), "R1": ("routing_card",), "A1": ("response",),
    "P1": ("permission_card", "boundary_card"), "G1": ("gate_card",), "S1": ("response",),
}
_BATCH_ID = re.compile(r"batch-[0-9]{4,8}\Z")
AUTO_VALIDATOR = "roguard-abstract-reproduction-v1"
INDEPENDENT_ORIGIN = "independent_abstract_v1"
INDEPENDENT_FACTORS = {
    "D1": ("source_role", "safety_or_support_anchor",
           "indirect_or_repeated_support_pattern", "explicit_support_request"),
    "S1": tuple(sorted(SUPPORT_FIELDS)),
}
VALIDATED_GENERATORS = {
    "batch-0008": "training/generate_challenge.py",
    "batch-0009": "training/generate_counterfactual.py",
    "batch-0010": "training/generate_counterfactual.py",
    "batch-0011": "training/generate_counterfactual.py",
    "batch-0012": "training/generate_binary_challenge.py",
    "batch-0013": "training/generate_margin_challenge.py",
    "batch-0014": "training/generate_compositional.py",
    "batch-0015": "training/generate_compositional.py",
    "batch-0016": "training/generate_compositional.py",
    "batch-0017": "training/generate_transfer_holdout.py",
    "batch-0018": "training/generate_balanced_holdout.py",
    "batch-0019": "training/generate_encoder_holdout.py",
    "batch-0020": "training/generate_cross_category_holdout.py",
    "batch-0021": "training/generate_surface_train.py",
    "batch-0022": "training/generate_surface_train.py",
    "batch-0023": "training/generate_surface_holdout.py",
    "batch-0024": "training/generate_surface_holdout_v12.py",
    "batch-0025": "training/generate_advisory_holdout.py",
    "batch-0026": "training/generate_hybrid_holdout.py",
    "batch-0027": "training/generate_advisory_prose_holdout.py",
    "batch-0028": "training/generate_advisory_repair.py",
    "batch-0029": "training/generate_advisory_repair.py",
    "batch-0030": "training/generate_advisory_repair.py",
    "batch-0031": "training/generate_advisory_calibration.py",
    "batch-0032": "training/generate_advisory_calibration.py",
    "batch-0033": "training/generate_factor_holdout.py",
    "batch-0034": "training/generate_factor_holdout.py",
    "batch-0035": "training/generate_joint_factor_holdout.py",
    "batch-0036": "training/generate_joint_factor_holdout.py",
    "batch-0037": "training/generate_balanced_joint_holdout.py",
    "batch-0038": "training/generate_nli_transfer_holdout.py",
}


class ReviewError(ValueError):
    """An approval, batch, or dataset invariant failed."""


def check_independent_card_script(text: str, language: str) -> None:
    """Reject a plainly wrong writing system; this does not validate language."""
    if not isinstance(text, str) or language not in LANGUAGES:
        raise ReviewError("Independent card needs text and a supported language")
    letters = [unicodedata.name(character, "") for character in text if character.isalpha()]
    expected = "CYRILLIC" if language == "uk" else "LATIN"
    matched = sum(name.startswith(expected) for name in letters)
    if len(letters) < 20 or matched * 2 < len(letters):
        raise ReviewError("Independent abstract card uses too little of its declared language script")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproduce_registered(root: Path, batch_id: str) -> list[dict]:
    """Run the locally registered, versioned generator for one batch."""
    generator = VALIDATED_GENERATORS.get(batch_id)
    if not generator:
        raise ReviewError(f"No trusted generator registered: {batch_id}")
    generator_path = root / generator
    if generator_path.is_symlink():
        raise ReviewError(f"Symlinked generator is not accepted: {batch_id}")
    spec = importlib.util.spec_from_file_location(f"roguard_generator_{batch_id.replace('-', '_')}", generator_path)
    if spec is None or spec.loader is None:
        raise ReviewError(f"Registered generator cannot be loaded: {batch_id}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parameters = inspect.signature(module.build_rows).parameters
    rows = module.build_rows(batch_id) if len(parameters) == 1 else module.build_rows()
    if not isinstance(rows, list):
        raise ReviewError(f"Registered generator returned invalid rows: {batch_id}")
    return rows


def taxonomy_sha256(root: Path, language: str) -> str:
    if language not in LANGUAGES:
        raise ReviewError("Unknown taxonomy language")
    digest = hashlib.sha256()
    name = f"taxonomy_{language}.md"
    path = root / "taxonomy" / name
    digest.update(name.encode("utf-8") + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def _approval(path: Path, digest_key: str, expected: str) -> None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Missing or invalid approval record: {path}") from exc
    if not isinstance(record, dict) or record.get("status") != "approved":
        raise ReviewError(f"Pending human review: {path}")
    if not str(record.get("reviewer", "")).strip() or not str(record.get("reviewed_at", "")).strip():
        raise ReviewError(f"Approval lacks reviewer or date: {path}")
    if record.get(digest_key) != expected:
        raise ReviewError(f"Approval digest differs from current content: {path}")


def verify_independent_reviews(root: Path, language: str) -> str:
    """Require distinct language and child-safety reviews of exact taxonomy bytes."""
    if language not in LANGUAGES:
        raise ReviewError("Unknown taxonomy language")
    digest = taxonomy_sha256(root, language)
    name = "review_ro_independent.json" if language == "ro" else "review_uk.json"
    path = root / "taxonomy" / name
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Missing independent review record: {path}") from exc
    if not isinstance(record, dict) or record.get("status") != "approved":
        raise ReviewError(f"Pending independent language and child-safety review: {path}")
    if record.get("taxonomy_sha256") != digest:
        raise ReviewError(f"Independent review digest differs from taxonomy: {path}")
    reviews = record.get("reviews")
    if (not isinstance(reviews, list) or len(reviews) != 2 or
            not all(isinstance(review, dict) for review in reviews) or
            not all(review.get("kind") in ("language", "child_safety") for review in reviews) or
            reviews[0]["kind"] == reviews[1]["kind"]):
        raise ReviewError(f"Independent review needs both distinct disciplines: {path}")
    names = []
    for review in reviews:
        reviewer = review.get("reviewer")
        reviewed_at = review.get("reviewed_at")
        decisions = review.get("category_decisions")
        if (not isinstance(reviewer, str) or not reviewer.strip() or
                review.get("independent_of_author") is not True or
                not isinstance(review.get("summary"), str) or
                not review["summary"].strip() or
                not isinstance(reviewed_at, str) or
                not isinstance(decisions, dict) or set(decisions) != set(CATEGORIES) or
                any(decisions[code] != "accept" for code in CATEGORIES)):
            raise ReviewError(f"Independent review is incomplete: {path}")
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", reviewed_at):
            raise ReviewError(f"Independent review has an invalid date: {path}")
        try:
            date.fromisoformat(reviewed_at)
        except ValueError as exc:
            raise ReviewError(f"Independent review has an invalid date: {path}") from exc
        names.append(reviewer.strip().casefold())
    if len(set(names)) != 2:
        raise ReviewError(f"Independent reviews need two different reviewers: {path}")
    return digest


def _batch_attestation(root: Path, batch_id: str, record_path: Path, expected: str) -> dict:
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError(f"Missing or invalid batch record: {record_path}") from exc
    if record.get("status") == "approved":
        _approval(record_path, "batch_sha256", expected)
    elif record.get("status") == "validated":
        generator = VALIDATED_GENERATORS.get(batch_id)
        if (record.get("validator") != AUTO_VALIDATOR or not generator or
                record.get("generator_path") != generator or not record.get("validated_at")):
            raise ReviewError(f"Invalid automated validation record: {batch_id}")
        generator_path = root / generator
        audit_path = root / "data" / "synthetic" / f"{batch_id}.audit.json"
        if (record.get("batch_sha256") != expected or
                record.get("generator_sha256") != sha256(generator_path) or
                record.get("audit_sha256") != sha256(audit_path)):
            raise ReviewError(f"Automated validation inputs changed: {batch_id}")
        if batch_id == "batch-0026":
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            sidecar = root / "data/synthetic/batch-0026.contracts.jsonl"
            if (sidecar.is_symlink() or
                    audit.get("contract_sidecar_sha256") != sha256(sidecar)):
                raise ReviewError("Hybrid contract sidecar differs from attestation")
        stored = [json.loads(line) for line in
                  (root / "data" / "synthetic" / f"{batch_id}.jsonl").read_text(encoding="utf-8").splitlines()]
        if stored != reproduce_registered(root, batch_id):
            raise ReviewError(f"Batch no longer reproduces registered generator: {batch_id}")
    else:
        raise ReviewError(f"Pending batch validation: {record_path}")
    return record


def _iso_date(value: object) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _verify_independent_batch(root: Path, batch_id: str, language: str,
                              record: dict, rows: list[dict]) -> None:
    """Check declarations for author-origin abstract cards; not human identity or label truth."""
    if batch_id in VALIDATED_GENERATORS or record.get("status") != "approved":
        raise ReviewError("Independent author cards need a separate human approval")
    verify_independent_reviews(root, language)
    path = root / "data/synthetic" / f"{batch_id}.provenance.json"
    if (path.is_symlink() or not path.is_file() or
            record.get("provenance_sha256") != sha256(path)):
        raise ReviewError("Independent author provenance differs")
    try:
        provenance = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewError("Invalid independent author provenance") from exc
    authors = provenance.get("authors") if isinstance(provenance, dict) else None
    safety = provenance.get("safety_review") if isinstance(provenance, dict) else None
    pair_authors = provenance.get("pair_authors") if isinstance(provenance, dict) else None
    pair_factors = provenance.get("pair_factors") if isinstance(provenance, dict) else None
    if (not isinstance(provenance, dict) or set(provenance) != {
            "schema_version", "batch", "language", "category", "authors", "pair_authors",
            "pair_factors", "model_freeze_path", "model_freeze_sha256",
            "safety_review", "label_status"} or
            provenance.get("schema_version") != 2 or provenance.get("batch") != batch_id or
            provenance.get("language") != language or
            provenance.get("category") not in ("D1", "S1") or
            provenance.get("label_status") != "author_intended_not_adjudicated" or
            not isinstance(authors, list) or len(authors) != 2 or
            not isinstance(pair_authors, list) or len(pair_authors) != 48 or
            not isinstance(pair_factors, list) or len(pair_factors) != 48 or
            not isinstance(safety, dict) or set(safety) != {
                "reviewer_id", "reviewed_at", "abstract_only_checked",
                "no_realistic_content_checked", "pair_logic_checked",
                "factor_balance_checked", "prior_overlap_checked"}):
        raise ReviewError("Independent author provenance is incomplete")
    from .study_freeze import verify_study_freeze
    if (not isinstance(provenance["model_freeze_path"], str) or
            not isinstance(provenance["model_freeze_sha256"], str)):
        raise ReviewError("Independent batch lacks a model freeze binding")
    try:
        freeze = verify_study_freeze(root, Path(provenance["model_freeze_path"]),
                                     language, provenance["category"])
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ReviewError("Independent batch model freeze differs") from exc
    if freeze["sha256"] != provenance["model_freeze_sha256"]:
        raise ReviewError("Independent batch model freeze digest differs")
    ids = []
    for author in authors:
        if (not isinstance(author, dict) or set(author) != {
                "author_id", "native_language", "independent_of_project", "authored_at"} or
                not isinstance(author["author_id"], str) or
                not re.fullmatch(r"[A-Za-z0-9_-]{3,40}", author["author_id"]) or
                author["native_language"] != language or
                author["independent_of_project"] is not True or
                not _iso_date(author["authored_at"])):
            raise ReviewError("Independent author declaration is incomplete")
        ids.append(author["author_id"])
    if (len(set(ids)) != 2 or
            any(author["authored_at"] < freeze["registered_on"] for author in authors) or
            any(type(item) is not str or item not in ids for item in pair_authors) or
            any(pair_authors.count(author_id) != 24 for author_id in ids) or
            safety["reviewer_id"] in ids or
            record.get("reviewer") != safety["reviewer_id"] or
            record.get("reviewed_at") != safety["reviewed_at"] or
            not isinstance(safety["reviewer_id"], str) or
            not re.fullmatch(r"[A-Za-z0-9_-]{3,40}", safety["reviewer_id"]) or
            not _iso_date(safety["reviewed_at"]) or
            any(safety["reviewed_at"] < author["authored_at"] for author in authors) or
            any(safety[key] is not True for key in (
                "abstract_only_checked", "no_realistic_content_checked", "pair_logic_checked",
                "factor_balance_checked", "prior_overlap_checked"))):
        raise ReviewError("Independent author or safety-review declarations differ")
    category = provenance["category"]
    factors = INDEPENDENT_FACTORS[category]
    per_author_factor = 24 // len(factors)
    if (any(type(factor) is not str or factor not in factors for factor in pair_factors) or
            any(sum(author == author_id and factor == expected
                    for author, factor in zip(pair_authors, pair_factors)) != per_author_factor
                for author_id in ids for expected in factors)):
        raise ReviewError("Independent declared factor coverage differs")
    kind = "message" if category == "D1" else "response"
    if (len(rows) != 96 or any(not isinstance(row, dict) for row in rows) or
            any(row.get("origin") != INDEPENDENT_ORIGIN or
                               row.get("source_kind") != kind or row.get("split") != "test"
                               for row in rows) or
            any(left.get("labels") != [] or right.get("labels") != [category]
                for left, right in zip(rows[::2], rows[1::2])) or
            any(not isinstance(row.get("text"), str) or
                not 20 <= len(row["text"]) <= 1000 or
                row["text"] != row["text"].strip() or
                re.search(r"\b(?:D1|R1|A1|P1|G1|S1)\b", row["text"]) or
                any(mark in row["text"] for mark in ('"', "“", "”", "|", "\n", "\r"))
                for row in rows)):
        raise ReviewError("Independent abstract batch shape differs")
    for row in rows:
        check_independent_card_script(row["text"], language)
    keys = {" ".join(row["text"].casefold().split()) for row in rows}
    if len(keys) != len(rows):
        raise ReviewError("Independent abstract batch repeats card text")
    for prior in (root / "data/synthetic").glob("batch-*.jsonl"):
        if prior.name == f"{batch_id}.jsonl":
            continue
        if prior.is_symlink():
            raise ReviewError("Symlinked prior batch is not accepted")
        for line in prior.read_text(encoding="utf-8").splitlines():
            try:
                earlier = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewError(f"Invalid prior batch while checking overlap: {prior.name}") from exc
            if (isinstance(earlier, dict) and earlier.get("language") == language and
                    earlier.get("source_kind") == kind and
                    isinstance(earlier.get("text"), str) and
                    " ".join(earlier["text"].casefold().split()) in keys):
                raise ReviewError("Independent abstract batch repeats an earlier card")


def verify_taxonomy(root: Path, language: str) -> str:
    if language not in TRAINING_LANGUAGES:
        raise ReviewError(f"Taxonomy is not ready for training in {language}")
    if language == "uk":
        return verify_independent_reviews(root, language)
    digest = taxonomy_sha256(root, language)
    _approval(root / "taxonomy" / f"review_{language}.json", "taxonomy_sha256", digest)
    return digest


def candidate_manifest(root: Path, batch_id: str) -> dict[str, object]:
    if not _BATCH_ID.fullmatch(batch_id):
        raise ReviewError("Batch ID must have form batch-0001")
    path = root / "data" / "synthetic" / f"{batch_id}.jsonl"
    preview = root / "data" / "synthetic" / f"{batch_id}.preview.md"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _check_preview_rows(rows, preview)
    languages = {row["language"] for row in rows}
    if len(languages) != 1:
        raise ReviewError("Each review batch must contain exactly one language")
    language = languages.pop()
    return {
        "status": "pending",
        "reviewer": "",
        "reviewed_at": "",
        "reviewed_ids": [],
        "language": language,
        "batch_sha256": sha256(path),
        "preview_sha256": sha256(preview),
        "taxonomy_sha256": taxonomy_sha256(root, language),
    }


def _check_preview_rows(rows: list[dict], preview: Path) -> None:
    """Ensure the readable review table displays every training label and card verbatim."""
    review_text = preview.read_text(encoding="utf-8")
    for row in rows:
        if "|" in row["text"] or "\n" in row["text"]:
            raise ReviewError(f"Card cannot be shown safely in a Markdown table: {row['id']}")
        table_line = (f"| {row['id']} | {row['language']} | {row['source_kind']} | "
                      f"{','.join(row['labels']) or 'none'} | {row['text']} |")
        if review_text.count(table_line) != 1:
            raise ReviewError(f"Review preview omits or misstates row: {row['id']}")


def load_approved(root: Path, batch_ids: list[str]) -> list[dict]:
    """Load explicitly selected human-approved or reproducibly validated batches."""
    if not batch_ids or len(set(batch_ids)) != len(batch_ids):
        raise ReviewError("Specify distinct batch IDs explicitly")
    rows: list[dict] = []
    seen_ids: set[str] = set()
    seen_texts: dict[tuple[str, str, str], str] = {}
    for batch_id in batch_ids:
        if not _BATCH_ID.fullmatch(batch_id):
            raise ReviewError("Invalid batch ID")
        batch = root / "data" / "synthetic" / f"{batch_id}.jsonl"
        preview = root / "data" / "synthetic" / f"{batch_id}.preview.md"
        record_path = root / "data" / "synthetic" / f"{batch_id}.review.json"
        if batch.is_symlink() or preview.is_symlink() or record_path.is_symlink():
            raise ReviewError("Symlinked batch files are not accepted")
        digest = sha256(batch)
        record = _batch_attestation(root, batch_id, record_path, digest)
        language = record.get("language")
        if language not in LANGUAGES:
            raise ReviewError(f"Batch review lacks a valid language: {batch_id}")
        raw_rows = [json.loads(line) for line in batch.read_text(encoding="utf-8").splitlines()]
        independent = any(isinstance(row, dict) and row.get("origin") == INDEPENDENT_ORIGIN
                          for row in raw_rows)
        if independent:
            _verify_independent_batch(root, batch_id, language, record, raw_rows)
        tax_digest = verify_independent_reviews(root, language) if independent else verify_taxonomy(root, language)
        if record.get("preview_sha256") != sha256(preview):
            raise ReviewError(f"Reviewed preview differs from current preview: {batch_id}")
        if record.get("taxonomy_sha256") != tax_digest:
            raise ReviewError(f"Batch was reviewed under another taxonomy: {batch_id}")
        batch_ids_seen: list[str] = []
        for number, line in enumerate(batch.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewError(f"Invalid JSON at {batch_id}:{number}") from exc
            required = {"id", "language", "source_kind", "split", "text", "labels", "origin"}
            if set(row) != required:
                raise ReviewError(f"Wrong row fields at {batch_id}:{number}")
            if not isinstance(row["id"], str) or not row["id"].startswith(batch_id + "-"):
                raise ReviewError(f"Invalid ID at {batch_id}:{number}")
            if row["id"] in seen_ids:
                raise ReviewError(f"Repeated row ID: {row['id']}")
            seen_ids.add(row["id"])
            batch_ids_seen.append(row["id"])
            if row["language"] not in LANGUAGES or row["source_kind"] not in KINDS or row["split"] not in SPLITS:
                raise ReviewError(f"Invalid language, kind or split at {batch_id}:{number}")
            if row["language"] != language:
                raise ReviewError(f"Mixed-language batch: {batch_id}")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ReviewError(f"Empty text at {batch_id}:{number}")
            if not isinstance(row["labels"], list) or len(row["labels"]) != len(set(row["labels"])) or not set(row["labels"]) <= set(CATEGORIES):
                raise ReviewError(f"Invalid labels at {batch_id}:{number}")
            if row["origin"] not in {"abstract_template_v3", "abstract_template_v4", "abstract_counterfactual_v1", "abstract_binary_challenge_v1", "abstract_margin_challenge_v1", "abstract_compositional_v1", "abstract_transfer_holdout_v1", "abstract_balanced_holdout_v1", "abstract_encoder_holdout_v1", "abstract_cross_category_holdout_v1", "abstract_surface_matrix_v1", "abstract_surface_independent_v1", "abstract_surface_independent_v2", "abstract_advisory_v1", "abstract_hybrid_v1", "abstract_advisory_prose_v1", "abstract_advisory_repair_v1", "abstract_advisory_calibration_v1", "abstract_factor_holdout_v1", "abstract_joint_factor_holdout_v1", "abstract_balanced_joint_holdout_v1", "abstract_nli_transfer_holdout_v1", INDEPENDENT_ORIGIN}:
                raise ReviewError(f"Disallowed source origin at {batch_id}:{number}")
            if (row["origin"] == INDEPENDENT_ORIGIN) != independent:
                raise ReviewError(f"Mixed author provenance at {batch_id}:{number}")
            if any(row["source_kind"] not in APPLIES_TO[code] for code in row["labels"]):
                raise ReviewError(f"Label does not apply to source kind at {batch_id}:{number}")
            key = (row["language"], row["source_kind"], " ".join(row["text"].casefold().split()))
            prior_split = seen_texts.get(key)
            if prior_split is not None:
                raise ReviewError(f"Duplicate text across batches or splits: {row['id']}")
            seen_texts[key] = row["split"]
            rows.append(row)
        ids_key = "reviewed_ids" if record["status"] == "approved" else "validated_ids"
        if not batch_ids_seen or record.get(ids_key) != batch_ids_seen:
            raise ReviewError(f"Batch attestation must enumerate every row in order: {batch_id}")
        _check_preview_rows(rows[-len(batch_ids_seen):], preview)
    return rows


def load_train_dev(root: Path, batch_ids: list[str], language: str, *,
                   required_categories: tuple[str, ...] = CATEGORIES) -> tuple[list[dict], list[dict]]:
    """Return attested train/dev rows with coverage for the declared task codes."""
    if (not required_categories or len(set(required_categories)) != len(required_categories) or
            any(code not in CATEGORIES for code in required_categories)):
        raise ReviewError("Specify distinct supported training categories")
    rows = load_approved(root, batch_ids)
    if any(row["split"] == "test" for row in rows):
        raise ReviewError("Do not give held-out test batches to the trainer")
    rows = [row for row in rows if row["language"] == language]
    train = [row for row in rows if row["split"] == "train"]
    dev = [row for row in rows if row["split"] == "dev"]
    if not train or not dev:
        raise ReviewError("Approved training and development splits are both required")
    if any(set(row["labels"]) - set(required_categories) for row in rows):
        raise ReviewError("Training task contains a label outside required categories")
    for code in required_categories:
        if not any(code in row["labels"] for row in train):
            raise ReviewError(f"Training split lacks positive {language}/{code}")
        applicable = [row for row in dev if row["source_kind"] in APPLIES_TO[code]]
        if not any(code in row["labels"] for row in applicable):
            raise ReviewError(f"Development split lacks positive {language}/{code}")
        if not any(code not in row["labels"] for row in applicable):
            raise ReviewError(f"Development split lacks negative {language}/{code}")
    return train, dev
