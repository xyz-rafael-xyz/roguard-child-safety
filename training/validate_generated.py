"""Attest a reproducible abstract batch without a per-batch human approval.

Only registered local generators may be attested. This verifies provenance,
labels under those generators, surface separation, and exact files. It cannot
establish validity on real child language or prove a text's ethical provenance.
"""

from __future__ import annotations

import argparse
import json
import re
import runpy
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

from roguard.review import (APPLIES_TO, AUTO_VALIDATOR, CATEGORIES,
                            ReviewError, VALIDATED_GENERATORS,
                            candidate_manifest, load_approved,
                            reproduce_registered, sha256)


BATCH_SPEC = {
    "batch-0008": ("test", 40, 4),
    "batch-0009": ("train", 288, 12),
    "batch-0010": ("dev", 36, 3),
    "batch-0011": ("test", 48, 4),
    "batch-0012": ("test", 48, 4),
    "batch-0013": ("test", 96, 8),
    "batch-0014": ("train", 576, 48),
    "batch-0015": ("dev", 96, 8),
    "batch-0016": ("test", 144, 12),
    "batch-0017": ("test", 144, 12),
    "batch-0018": ("test", 144, 12),
    "batch-0019": ("test", 144, 12),
    "batch-0020": ("test", 144, 12),
    "batch-0021": ("train", 2304, 12),
    "batch-0022": ("dev", 144, 12),
    "batch-0023": ("test", 144, 12),
    "batch-0024": ("test", 144, 12),
    "batch-0025": ("test", 96, 24),
    "batch-0026": ("test", 144, 12),
    "batch-0027": ("test", 96, 24),
    "batch-0028": ("train", 384, 96),
    "batch-0029": ("dev", 48, 12),
    "batch-0030": ("test", 96, 24),
    "batch-0031": ("dev", 96, 24),
    "batch-0032": ("test", 96, 24),
    "batch-0033": ("dev", 96, 24),
    "batch-0034": ("test", 144, 24),
    "batch-0035": ("dev", 48, 24),
    "batch-0036": ("test", 96, 48),
    "batch-0037": ("test", 96, 48),
    "batch-0038": ("test", 96, 48),
}


def validate(root: Path, batch_id: str) -> dict:
    if batch_id not in BATCH_SPEC or batch_id not in VALIDATED_GENERATORS:
        raise ReviewError("No trusted generator registered for this batch")
    root = root.resolve()
    batch = root / "data" / "synthetic" / f"{batch_id}.jsonl"
    preview = root / "data" / "synthetic" / f"{batch_id}.preview.md"
    record_path = root / "data" / "synthetic" / f"{batch_id}.review.json"
    audit_path = root / "data" / "synthetic" / f"{batch_id}.audit.json"
    generator_path = root / VALIDATED_GENERATORS[batch_id]
    if any(path.is_symlink() for path in (batch, preview, record_path, audit_path, generator_path)):
        raise ReviewError("Validation inputs cannot be symlinks")
    expected = candidate_manifest(root, batch_id)
    rows = [json.loads(line) for line in batch.read_text(encoding="utf-8").splitlines()]
    split, expected_count, min_positive = BATCH_SPEC[batch_id]
    origin = ("abstract_template_v4" if batch_id == "batch-0008" else
              "abstract_binary_challenge_v1" if batch_id == "batch-0012" else
              "abstract_margin_challenge_v1" if batch_id == "batch-0013" else
              "abstract_compositional_v1" if batch_id in ("batch-0014", "batch-0015", "batch-0016") else
              "abstract_transfer_holdout_v1" if batch_id == "batch-0017" else
              "abstract_balanced_holdout_v1" if batch_id == "batch-0018" else
              "abstract_encoder_holdout_v1" if batch_id == "batch-0019" else
              "abstract_cross_category_holdout_v1" if batch_id == "batch-0020" else
              "abstract_surface_matrix_v1" if batch_id in ("batch-0021", "batch-0022") else
              "abstract_surface_independent_v1" if batch_id == "batch-0023" else
              "abstract_surface_independent_v2" if batch_id == "batch-0024" else
              "abstract_advisory_v1" if batch_id == "batch-0025" else
              "abstract_hybrid_v1" if batch_id == "batch-0026" else
              "abstract_advisory_prose_v1" if batch_id == "batch-0027" else
              "abstract_advisory_repair_v1" if batch_id in ("batch-0028", "batch-0029", "batch-0030") else
              "abstract_advisory_calibration_v1" if batch_id in ("batch-0031", "batch-0032") else
              "abstract_factor_holdout_v1" if batch_id in ("batch-0033", "batch-0034") else
              "abstract_joint_factor_holdout_v1" if batch_id in ("batch-0035", "batch-0036") else
              "abstract_balanced_joint_holdout_v1" if batch_id == "batch-0037" else
              "abstract_nli_transfer_holdout_v1" if batch_id == "batch-0038" else
              "abstract_counterfactual_v1")
    if rows != reproduce_registered(root, batch_id) or any(row["origin"] != origin or row["split"] != split for row in rows):
        raise ReviewError("Batch does not reproduce the registered abstract generator")
    if len(rows) != expected_count or len({row["id"] for row in rows}) != expected_count:
        raise ReviewError("Batch contains an unexpected number of distinct IDs")
    if any(re.search(r"\b(?:D1|R1|A1|P1|G1|S1)\b", row["text"]) for row in rows):
        raise ReviewError("Category code leaked into card text")
    if any("\"" in row["text"] or "“" in row["text"] for row in rows):
        raise ReviewError("Generated card contains quoted speech")
    for code in (("D1",) if batch_id in ("batch-0035", "batch-0036", "batch-0037", "batch-0038") else
                 ("D1", "S1") if batch_id in ("batch-0025", "batch-0027", "batch-0028", "batch-0029", "batch-0030", "batch-0031", "batch-0032", "batch-0033", "batch-0034") else CATEGORIES):
        applicable = [row for row in rows if row["source_kind"] in APPLIES_TO[code]]
        positive = sum(code in row["labels"] for row in applicable)
        if positive < min_positive or not any(code not in row["labels"] for row in applicable):
            raise ReviewError(f"Batch lacks positive/negative balance for {code}")
    prior_ids = [f"batch-{number:04d}" for number in range(4, int(batch_id[-4:]))]
    prior = load_approved(root, prior_ids)
    prior_keys = {(r["source_kind"], " ".join(r["text"].casefold().split())) for r in prior}
    keys = [(r["source_kind"], " ".join(r["text"].casefold().split())) for r in rows]
    if len(set(keys)) != len(keys) or any(key in prior_keys for key in keys):
        raise ReviewError("Exact text overlap with earlier batches")
    surface_rows = rows[::8] if batch_id == "batch-0021" else rows[::4] if batch_id == "batch-0028" else rows
    nearest = [max(SequenceMatcher(None, row["text"].casefold(), other["text"].casefold()).ratio()
                   for other in prior if row["source_kind"] == other["source_kind"])
               for row in surface_rows]
    measured = {"batch": batch_id, "batch_sha256": sha256(batch), "rows": expected_count,
                "exact_duplicate_texts": 0, "category_code_tokens_in_card_text": 0,
                "mean_nearest_same_kind_character_similarity": round(sum(nearest) / len(nearest), 3),
                "max_nearest_same_kind_character_similarity": round(max(nearest), 3),
                "note": "Surface check only; rule structures and abstract-card domain overlap earlier batches. Automated validation cannot establish real-world validity."}
    if batch_id == "batch-0026":
        build_contracts = runpy.run_path(str(generator_path))["build_contracts"]
        sidecar_path = root / "data/synthetic/batch-0026.contracts.jsonl"
        if sidecar_path.is_symlink():
            raise ReviewError("Hybrid contract sidecar cannot be a symlink")
        sidecars = [json.loads(line) for line in sidecar_path.read_text(encoding="utf-8").splitlines()]
        if (sidecars != build_contracts(batch_id) or len(sidecars) != len(rows) or
                any(item["id"] != row["id"] for item, row in zip(sidecars, rows))):
            raise ReviewError("Hybrid contract sidecar differs from registered generator")
        measured["contract_sidecar_sha256"] = sha256(sidecar_path)
    if batch_id == "batch-0021":
        measured["surface_audit_sample_rows"] = len(surface_rows)
        measured["note"] = ("Nearest-character-similarity was checked on every eighth training row; "
                            "exact duplicates were checked on all rows. Typed rule structures overlap. "
                            "This audit cannot establish linguistic or real-world validity.")
    if batch_id == "batch-0028":
        measured["surface_audit_sample_rows"] = len(surface_rows)
        measured["note"] = ("Nearest-character-similarity was checked on every fourth training row; "
                            "exact duplicates were checked on all rows. Typed rule structures overlap. "
                            "This audit cannot establish linguistic or real-world validity.")
    if measured["max_nearest_same_kind_character_similarity"] >= 0.90:
        raise ReviewError("Surface audit exceeds registered separation limit")
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        audit_path.write_text(json.dumps(measured, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audit = measured
    if audit != measured:
        raise ReviewError("Surface audit differs from current measurements")
    record = {key: value for key, value in expected.items() if key not in ("reviewer", "reviewed_at", "reviewed_ids")}
    record.update(status="validated", validator=AUTO_VALIDATOR,
                  validated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  validated_ids=[row["id"] for row in rows],
                  generator_path=VALIDATED_GENERATORS[batch_id],
                  generator_sha256=sha256(generator_path), audit_sha256=sha256(audit_path))
    try:
        existing = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        existing = None
    without_time = lambda value: {key: item for key, item in value.items() if key != "validated_at"}
    if (isinstance(existing, dict) and existing.get("validated_at") and
            without_time(existing) == without_time(record)):
        record = existing
    else:
        record_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    loaded = load_approved(root, [batch_id])
    if loaded != rows:
        raise ReviewError("Post-validation load disagrees with generator")
    return {"batch": batch_id, "rows": len(rows), "status": "validated",
            "batch_sha256": record["batch_sha256"], "generator_sha256": record["generator_sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch", choices=tuple(VALIDATED_GENERATORS), required=True)
    args = parser.parse_args()
    print(json.dumps(validate(args.root, args.batch), indent=2))


if __name__ == "__main__":
    main()
