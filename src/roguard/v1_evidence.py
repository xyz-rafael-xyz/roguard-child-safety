"""Read-only aggregation of independent abstract-study evidence for v1 planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .human_eval_registered import evaluate_adjudicated_registered
from .review import ReviewError, verify_independent_reviews

CELLS = ("ro:D1", "ro:S1", "uk:D1", "uk:S1")
STUDY_FILES = frozenset({"packet_dir", "reviewer_a", "reviewer_b", "adjudicated", "predictions"})


def _inside(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Study paths must be nonempty and relative to the repository")
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Study path escapes the repository")
    return path


def _manifest(root: Path, path: Path | None) -> dict:
    if path is None:
        return {}
    if path.is_symlink() or not path.is_file():
        raise ValueError("Study manifest must be a regular file")
    record = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or set(record) != {"schema_version", "studies"} or
            record["schema_version"] != 1 or not isinstance(record["studies"], dict) or
            set(record["studies"]) - set(CELLS)):
        raise ValueError("Invalid study manifest")
    studies = {}
    for cell, entry in record["studies"].items():
        if not isinstance(entry, dict) or set(entry) != STUDY_FILES:
            raise ValueError("Invalid study file entry")
        studies[cell] = {name: _inside(root, entry[name]) for name in STUDY_FILES}
    return studies


def audit_v1_evidence(root: Path, study_manifest: Path | None = None) -> dict:
    """Recompute supplied study results; never infer real-world readiness."""
    root = root.resolve()
    studies = _manifest(root, study_manifest)
    taxonomy = {}
    for language in ("ro", "uk"):
        try:
            digest = verify_independent_reviews(root, language)
        except (ReviewError, OSError, ValueError):
            taxonomy[language] = {"status": "pending_or_invalid_record"}
        else:
            taxonomy[language] = {"status": "record_structure_verified",
                                  "taxonomy_sha256": digest}

    results = {}
    for cell in CELLS:
        language, category = cell.split(":")
        if taxonomy[language]["status"] != "record_structure_verified":
            results[cell] = {"status": "blocked_by_taxonomy_review"}
            continue
        paths = studies.get(cell)
        if paths is None:
            results[cell] = {"status": "pending_study"}
            continue
        try:
            report = evaluate_adjudicated_registered(
                root, paths["packet_dir"], paths["reviewer_a"], paths["reviewer_b"],
                paths["adjudicated"], paths["predictions"])
        except (OSError, TypeError, ValueError, json.JSONDecodeError, ReviewError):
            results[cell] = {"status": "invalid_study_evidence"}
            continue
        if (not isinstance(report, dict) or report.get("language") != language or
                report.get("category") != category or
                report.get("prediction_model_id_matches_freeze") is not True or
                report.get("annotation_packet_binding_verified") is not True or
                report.get("packet_integrity_verified") is not True or
                report.get("answer_id_alignment_verified") is not True or
                report.get("adjudication_id_alignment_verified") is not True or
                report.get("prediction_source_alignment_verified") is not True):
            results[cell] = {"status": "invalid_study_evidence"}
            continue
        try:
            target = report["abstract_numeric_target"]["status"]
            if target not in {"meets_numeric_target", "misses_numeric_target",
                              "inconclusive_insufficient_valid_pairs"}:
                raise ValueError("Unexpected target status")
            results[cell] = {
                "status": target,
                "batch": report["batch"],
                "adjudicated_valid_one_fact_pairs": report["pairs"]["adjudicated_valid_one_fact_pairs"],
                "model_exact_valid_one_fact_pair_rate": report["model_exact_valid_one_fact_pair_rate"],
                "model_precision": report["model_precision"],
                "model_recall": report["model_recall_with_abstentions_as_misses"],
                "model_specificity": report["model_specificity_with_abstentions_as_misses"],
                "model_parse_coverage": report["model_parse_coverage_all_cards"],
            }
        except (KeyError, TypeError, ValueError):
            results[cell] = {"status": "invalid_study_evidence"}
    passed = (all(item["status"] == "record_structure_verified" for item in taxonomy.values()) and
              all(item["status"] == "meets_numeric_target" for item in results.values()))
    return {
        "schema_version": 1,
        "scope": "independently_authored_abstract_d1_s1_only",
        "taxonomy": taxonomy,
        "studies": results,
        "machine_abstract_gates_passed": passed,
        "human_identity_and_independence_verified_by_software": False,
        "model_execution_verified_by_software": False,
        "real_child_message_accuracy_established": False,
        "operational_child_safety_release_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit available v1 abstract-study evidence without releasing a model")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--study-manifest", type=Path)
    args = parser.parse_args()
    try:
        report = audit_v1_evidence(args.root, args.study_manifest)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
