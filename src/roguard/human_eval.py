"""Evaluate abstract-card predictions against separately adjudicated declarations."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from .agreement_batch import _one_category, _read_jsonl
from .annotate_packet import _packet_rows
from .blind_packets import KINDS
from .cli import assess_json
from .review import (INDEPENDENT_FACTORS, INDEPENDENT_ORIGIN,
                     VALIDATED_GENERATORS, load_approved, sha256, taxonomy_sha256)
from .review_audit import audit_review_session

MODEL_ID = re.compile(r"[A-Za-z0-9._-]{1,80}\Z")
ABSTRACT_TARGET = {
    "minimum_valid_one_fact_pairs": 44,
    "minimum_valid_pairs_per_author_surface": 22,
    "minimum_valid_pairs_per_d1_factor": 10,
    "minimum_valid_pairs_per_s1_factor": 5,
    "minimum_recall": 0.90,
    "minimum_specificity": 0.90,
    "minimum_precision": 0.90,
    "minimum_exact_valid_pair_rate": 0.90,
    "minimum_exact_valid_pair_rate_per_author_surface": 0.85,
    "minimum_exact_valid_pair_rate_per_factor": 0.80,
    "required_parse_coverage": 1.0,
}


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _wilson_95(successes: int, total: int) -> list[float] | None:
    """Descriptive binomial interval; card dependence can make it optimistic."""
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    scale = 1 + z * z / total
    center = (p + z * z / (2 * total)) / scale
    radius = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / scale
    return [round(max(0.0, center - radius), 6), round(min(1.0, center + radius), 6)]


def _one_fact_change(category: str, left: dict, right: dict) -> str | None:
    """Find the sole adjudicated fact changed in a pass-to-review pair."""
    if category == "D1":
        left_facts, right_facts = left["disclosure"], right["disclosure"]
        changed = [name for name in INDEPENDENT_FACTORS["D1"]
                   if left_facts[name] != right_facts[name]]
        return changed[0] if len(changed) == 1 else None
    left_facts, right_facts = left["support"], right["support"]
    if left_facts["applicable_fields"] is None or right_facts["applicable_fields"] is None:
        return None
    if set(left_facts["applicable_fields"]) != set(right_facts["applicable_fields"]):
        return None
    left_passed, right_passed = set(left_facts["passed_fields"]), set(right_facts["passed_fields"])
    removed = left_passed - right_passed
    return next(iter(removed)) if len(removed) == 1 and not right_passed - left_passed else None


def _numeric_target(category: str, independent: bool, positive: int, negative: int,
                    counts: dict[str, int], pair_counts: dict[str, int],
                    author_surfaces: list[dict], factor_surfaces: list[dict], parsed_total: int,
                    total_rows: int) -> dict:
    if not independent:
        return {"status": "not_applicable_generator_batch"}
    exact_rate = _rate(pair_counts["model_exact_valid_one_fact_pairs"],
                       pair_counts["adjudicated_valid_one_fact_pairs"])
    recall = _rate(counts["tp"], positive)
    specificity = _rate(counts["tn"], negative)
    precision = _rate(counts["tp"], counts["tp"] + counts["fp"])
    factor_minimum = ABSTRACT_TARGET[
        "minimum_valid_pairs_per_d1_factor" if category == "D1" else
        "minimum_valid_pairs_per_s1_factor"]
    enough = (pair_counts["adjudicated_valid_one_fact_pairs"] >=
              ABSTRACT_TARGET["minimum_valid_one_fact_pairs"] and
              all(surface["adjudicated_valid_one_fact_pairs"] >=
                  ABSTRACT_TARGET["minimum_valid_pairs_per_author_surface"]
                  for surface in author_surfaces) and
              all(surface["adjudicated_valid_one_fact_pairs"] >= factor_minimum
                  for surface in factor_surfaces))
    meets = (enough and recall is not None and recall >= ABSTRACT_TARGET["minimum_recall"] and
             specificity is not None and specificity >= ABSTRACT_TARGET["minimum_specificity"] and
             precision is not None and precision >= ABSTRACT_TARGET["minimum_precision"] and
             exact_rate is not None and exact_rate >= ABSTRACT_TARGET["minimum_exact_valid_pair_rate"] and
             all(surface["exact_valid_one_fact_pair_rate"] is not None and
                 surface["exact_valid_one_fact_pair_rate"] >=
                 ABSTRACT_TARGET["minimum_exact_valid_pair_rate_per_author_surface"]
                 for surface in author_surfaces) and
             all(surface["exact_valid_one_fact_pair_rate"] is not None and
                 surface["exact_valid_one_fact_pair_rate"] >=
                 ABSTRACT_TARGET["minimum_exact_valid_pair_rate_per_factor"]
                 for surface in factor_surfaces) and
             _rate(parsed_total, total_rows) == ABSTRACT_TARGET["required_parse_coverage"])
    return {
        "status": ("inconclusive_insufficient_valid_pairs" if not enough else
                   "meets_numeric_target" if meets else "misses_numeric_target"),
        "category": category,
        "scope": "independently_authored_abstract_cards_only",
        "thresholds": dict(ABSTRACT_TARGET),
        "human_independence_or_label_validity_verified": False,
        "real_child_language_accuracy_established": False,
    }


def _read_adjudication(path: Path, mapping: dict[str, str],
                       language: str, category: str) -> dict[str, str]:
    items = _read_jsonl(path)
    if len(items) != len(mapping):
        raise ValueError("Adjudication needs one decision for every packet item")
    results = {}
    expected_kind = "disclosure" if category == "D1" else "support"
    for item in items:
        kind, card = _one_category(item)
        opaque = item["item_id"]
        if opaque not in mapping or opaque in results or kind != expected_kind or card["language"] != language:
            raise ValueError("Adjudication IDs, category, or language differ from the packet")
        results[opaque] = assess_json(card)["findings"][0]["status"]
    if set(results) != set(mapping):
        raise ValueError("Adjudication IDs differ from the packet")
    return results


def _read_predictions(path: Path, batch: str, batch_hash: str,
                      category: str, source_ids: list[str]) -> tuple[str, dict[str, bool | None]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict) or set(payload) != {
            "schema_version", "batch", "batch_sha256", "category", "model_id", "predictions"} or
            payload["schema_version"] != 1 or payload["batch"] != batch or
            payload["batch_sha256"] != batch_hash or payload["category"] != category or
            not isinstance(payload["model_id"], str) or
            not MODEL_ID.fullmatch(payload["model_id"]) or
            not isinstance(payload["predictions"], list) or
            len(payload["predictions"]) != len(source_ids)):
        raise ValueError("Frozen prediction file does not name this exact abstract batch")
    predictions = {}
    for source_id, item in zip(source_ids, payload["predictions"]):
        if (not isinstance(item, dict) or set(item) != {"id", "predicted", "strict_parse"} or
                item["id"] != source_id or type(item["strict_parse"]) is not bool or
                item["predicted"] not in ([], [category]) or
                (not item["strict_parse"] and item["predicted"])):
            raise ValueError("Frozen predictions differ from source order or output contract")
        predictions[source_id] = (category in item["predicted"]
                                  if item["strict_parse"] else None)
    return payload["model_id"], predictions


def _read_blind_predictions(path: Path, packet: Path, packet_hash: str,
                            language: str, category: str,
                            id_map: dict[str, str]) -> tuple[str, dict[str, bool | None]]:
    """Join opaque, packet-ordered decisions to source IDs only inside evaluation."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows, packet_language, packet_category = _packet_rows(packet)
    if (not isinstance(payload, dict) or set(payload) != {
            "schema_version", "packet_sha256", "language", "category", "model_id", "predictions"} or
            payload["schema_version"] != 2 or
            payload["packet_sha256"] != packet_hash or sha256(packet) != packet_hash or
            payload["language"] != language or packet_language != language or
            payload["category"] != category or packet_category != category or
            not isinstance(payload["model_id"], str) or
            not MODEL_ID.fullmatch(payload["model_id"]) or
            not isinstance(payload["predictions"], list) or
            len(payload["predictions"]) != len(rows)):
        raise ValueError("Frozen blind predictions differ from the exact reviewer packet")
    predictions = {}
    for row, item in zip(rows, payload["predictions"]):
        if (not isinstance(item, dict) or set(item) != {"item_id", "predicted", "strict_parse"} or
                item["item_id"] != row["item_id"] or
                item["item_id"] not in id_map or type(item["strict_parse"]) is not bool or
                item["predicted"] not in ([], [category]) or
                (not item["strict_parse"] and item["predicted"])):
            raise ValueError("Frozen blind predictions differ from packet order or output contract")
        predictions[id_map[item["item_id"]]] = (
            category in item["predicted"] if item["strict_parse"] else None)
    if len(predictions) != len(rows):
        raise ValueError("Frozen blind predictions do not map one-to-one to source cards")
    return payload["model_id"], predictions


def evaluate_adjudicated(
    root: Path,
    packet_dir: Path,
    reviewer_a: Path,
    reviewer_b: Path,
    adjudicated: Path,
    predictions_path: Path,
) -> dict:
    """Audit exact inputs and score parsed labels against complete adjudications."""
    root, packet_dir = root.resolve(), packet_dir.resolve()
    files = {
        "owner_map": packet_dir / "owner-map.json",
        "packet_a": packet_dir / "reviewer-a.jsonl",
        "packet_b": packet_dir / "reviewer-b.jsonl",
        "reviewer_a": reviewer_a,
        "reviewer_b": reviewer_b,
        "adjudicated": adjudicated,
        "predictions": predictions_path,
    }
    input_hashes = {name: sha256(path) for name, path in files.items()}
    audit = audit_review_session(root, packet_dir, reviewer_a, reviewer_b)
    owner = json.loads(files["owner_map"].read_text(encoding="utf-8"))
    batch, category, language = audit["batch"], audit["category"], audit["language"]
    source_rows = [row for row in load_approved(root, [batch])
                   if row["source_kind"] == KINDS[category]]
    independent = all(row["origin"] == INDEPENDENT_ORIGIN for row in source_rows)
    source_base = root / "data/synthetic" / batch
    source_files = {
        "source_batch": source_base.with_suffix(".jsonl"),
        "source_preview": source_base.with_suffix(".preview.md"),
        "source_attestation": source_base.with_suffix(".review.json"),
        "taxonomy": root / "taxonomy" / f"taxonomy_{language}.md",
    }
    if independent:
        source_files["source_provenance"] = source_base.with_suffix(".provenance.json")
        review_name = "review_ro_independent.json" if language == "ro" else "review_uk.json"
    else:
        review_name = "review_ro.json" if language == "ro" else "review_uk.json"
        if batch in VALIDATED_GENERATORS:
            source_files["source_audit"] = source_base.with_suffix(".audit.json")
            source_files["source_generator"] = root / VALIDATED_GENERATORS[batch]
    source_files["taxonomy_review"] = root / "taxonomy" / review_name
    files.update(source_files)
    input_hashes.update({name: sha256(path) for name, path in source_files.items()})
    source_record = json.loads(source_files["source_attestation"].read_text(encoding="utf-8"))
    if (input_hashes["source_batch"] != owner["batch_sha256"] or
            input_hashes["source_attestation"] != owner["attestation_sha256"] or
            taxonomy_sha256(root, language) != owner["taxonomy_sha256"] or
            input_hashes["source_preview"] != source_record["preview_sha256"] or
            ("source_provenance" in source_files and
             input_hashes["source_provenance"] != source_record["provenance_sha256"]) or
            ("source_generator" in source_files and
             (input_hashes["source_generator"] != source_record["generator_sha256"] or
              input_hashes["source_audit"] != source_record["audit_sha256"]))):
        raise ValueError("Attested source changed during evaluation")
    mapping = {entry["item_id"]: entry["source_id"] for entry in owner["id_map"]}
    reverse = {source_id: opaque for opaque, source_id in mapping.items()}
    if len(reverse) != len(source_rows):
        raise ValueError("Owner mapping is not one-to-one")
    judgments = _read_adjudication(adjudicated, mapping, language, category)
    if independent:
        model_id, predictions = _read_blind_predictions(
            predictions_path, packet_dir / "reviewer-a.jsonl",
            owner["packet_sha256"]["reviewer-a.jsonl"], language, category, mapping)
    else:
        model_id, predictions = _read_predictions(
            predictions_path, batch, owner["batch_sha256"], category,
            [row["id"] for row in source_rows])
    original_a = {item["item_id"]: item["card"] for item in _read_jsonl(reviewer_a)}
    original_b = {item["item_id"]: item["card"] for item in _read_jsonl(reviewer_b)}
    final_cards = {item["item_id"]: item["card"] for item in _read_jsonl(adjudicated)}
    counts = {key: 0 for key in ("tp", "fp", "fn", "tn", "model_abstentions",
                                  "positive_abstentions", "negative_abstentions")}
    complete = positive = negative = ambiguous = generator_matches = 0
    for row in source_rows:
        status = judgments[reverse[row["id"]]]
        if status == "incomplete":
            ambiguous += 1
            continue
        if status not in ("pass", "review"):
            raise ValueError("Adjudication has an unsupported decision")
        complete += 1
        reference = status == "review"
        positive += reference
        negative += not reference
        generator_matches += (reference == (category in row["labels"]))
        prediction = predictions[row["id"]]
        if prediction is None:
            counts["model_abstentions"] += 1
            counts["positive_abstentions" if reference else "negative_abstentions"] += 1
        else:
            counts["tp" if prediction and reference else
                   "fp" if prediction else "fn" if reference else "tn"] += 1
    pair_counts = {key: 0 for key in (
        "adjudicated_contrast_pairs", "adjudicated_other_pairs", "incomplete_pairs",
        "model_exact_contrast_pairs", "adjudicated_valid_one_fact_pairs",
        "adjudicated_multifact_or_nochange_pairs", "adjudicated_factor_mismatch_pairs",
        "model_exact_valid_one_fact_pairs")}
    if len(source_rows) % 2:
        raise ValueError("Abstract review batch needs complete adjacent pairs")
    author_surfaces = []
    factor_surfaces = []
    factor_by_code = {}
    pair_slots = None
    pair_factors = None
    if independent:
        provenance = json.loads((root / f"data/synthetic/{batch}.provenance.json").read_text(encoding="utf-8"))
        authors = [item["author_id"] for item in provenance["authors"]]
        pair_slots = [authors.index(author) for author in provenance["pair_authors"]]
        pair_factors = provenance["pair_factors"]
        author_surfaces = [{"slot": index + 1, "pairs": 0,
                            "adjudicated_contrast_pairs": 0,
                            "adjudicated_other_pairs": 0, "incomplete_pairs": 0,
                            "model_exact_contrast_pairs": 0,
                            "adjudicated_valid_one_fact_pairs": 0,
                            "adjudicated_multifact_or_nochange_pairs": 0,
                            "adjudicated_factor_mismatch_pairs": 0,
                            "model_exact_valid_one_fact_pairs": 0,
                            "false_reviews_on_contrast_negatives": 0,
                            "missed_reviews_on_contrast_positives": 0,
                            "abstentions_on_contrast_cards": 0}
                           for index in range(len(authors))]
        factor_surfaces = [{"factor": factor, "pairs": 0,
                            "adjudicated_valid_one_fact_pairs": 0,
                            "model_exact_valid_one_fact_pairs": 0,
                            "false_reviews_on_contrast_negatives": 0,
                            "missed_reviews_on_contrast_positives": 0}
                           for factor in INDEPENDENT_FACTORS[category]]
        factor_by_code = {surface["factor"]: surface for surface in factor_surfaces}
    for pair_index, (left, right) in enumerate(zip(source_rows[::2], source_rows[1::2])):
        if category in left["labels"] or category not in right["labels"]:
            raise ValueError("Source rows are not intended negative-to-positive pairs")
        surface = author_surfaces[pair_slots[pair_index]] if pair_slots is not None else None
        factor_surface = factor_by_code[pair_factors[pair_index]] if pair_factors is not None else None
        if surface is not None:
            surface["pairs"] += 1
            factor_surface["pairs"] += 1
        left_status = judgments[reverse[left["id"]]]
        right_status = judgments[reverse[right["id"]]]
        if "incomplete" in (left_status, right_status):
            pair_counts["incomplete_pairs"] += 1
            if surface is not None:
                surface["incomplete_pairs"] += 1
        elif (left_status, right_status) != ("pass", "review"):
            pair_counts["adjudicated_other_pairs"] += 1
            if surface is not None:
                surface["adjudicated_other_pairs"] += 1
        else:
            pair_counts["adjudicated_contrast_pairs"] += 1
            exact = (
                predictions[left["id"]] is False and predictions[right["id"]] is True)
            pair_counts["model_exact_contrast_pairs"] += exact
            if surface is not None:
                surface["adjudicated_contrast_pairs"] += 1
                surface["model_exact_contrast_pairs"] += exact
                surface["false_reviews_on_contrast_negatives"] += predictions[left["id"]] is True
                surface["missed_reviews_on_contrast_positives"] += predictions[right["id"]] is not True
                surface["abstentions_on_contrast_cards"] += sum(
                    predictions[row["id"]] is None for row in (left, right))
                factor_surface["false_reviews_on_contrast_negatives"] += predictions[left["id"]] is True
                factor_surface["missed_reviews_on_contrast_positives"] += predictions[right["id"]] is not True
                changed = _one_fact_change(
                    category, final_cards[reverse[left["id"]]],
                    final_cards[reverse[right["id"]]])
                if changed is None:
                    key = "adjudicated_multifact_or_nochange_pairs"
                elif changed != pair_factors[pair_index]:
                    key = "adjudicated_factor_mismatch_pairs"
                else:
                    key = "adjudicated_valid_one_fact_pairs"
                    pair_counts["model_exact_valid_one_fact_pairs"] += exact
                    surface["model_exact_valid_one_fact_pairs"] += exact
                    factor_surface["adjudicated_valid_one_fact_pairs"] += 1
                    factor_surface["model_exact_valid_one_fact_pairs"] += exact
                pair_counts[key] += 1
                surface[key] += 1

    for surface in author_surfaces:
        surface["exact_contrast_pair_rate"] = _rate(
            surface["model_exact_contrast_pairs"], surface["adjudicated_contrast_pairs"])
        surface["exact_valid_one_fact_pair_rate"] = _rate(
            surface["model_exact_valid_one_fact_pairs"],
            surface["adjudicated_valid_one_fact_pairs"])
    for surface in factor_surfaces:
        surface["exact_valid_one_fact_pair_rate"] = _rate(
            surface["model_exact_valid_one_fact_pairs"],
            surface["adjudicated_valid_one_fact_pairs"])

    parsed_total = sum(value is not None for value in predictions.values())
    report = {
        "schema_version": 1, "scope": "abstract_cards_with_adjudicated_declarations_only",
        "batch": batch, "batch_sha256": owner["batch_sha256"],
        "taxonomy_sha256": owner["taxonomy_sha256"],
        "category": category, "language": language, "model_id": model_id,
        "input_sha256": input_hashes,
        "packet_sha256": owner["packet_sha256"],
        "packet_integrity_verified": audit["packet_integrity_verified"],
        "answer_id_alignment_verified": audit["answer_id_alignment_verified"],
        "adjudication_id_alignment_verified": True,
        "prediction_source_alignment_verified": True,
        "reviewer_independence_verified": False,
        "adjudication_validity_verified": False,
        "model_frozen_before_adjudication_verified": False,
        "rows": len(source_rows), "complete_adjudications": complete,
        "ambiguous_adjudications": ambiguous,
        "adjudicated_positive": positive, "adjudicated_negative": negative,
        "adjudicated_equal_to_reviewer_a": sum(final_cards[item] == original_a[item] for item in mapping),
        "adjudicated_equal_to_reviewer_b": sum(final_cards[item] == original_b[item] for item in mapping),
        "generator_label_matches_on_complete": generator_matches,
        "generator_label_agreement_rate": _rate(generator_matches, complete),
        "original_reviewer_agreement": audit["agreement"],
        "model_counts_on_complete": counts,
        "model_parsed_cards": parsed_total,
        "model_parse_coverage_all_cards": _rate(parsed_total, len(source_rows)),
        "model_precision": _rate(counts["tp"], counts["tp"] + counts["fp"]),
        "model_recall_with_abstentions_as_misses": _rate(counts["tp"], positive),
        "model_specificity_with_abstentions_as_misses": _rate(counts["tn"], negative),
        "model_strict_accuracy": _rate(counts["tp"] + counts["tn"], complete),
        "model_parse_coverage_on_complete": _rate(complete - counts["model_abstentions"], complete),
        "pairs": pair_counts,
        "author_surfaces": author_surfaces,
        "factor_surfaces": factor_surfaces,
        "model_exact_contrast_pair_rate": _rate(pair_counts["model_exact_contrast_pairs"],
                                                 pair_counts["adjudicated_contrast_pairs"]),
        "model_exact_valid_one_fact_pair_rate": _rate(
            pair_counts["model_exact_valid_one_fact_pairs"],
            pair_counts["adjudicated_valid_one_fact_pairs"]),
        "descriptive_wilson_95": {
            "recall": _wilson_95(counts["tp"], positive),
            "specificity": _wilson_95(counts["tn"], negative),
            "precision": _wilson_95(counts["tp"], counts["tp"] + counts["fp"]),
            "exact_contrast_pair_rate": _wilson_95(
                pair_counts["model_exact_contrast_pairs"], pair_counts["adjudicated_contrast_pairs"]),
            "exact_valid_one_fact_pair_rate": _wilson_95(
                pair_counts["model_exact_valid_one_fact_pairs"],
                pair_counts["adjudicated_valid_one_fact_pairs"]),
        },
        "abstract_numeric_target": _numeric_target(
            category, independent, positive, negative, counts,
            pair_counts, author_surfaces, factor_surfaces, parsed_total, len(source_rows)),
        "always_review": {"tp": positive, "fp": negative, "fn": 0, "tn": 0,
                          "exact_contrast_pairs": 0},
        "never_review": {"tp": 0, "fp": 0, "fn": positive, "tn": negative,
                         "exact_contrast_pairs": 0},
    }
    if {name: sha256(path) for name, path in files.items()} != input_hashes:
        raise ValueError("Review or prediction inputs changed during evaluation")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare frozen abstract-card predictions with blinded adjudicated declarations")
    parser.add_argument("packet_dir", type=Path)
    parser.add_argument("reviewer_a", type=Path)
    parser.add_argument("reviewer_b", type=Path)
    parser.add_argument("adjudicated", type=Path)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        result = evaluate_adjudicated(
            args.root, args.packet_dir, args.reviewer_a,
            args.reviewer_b, args.adjudicated, args.predictions)
        rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            if args.output.exists():
                raise FileExistsError("Human evaluation report already exists")
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
