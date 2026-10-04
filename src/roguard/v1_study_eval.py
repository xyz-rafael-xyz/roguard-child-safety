"""Score candidate-v1 abstract cards against separate blinded adjudication."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from .review import sha256
from .v1_contract import assess_v1_json
from .v1_study_kit import ABSTRACT_TARGET, FACTORS, verify_v1_study_freeze
from .v1_study_packet import _validate_workbooks

TARGET = ABSTRACT_TARGET
D1_FIELDS = frozenset({
    "source_role", "safety_or_support_anchor",
    "indirect_or_repeated_support_pattern", "explicit_support_request",
    "direct_safety_statement", "retracted_or_denied",
})
S1_FIELDS = frozenset({"applicable_fields", "passed_fields"})
S1_ALLOWED = frozenset(FACTORS["S1"])


def _read_json(path: Path) -> object:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular file: {path}")

    def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def _read_jsonl(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected a regular JSONL file: {path}")
    raw = path.read_text(encoding="utf-8")
    if not raw.endswith("\n") or not raw.strip():
        raise ValueError("JSONL file needs complete nonempty lines")
    rows = []
    for line in raw.splitlines():
        if not line.strip():
            raise ValueError("JSONL file has an empty line")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("JSONL rows must be objects")
        rows.append(value)
    return rows


def _packet(path: Path, language: str, category: str) -> list[dict]:
    rows = _read_jsonl(path)
    if len(rows) != 96:
        raise ValueError("V1 study packet must contain 96 cards")
    ids = []
    for row in rows:
        if (set(row) != {"item_id", "language", "category", "source_kind", "abstract_card"} or
                not isinstance(row["item_id"], str) or not row["item_id"].startswith("item_") or
                row["language"] != language or row["category"] != category or
                row["source_kind"] != ("message" if category == "D1" else "response") or
                not isinstance(row["abstract_card"], str) or
                not row["abstract_card"].strip()):
            raise ValueError("Invalid v1 blind packet row")
        ids.append(row["item_id"])
    if len(set(ids)) != 96:
        raise ValueError("Blind packet has repeated IDs")
    return rows


def _decision(language: str, category: str, card: object) -> tuple[bool | None, dict]:
    if not isinstance(card, dict):
        raise ValueError("An answer card must be an object")
    key = "disclosure" if category == "D1" else "support"
    if set(card) != {"language", key} or card["language"] != language:
        raise ValueError("Answer card language or category differs")
    facts = card[key]
    if not isinstance(facts, dict):
        raise ValueError("Answer facts must be an object")
    if category == "D1":
        if set(facts) != D1_FIELDS:
            raise ValueError("D1 answer fields differ from v1")
    else:
        if set(facts) != S1_FIELDS:
            raise ValueError("S1 answer fields differ from v1")
        for name in S1_FIELDS:
            value = facts[name]
            if (value is not None and
                    (not isinstance(value, list) or len(value) != len(set(value)) or
                     not set(value) <= S1_ALLOWED)):
                raise ValueError("S1 answer has an invalid field set")
    report = assess_v1_json(card)
    decision = report["checks"][0]["decision_code"]
    if decision in {"INSUFFICIENT_CONTEXT", "INSUFFICIENT_RESPONSE_CONTRACT"}:
        return None, facts
    return decision in {"REVIEW_SUPPORT_SIGNAL", "REVIEW_RESPONSE"}, facts


def _answers(path: Path, packet_rows: list[dict], language: str,
             category: str, *, ordered: bool) -> dict[str, tuple[bool | None, dict]]:
    rows = _read_jsonl(path)
    if len(rows) != 96:
        raise ValueError("Answer stream needs one row per card")
    packet_ids = [row["item_id"] for row in packet_rows]
    ids = [row.get("item_id") for row in rows]
    if (len(set(ids)) != 96 or
            (ids != packet_ids if ordered else set(ids) != set(packet_ids))):
        raise ValueError("Answer IDs differ from the assigned blind packet")
    result = {}
    for row in rows:
        if set(row) != {"item_id", "card"}:
            raise ValueError("Answer rows may contain only ID and structured card")
        result[row["item_id"]] = _decision(language, category, row["card"])
    return result


def _pair_factor(category: str, negative: tuple[bool | None, dict],
                 positive: tuple[bool | None, dict]) -> str | None:
    if negative[0] is not False or positive[0] is not True:
        return None
    left, right = negative[1], positive[1]
    if category == "D1":
        changed = [name for name in FACTORS["D1"] if left[name] != right[name]]
        return changed[0] if len(changed) == 1 and left["retracted_or_denied"] == right["retracted_or_denied"] else None
    applicable_left, applicable_right = left["applicable_fields"], right["applicable_fields"]
    passed_left, passed_right = left["passed_fields"], right["passed_fields"]
    if None in (applicable_left, applicable_right, passed_left, passed_right):
        return None
    if set(applicable_left) != set(applicable_right):
        return None
    removed = set(passed_left) - set(passed_right)
    added = set(passed_right) - set(passed_left)
    return next(iter(removed)) if len(removed) == 1 and not added else None


def _predictions(path: Path, packet_rows: list[dict], language: str,
                 category: str, model_id: str, packet_hash: str) -> dict[str, bool | None]:
    value = _read_json(path)
    if (not isinstance(value, dict) or set(value) != {
            "schema_version", "packet_sha256", "category", "language", "model_id", "predictions"} or
            value["schema_version"] != 2 or value["packet_sha256"] != packet_hash or
            value["language"] != language or value["category"] != category or
            value["model_id"] != model_id or not isinstance(value["predictions"], list) or
            len(value["predictions"]) != 96):
        raise ValueError("Prediction file differs from the frozen v1 cell and packet")
    result = {}
    for expected, row in zip(packet_rows, value["predictions"]):
        if (not isinstance(row, dict) or set(row) != {"item_id", "predicted", "strict_parse"} or
                row["item_id"] != expected["item_id"] or
                type(row["strict_parse"]) is not bool or
                row["predicted"] not in ([], [category]) or
                (not row["strict_parse"] and row["predicted"])):
            raise ValueError("Prediction row differs from blind packet order or format")
        result[row["item_id"]] = (category in row["predicted"] if row["strict_parse"] else None)
    return result


def _rate(a: int, b: int) -> float | None:
    return a / b if b else None


def evaluate_v1_study(root: Path, owner_dir: Path, packet_seal: Path,
                      reviewer_a: Path, reviewer_b: Path, adjudicated: Path,
                      predictions_path: Path) -> dict:
    """Recompute the preregistered abstract target from exact private inputs."""
    root = root.resolve()
    owner_dir = owner_dir.resolve()
    owner_path = owner_dir / "OWNER-MAP.json"
    owner = _read_json(owner_path)
    if not isinstance(owner, dict) or set(owner) != {
            "schema_version", "study_kind", "language", "category",
            "candidate_taxonomy_sha256", "freeze_path", "freeze_sha256",
            "packet_sha256", "author_workbook_sha256", "content_review_sha256",
            "pairs"}:
        raise ValueError("Owner map has an invalid v1 study shape")
    language, category = owner["language"], owner["category"]
    if (owner["schema_version"] != 1 or
            owner["study_kind"] != "roguard_v1_candidate_independent_abstract" or
            language not in ("ro", "uk") or category not in FACTORS or
            not isinstance(owner["freeze_path"], str) or
            not isinstance(owner["freeze_sha256"], str)):
        raise ValueError("Owner map has an invalid v1 study scope")
    seal = _read_json(packet_seal)
    if (not isinstance(seal, dict) or set(seal) != {
            "schema_version", "study_kind", "language", "category",
            "candidate_taxonomy_sha256", "freeze_path", "freeze_sha256",
            "owner_map_sha256", "packet_sha256", "source_cards_sha256",
            "content_review_sha256", "no_card_text_or_intended_labels_in_seal",
            "human_chronology_verified_by_software"} or
            seal["schema_version"] != 1 or seal["study_kind"] != owner["study_kind"] or
            seal["language"] != language or seal["category"] != category or
            seal["candidate_taxonomy_sha256"] != owner["candidate_taxonomy_sha256"] or
            seal["freeze_path"] != owner["freeze_path"] or
            seal["freeze_sha256"] != owner["freeze_sha256"] or
            seal["owner_map_sha256"] != sha256(owner_path) or
            seal["packet_sha256"] != owner["packet_sha256"] or
            seal["no_card_text_or_intended_labels_in_seal"] is not True or
            seal["human_chronology_verified_by_software"] is not False):
        raise ValueError("Packet seal differs from the private owner map")
    frozen = verify_v1_study_freeze(root, Path(owner["freeze_path"]), language, category)
    if (owner["freeze_sha256"] != frozen["sha256"] or
            owner["candidate_taxonomy_sha256"] != frozen["candidate_taxonomy_sha256"] or
            not isinstance(owner["packet_sha256"], dict) or
            set(owner["packet_sha256"]) != {"reviewer-a", "reviewer-b"} or
            not isinstance(owner["author_workbook_sha256"], dict) or
            set(owner["author_workbook_sha256"]) != {
                f"{language.upper()}-A1", f"{language.upper()}-A2"} or
            any(sha256(owner_dir / f"{author}.json") != digest
                for author, digest in owner["author_workbook_sha256"].items()) or
            owner["content_review_sha256"] != sha256(owner_dir / "CONTENT-REVIEW.json")):
        raise ValueError("Owner map differs from candidate/model freeze")
    kit_owner, source_rows, provenance = _validate_workbooks(root, owner_dir)
    source_path = owner_dir / "SOURCE-CARDS.jsonl"
    source_file_rows = _read_jsonl(source_path)
    if (len(source_file_rows) != 96 or source_file_rows != source_rows or
            kit_owner["freeze_sha256"] != frozen["sha256"] or
            provenance["workbook_sha256"] != owner["author_workbook_sha256"] or
            seal["source_cards_sha256"] != sha256(source_path) or
            seal["content_review_sha256"] != owner["content_review_sha256"]):
        raise ValueError("Sealed source cards differ from author workbooks")
    content_review = _read_json(owner_dir / "CONTENT-REVIEW.json")
    if (not isinstance(content_review, dict) or
            content_review.get("reviewer_id") != f"{language.upper()}-C" or
            content_review.get("source_cards_sha256") != sha256(source_path) or
            content_review.get("overlap_sha256") != sha256(owner_dir / "OVERLAP.json") or
            content_review.get("owner_assignments_sha256") !=
            sha256(owner_dir / "OWNER-ASSIGNMENTS.json") or
            content_review.get("author_workbook_sha256") != provenance["workbook_sha256"] or
            any(content_review.get(flag) is not True for flag in (
                "abstract_only_checked", "no_realistic_content_checked",
                "language_checked", "one_fact_pair_logic_checked",
                "factor_balance_checked", "prior_and_cross_pair_overlap_checked"))):
        raise ValueError("Sealed content review does not cover these source cards")
    packet_a = owner_dir / "reviewer-a.jsonl"
    packet_b = owner_dir / "reviewer-b.jsonl"
    if (sha256(packet_a) != owner["packet_sha256"]["reviewer-a"] or
            sha256(packet_b) != owner["packet_sha256"]["reviewer-b"]):
        raise ValueError("Blind packet bytes changed")
    rows_a = _packet(packet_a, language, category)
    rows_b = _packet(packet_b, language, category)
    card_map_a = {row["item_id"]: row["abstract_card"] for row in rows_a}
    card_map_b = {row["item_id"]: row["abstract_card"] for row in rows_b}
    if card_map_a != card_map_b:
        raise ValueError("Reviewers received different card sets")
    answers_a = _answers(reviewer_a, rows_a, language, category, ordered=True)
    answers_b = _answers(reviewer_b, rows_b, language, category, ordered=True)
    final = _answers(adjudicated, rows_a, language, category, ordered=False)
    predictions = _predictions(predictions_path, rows_a, language, category,
                               frozen["model_id"], owner["packet_sha256"]["reviewer-a"])
    pairs = owner["pairs"]
    if not isinstance(pairs, list) or len(pairs) != 48:
        raise ValueError("Owner map needs 48 pairs")
    used_ids, author_counts = set(), Counter()
    factor_counts = Counter()
    valid_pairs, exact_pairs = 0, 0
    author_surfaces = defaultdict(lambda: Counter(valid=0, exact=0))
    factor_surfaces = defaultdict(lambda: Counter(valid=0, exact=0))
    for index, pair in enumerate(pairs):
        if (not isinstance(pair, dict) or set(pair) != {
                "author_id", "factor", "negative_item_id", "positive_item_id"} or
                pair["author_id"] not in {f"{language.upper()}-A1", f"{language.upper()}-A2"} or
                pair["factor"] not in FACTORS[category] or
                pair["negative_item_id"] == pair["positive_item_id"] or
                pair["negative_item_id"] not in final or
                pair["positive_item_id"] not in final):
            raise ValueError("Owner pair map is incomplete")
        author, factor = pair["author_id"], pair["factor"]
        ids = {pair["negative_item_id"], pair["positive_item_id"]}
        if used_ids & ids:
            raise ValueError("Owner map reuses a card in multiple pairs")
        used_ids |= ids
        source_negative, source_positive = source_rows[index * 2:index * 2 + 2]
        if (source_negative["author_id"] != author or
                source_positive["author_id"] != author or
                source_negative["factor"] != factor or
                source_positive["factor"] != factor or
                card_map_a[pair["negative_item_id"]] != source_negative["text"] or
                card_map_a[pair["positive_item_id"]] != source_positive["text"]):
            raise ValueError("Owner pair mapping differs from frozen author cards")
        author_counts[author] += 1
        factor_counts[factor] += 1
        observed = _pair_factor(category, final[pair["negative_item_id"]],
                                final[pair["positive_item_id"]])
        if observed == factor:
            valid_pairs += 1
            author_surfaces[author]["valid"] += 1
            factor_surfaces[factor]["valid"] += 1
            if (predictions[pair["negative_item_id"]] is False and
                    predictions[pair["positive_item_id"]] is True):
                exact_pairs += 1
                author_surfaces[author]["exact"] += 1
                factor_surfaces[factor]["exact"] += 1
    if (len(used_ids) != 96 or sorted(author_counts.values()) != [24, 24] or
            set(factor_counts) != set(FACTORS[category])):
        raise ValueError("Owner map does not cover balanced authors and factors")
    counts = Counter(tp=0, fp=0, fn=0, tn=0, incomplete=0, parse_failures=0)
    for item_id, (truth, _) in final.items():
        prediction = predictions[item_id]
        if prediction is None:
            counts["parse_failures"] += 1
        if truth is None:
            counts["incomplete"] += 1
        elif truth is True:
            counts["tp" if prediction is True else "fn"] += 1
        else:
            counts["tn" if prediction is False else "fp"] += 1
    positive = counts["tp"] + counts["fn"]
    negative = counts["tn"] + counts["fp"]
    precision = _rate(counts["tp"], counts["tp"] + counts["fp"])
    recall = _rate(counts["tp"], positive)
    specificity = _rate(counts["tn"], negative)
    parse_coverage = _rate(96 - counts["parse_failures"], 96)
    exact_rate = _rate(exact_pairs, valid_pairs)
    minimum_factor = (TARGET["minimum_valid_pairs_per_d1_factor"] if category == "D1"
                      else TARGET["minimum_valid_pairs_per_s1_factor"])
    enough = (valid_pairs >= TARGET["minimum_valid_pairs"] and
              all(author_surfaces[author]["valid"] >= TARGET["minimum_valid_pairs_per_author"]
                  for author in author_counts) and
              all(factor_surfaces[factor]["valid"] >= minimum_factor
                  for factor in FACTORS[category]))
    meets = (enough and precision is not None and precision >= TARGET["minimum_precision"] and
             recall is not None and recall >= TARGET["minimum_recall"] and
             specificity is not None and specificity >= TARGET["minimum_specificity"] and
             exact_rate is not None and exact_rate >= TARGET["minimum_exact_valid_pair_rate"] and
             parse_coverage == TARGET["required_parse_coverage"] and
             all(_rate(surface["exact"], surface["valid"]) is not None and
                 _rate(surface["exact"], surface["valid"]) >=
                 TARGET["minimum_exact_valid_pair_rate_per_author"]
                 for surface in author_surfaces.values()) and
             all(_rate(surface["exact"], surface["valid"]) is not None and
                 _rate(surface["exact"], surface["valid"]) >=
                 TARGET["minimum_exact_valid_pair_rate_per_factor"]
                 for surface in factor_surfaces.values()))
    disagreements = sum(answers_a[item_id][0] != answers_b[item_id][0] for item_id in final)
    return {
        "schema_version": 1, "scope": "v1_candidate_independent_abstract_cards_only",
        "language": language, "category": category,
        "candidate_taxonomy_sha256": frozen["candidate_taxonomy_sha256"],
        "freeze_sha256": frozen["sha256"], "model_id": frozen["model_id"],
        "status": ("inconclusive_insufficient_valid_pairs" if not enough else
                   "meets_numeric_target" if meets else "misses_numeric_target"),
        "target": dict(TARGET), "adjudicated_valid_one_fact_pairs": valid_pairs,
        "model_exact_valid_one_fact_pairs": exact_pairs,
        "model_exact_valid_one_fact_pair_rate": exact_rate,
        "model_precision": precision, "model_recall": recall,
        "model_specificity": specificity, "model_parse_coverage": parse_coverage,
        "confusion": dict(counts), "annotator_decision_disagreements": disagreements,
        "author_surfaces": [
            {"author_role": author, "valid_pairs": surface["valid"],
             "exact_pairs": surface["exact"],
             "exact_pair_rate": _rate(surface["exact"], surface["valid"])}
            for author, surface in sorted(author_surfaces.items())
        ],
        "factor_surfaces": [
            {"factor": factor, "valid_pairs": factor_surfaces[factor]["valid"],
             "exact_pairs": factor_surfaces[factor]["exact"],
             "exact_pair_rate": _rate(factor_surfaces[factor]["exact"],
                                      factor_surfaces[factor]["valid"])}
            for factor in FACTORS[category]
        ],
        "input_sha256": {
            "owner_map": sha256(owner_path), "packet_seal": sha256(packet_seal),
            "reviewer_a": sha256(reviewer_a),
            "reviewer_b": sha256(reviewer_b), "adjudicated": sha256(adjudicated),
            "predictions": sha256(predictions_path),
        },
        "human_identity_or_independence_verified_by_software": False,
        "model_execution_verified_by_software": False,
        "real_child_message_accuracy_established": False,
        "operational_release_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit a corrected-v1 abstract-card study")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--owner-dir", type=Path, required=True)
    parser.add_argument("--packet-seal", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--adjudicated", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate_v1_study(args.root, args.owner_dir, args.packet_seal, args.reviewer_a,
                                   args.reviewer_b, args.adjudicated, args.predictions)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
