"""Recompute the frozen mixed-input study from saved scores and declared facts."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.cli import assess_json
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-v14-test-0026.json")
BATCH = "batch-0026"
ARTIFACT = Path("models/ro-mmbert-v11-abstract")
SIDECAR = Path("data/synthetic/batch-0026.contracts.jsonl")


def _strata(rows: list[dict], items: list[dict], sides: list[dict]) -> dict:
    reports = {}
    for name, has_contract in (("advisory", False), ("declared_contract", True)):
        indices = [index for index, side in enumerate(sides)
                   if (side["contract"] is not None) == has_contract]
        reports[name] = evaluate_pairs([rows[index] for index in indices],
                                       [items[index] for index in indices])
    return reports


def _discordance(rows: list[dict], candidate: list[dict], comparator: list[dict]) -> dict:
    counts = [0, 0]
    for index in range(0, len(rows), 2):
        expected = [row["labels"] for row in rows[index:index + 2]]
        left = all(item["predicted"] == label for item, label in
                   zip(candidate[index:index + 2], expected))
        right = all(item["predicted"] == label for item, label in
                    zip(comparator[index:index + 2], expected))
        counts[0] += left and not right
        counts[1] += right and not left
    discordant = sum(counts)
    probability = (min(1.0, 2 * sum(math.comb(discordant, k)
                                   for k in range(min(counts) + 1)) / 2 ** discordant)
                   if discordant else 1.0)
    return {"candidate_only_exact_pairs": counts[0],
            "comparator_only_exact_pairs": counts[1],
            "two_sided_exact_mcnemar_p": probability}


def verify(root: Path = ROOT) -> dict:
    run = json.loads((root / RUN).read_text(encoding="utf-8"))
    manifest = json.loads((root / ARTIFACT / "research.json").read_text(encoding="utf-8"))
    audit = json.loads((root / f"data/synthetic/{BATCH}.audit.json").read_text(
        encoding="utf-8"))
    rows = load_approved(root, [BATCH])
    sides = [json.loads(line) for line in (root / SIDECAR).read_text(
        encoding="utf-8").splitlines()]
    build_contracts = runpy.run_path(str(root / "training/generate_hybrid_holdout.py"))[
        "build_contracts"]
    if (run.get("status") != "frozen_v14_mixed_input_test" or
            run.get("batch") != BATCH or len(rows) != 144 or
            run.get("test_sha256") != sha256(root / f"data/synthetic/{BATCH}.jsonl") or
            run.get("sidecar_sha256") != sha256(root / SIDECAR) or
            audit.get("contract_sidecar_sha256") != run["sidecar_sha256"] or
            run.get("attestation_sha256") != sha256(root / f"data/synthetic/{BATCH}.review.json") or
            run.get("protocol_sha256") != sha256(root / "docs/V14_PROTOCOL.md") or
            run.get("adapter_weight_sha256") != manifest["adapter_weight_sha256"] or
            run.get("adapter_weight_sha256") != sha256(root / ARTIFACT / "adapter_model.safetensors") or
            run.get("artifact_manifest_sha256") != sha256(root / ARTIFACT / "research.json") or
            run.get("threshold") != manifest["shared_threshold"] or
            sides != build_contracts(BATCH) or
            [side["id"] for side in sides] != [row["id"] for row in rows]):
        raise ValueError("V14 frozen inputs, sidecars, or adapter differ")
    model = run.get("model_predictions")
    hybrid = run.get("hybrid_predictions")
    if (not isinstance(model, list) or not isinstance(hybrid, list) or
            len(model) != len(rows) or len(hybrid) != len(rows)):
        raise ValueError("V14 prediction count differs")
    for row, side, scored, chosen in zip(rows, sides, model, hybrid):
        codes = applicable_codes(row["source_kind"])
        scores = scored.get("scores")
        if (scored.get("id") != row["id"] or scored.get("expected") != row["labels"] or
                scored.get("strict_parse") is not True or not isinstance(scores, dict) or
                set(scores) != set(codes) or
                any(type(scores[code]) not in (int, float) or
                    not math.isfinite(scores[code]) or not 0 <= scores[code] <= 1
                    for code in codes) or
                scored.get("predicted") != [code for code in codes
                                            if scores[code] >= run["threshold"]] or
                chosen.get("id") != row["id"] or
                chosen.get("expected") != row["labels"] or
                chosen.get("strict_parse") is not True):
            raise ValueError(f"Invalid V14 saved score or row: {row['id']}")
        if side["contract"] is None:
            if (row["labels"] not in ([], ["D1"], ["S1"]) or
                    chosen != {**scored, "decision_basis": "abstract_model_advisory"}):
                raise ValueError(f"Invalid advisory decision: {row['id']}")
        else:
            declared = assess_json({"language": "ro", **side["contract"]})
            expected_decision = [finding["category"] for finding in declared["findings"]
                                 if finding["status"] != "pass"]
            if chosen != {"id": row["id"], "expected": row["labels"],
                          "predicted": expected_decision, "strict_parse": True,
                          "decision_basis": "caller_declared_contract"}:
                raise ValueError(f"Invalid declared-contract decision: {row['id']}")
    hybrid_report = evaluate_pairs(rows, hybrid)
    model_report = evaluate_pairs(rows, model)
    strata = _strata(rows, hybrid, sides)
    discordance = _discordance(rows, hybrid, model)
    by_code = {item["category"]: item for item in hybrid_report["per_category"]}
    success = (hybrid_report["exact_pairs"] >= 66 and
               strata["advisory"]["exact_pairs"] >= 18 and
               strata["declared_contract"]["exact_pairs"] == 48 and
               hybrid_report["false_review_on_negatives"] <= 4 and
               all(by_code[code]["tp"] >= 9 for code in by_code))
    if (run.get("hybrid_report") != hybrid_report or
            run.get("text_only_report") != model_report or
            run.get("hybrid_strata") != strata or
            run.get("paired_discordance") != discordance or
            run.get("preregistered_success") != success):
        raise ValueError("V14 stored metrics differ from saved decisions")
    return {"hybrid_exact_pairs": hybrid_report["exact_pairs"],
            "advisory_exact_pairs": strata["advisory"]["exact_pairs"],
            "declared_contract_exact_pairs": strata["declared_contract"]["exact_pairs"],
            "text_only_exact_pairs": model_report["exact_pairs"],
            "hybrid_false_reviews": hybrid_report["false_review_on_negatives"],
            "preregistered_success": success}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
