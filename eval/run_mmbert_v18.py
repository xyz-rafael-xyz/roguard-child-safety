"""Compare frozen factorized D1 and V16 on sealed abstract batch 0034."""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
from pathlib import Path

from roguard import D1FactorResearchClassifier, MMBertResearchClassifier
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v18 import FIELDS, decision
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("models/ro-mmbert-v18-factor-abstract")
BASELINE = Path("models/ro-mmbert-v16-abstract")
BATCH = "batch-0034"
OUTPUT = Path("eval/runs/ro-mmbert-v18-test-0034.json")


def committed(path: Path) -> None:
    saved = subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], cwd=ROOT,
                           check=True, capture_output=True).stdout
    if (ROOT / path).read_bytes() != saved:
        raise ValueError(f"Commit frozen V18 input before test inference: {path}")


def candidate_from_scores(row: dict, factor_scores: dict | None,
                          baseline_scores: dict[str, float]) -> list[str]:
    if row["source_kind"] == "message":
        if factor_scores is None or set(factor_scores) != set(FIELDS):
            raise ValueError("D1 message needs all four factor scores")
        return ["D1"] if decision({field: factor_scores[field] >= 0.5 for field in FIELDS}) else []
    if factor_scores is not None:
        raise ValueError("Only D1 messages may receive factor scores")
    return [code for code in applicable_codes(row["source_kind"])
            if baseline_scores[code] >= 0.5]


def baseline_from_scores(row: dict, baseline_scores: dict[str, float]) -> list[str]:
    return [code for code in applicable_codes(row["source_kind"])
            if baseline_scores[code] >= 0.5]


def factor_metrics(cards: list[tuple[dict, dict]], items: list[dict]) -> dict:
    result = {field: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for field in FIELDS}
    for (_, expected), item in zip(cards, items):
        for field in FIELDS:
            predicted = item["factor_scores"][field] >= 0.5
            truth = expected[field]
            name = "tp" if predicted and truth else "fp" if predicted else "fn" if truth else "tn"
            result[field][name] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    inputs = [Path("docs/V18_PROTOCOL.md"), Path("docs/V18_PREFIT_AMENDMENT.md"),
              Path("src/roguard/prompt_v18.py"), Path("training/v18_facts.py"),
              Path("training/generate_factor_holdout.py"),
              Path("training/train_mmbert_factors.py"), Path(__file__).relative_to(ROOT),
              Path(f"data/synthetic/{BATCH}.jsonl"),
              Path(f"data/synthetic/{BATCH}.review.json"),
              Path(f"data/synthetic/{BATCH}.audit.json"),
              Path("eval/runs/ro-mmbert-v18-dev-selection.json"),
              MODEL / "adapter_model.safetensors", MODEL / "research.json",
              BASELINE / "adapter_model.safetensors", BASELINE / "research.json"]
    for path in inputs:
        committed(path)
    selection = runpy.run_path(str(ROOT / "eval/verify_committed_v18_selection.py"))["verify"](ROOT)
    if (ROOT / OUTPUT).exists():
        raise FileExistsError("Preserve prior V18 test run")
    rows = load_approved(ROOT, [BATCH])
    if len(rows) != 144:
        raise ValueError("V18 test must contain 144 cards")
    facts_module = runpy.run_path(str(ROOT / "training/v18_facts.py"))
    d1_cards = facts_module["load_d1_rows_with_facts"](ROOT, BATCH)
    factor = D1FactorResearchClassifier(args.base_model_path, ROOT / MODEL)
    baseline = MMBertResearchClassifier(args.base_model_path, ROOT / BASELINE)
    if baseline.thresholds["D1"] != 0.5:
        raise ValueError("V16 baseline cutoff changed")
    raw, candidate, comparator = [], [], []
    for row in rows:
        baseline_all = baseline.score(row["text"], "ro", row["source_kind"])
        baseline_scores = {code: baseline_all[code] for code in applicable_codes(row["source_kind"])}
        factor_scores = factor.score(row["text"]) if row["source_kind"] == "message" else None
        item = {"id": row["id"], "expected": row["labels"], "strict_parse": True,
                "baseline_scores": baseline_scores, "factor_scores": factor_scores}
        raw.append(item)
        candidate.append({**item, "predicted": candidate_from_scores(row, factor_scores, baseline_scores)})
        comparator.append({**item, "predicted": baseline_from_scores(row, baseline_scores)})
    candidate_report = evaluate_pairs(rows, candidate)
    comparator_report = evaluate_pairs(rows, comparator)
    factor_report = factor_metrics(d1_cards, [item for item in raw if item["factor_scores"] is not None])
    metrics = {entry["category"]: entry for entry in candidate_report["per_category"]}
    success = (candidate_report["exact_pairs"] >= 64 and
               metrics["D1"]["tp"] >= 42 and metrics["S1"]["tp"] >= 21 and
               candidate_report["false_review_on_negatives"] <= 6 and
               candidate_report["exact_pairs"] > comparator_report["exact_pairs"])
    record = {
        "status": "frozen_v18_factor_test", "batch": BATCH,
        "batch_sha256": sha256(ROOT / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(ROOT / f"data/synthetic/{BATCH}.review.json"),
        "protocol_sha256": sha256(ROOT / "docs/V18_PROTOCOL.md"),
        "prefit_amendment_sha256": sha256(ROOT / "docs/V18_PREFIT_AMENDMENT.md"),
        "runner_sha256": sha256(Path(__file__)),
        "selection_record_sha256": sha256(ROOT / "eval/runs/ro-mmbert-v18-dev-selection.json"),
        "v18_manifest_sha256": sha256(ROOT / MODEL / "research.json"),
        "v18_weight_sha256": sha256(ROOT / MODEL / "adapter_model.safetensors"),
        "v16_manifest_sha256": sha256(ROOT / BASELINE / "research.json"),
        "v16_weight_sha256": sha256(ROOT / BASELINE / "adapter_model.safetensors"),
        "selected_development": selection,
        "scores": raw, "candidate_predictions": candidate,
        "comparator_predictions": comparator,
        "factor_metrics": factor_report,
        "candidate_report": candidate_report, "comparator_report": comparator_report,
        "preregistered_success": success,
        "limitation": "Invented state descriptions only; no authentic child-language validity.",
    }
    (ROOT / OUTPUT).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_pairs": candidate_report["exact_pairs"],
                      "comparator_pairs": comparator_report["exact_pairs"],
                      "candidate_false_reviews": candidate_report["false_review_on_negatives"],
                      "factor_metrics": factor_report,
                      "preregistered_success": success}, indent=2))


if __name__ == "__main__":
    main()
