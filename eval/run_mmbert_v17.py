"""Freeze a D1 cutoff on new development prose, then test it on sealed prose."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

from roguard import MMBertResearchClassifier
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path("models/ro-mmbert-v16-abstract")
DEV = "batch-0031"
TEST = "batch-0032"
SELECTION = Path("eval/runs/ro-mmbert-v17-dev-selection.json")
TEST_RUN = Path("eval/runs/ro-mmbert-v17-test-0032.json")


def committed(path: Path) -> None:
    saved = subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], cwd=ROOT,
                           check=True, capture_output=True).stdout
    if (ROOT / path).read_bytes() != saved:
        raise ValueError(f"Commit frozen study input before inference: {path}")


def score_rows(rows: list[dict], backend: MMBertResearchClassifier) -> list[dict]:
    result = []
    for row in rows:
        scores = backend.score(row["text"], row["language"], row["source_kind"])
        codes = applicable_codes(row["source_kind"])
        result.append({"id": row["id"], "expected": row["labels"],
                       "strict_parse": True,
                       "scores": {code: scores[code] for code in codes}})
    return result


def decisions(rows: list[dict], scored: list[dict], d1_cutoff: float) -> list[dict]:
    result = []
    for row, item in zip(rows, scored):
        codes = applicable_codes(row["source_kind"])
        if item["id"] != row["id"] or item["expected"] != row["labels"] or set(item["scores"]) != set(codes):
            raise ValueError("Score vector does not match batch")
        predicted = [code for code in codes if item["scores"][code] >=
                     (d1_cutoff if code == "D1" else 0.5)]
        result.append({**item, "predicted": predicted})
    return result


def select_d1(rows: list[dict], scored: list[dict]) -> tuple[float, dict, bool, int]:
    d1_rows = [row for row in rows if row["source_kind"] == "message"]
    d1_scores = [item for row, item in zip(rows, scored) if row["source_kind"] == "message"]
    if len(d1_rows) != 48 or len(d1_scores) != 48:
        raise ValueError("Development batch must have 24 D1 pairs")
    values = sorted({item["scores"]["D1"] for item in d1_scores})
    thresholds = sorted({0.0, 0.5, 1.0} |
                        {(a + b) / 2 for a, b in zip(values, values[1:])})
    options = []
    for cutoff in thresholds:
        report = evaluate_pairs(d1_rows, decisions(d1_rows, d1_scores, cutoff))
        tp = next(item["tp"] for item in report["per_category"] if item["category"] == "D1")
        options.append((cutoff, report, tp))
    eligible = [option for option in options if option[2] >= 21]
    if eligible:
        cutoff, report, _ = max(eligible, key=lambda option: (
            option[1]["exact_pairs"], -option[1]["false_review_on_negatives"],
            option[2], option[0]))
    else:
        cutoff, report, _ = max(options, key=lambda option: (
            option[2], option[1]["exact_pairs"],
            -option[1]["false_review_on_negatives"], option[0]))
    return cutoff, report, bool(eligible), len(thresholds)


def verify_selection() -> dict:
    record = json.loads((ROOT / SELECTION).read_text(encoding="utf-8"))
    rows = load_approved(ROOT, [DEV])
    scored = record.get("scores")
    if (record.get("status") != "selected_development_only" or
            record.get("batch") != DEV or
            record.get("batch_sha256") != sha256(ROOT / f"data/synthetic/{DEV}.jsonl") or
            record.get("attestation_sha256") != sha256(ROOT / f"data/synthetic/{DEV}.review.json") or
            record.get("protocol_sha256") != sha256(ROOT / "docs/V17_PROTOCOL.md") or
            record.get("runner_sha256") != sha256(Path(__file__)) or
            record.get("model_manifest_sha256") != sha256(ROOT / MODEL / "research.json") or
            record.get("model_weight_sha256") != sha256(ROOT / MODEL / "adapter_model.safetensors") or
            not isinstance(scored, list) or len(scored) != len(rows)):
        raise ValueError("V17 selection or frozen inputs differ")
    for row, item in zip(rows, scored):
        if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                item.get("strict_parse") is not True or
                set(item.get("scores", {})) != set(applicable_codes(row["source_kind"])) or
                any(type(value) not in (int, float) or not math.isfinite(value) or
                    not 0 <= value <= 1 for value in item["scores"].values())):
            raise ValueError("V17 saved development scores differ")
    cutoff, d1_report, eligible, candidates = select_d1(rows, scored)
    if (record.get("d1_cutoff") != cutoff or record.get("d1_report") != d1_report or
            record.get("development_recall_floor_met") != eligible or
            record.get("candidate_count") != candidates or
            record.get("full_report") != evaluate_pairs(rows, decisions(rows, scored, cutoff))):
        raise ValueError("V17 development selection does not recompute")
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("development", "test"), required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    inputs = [Path("docs/V17_PROTOCOL.md"), Path("training/generate_advisory_calibration.py"),
              Path(__file__).relative_to(ROOT), MODEL / "research.json",
              MODEL / "adapter_model.safetensors"]
    batch = DEV if args.phase == "development" else TEST
    inputs.extend(Path(f"data/synthetic/{batch}{suffix}") for suffix in
                  (".jsonl", ".review.json", ".audit.json"))
    if args.phase == "test":
        inputs.append(SELECTION)
    for path in inputs:
        committed(path)
    output = ROOT / (SELECTION if args.phase == "development" else TEST_RUN)
    if output.exists():
        raise FileExistsError("Preserve prior V17 run")
    rows = load_approved(ROOT, [batch])
    if len(rows) != 96:
        raise ValueError("V17 batch must contain 96 cards")
    backend = MMBertResearchClassifier(args.base_model_path, ROOT / MODEL)
    if backend.thresholds["D1"] != 0.5:
        raise ValueError("V16 cutoff has changed")
    if args.phase == "development":
        scored = score_rows(rows, backend)
        cutoff, d1_report, eligible, candidates = select_d1(rows, scored)
        record = {
            "status": "selected_development_only", "batch": batch,
            "batch_sha256": sha256(ROOT / f"data/synthetic/{batch}.jsonl"),
            "attestation_sha256": sha256(ROOT / f"data/synthetic/{batch}.review.json"),
            "protocol_sha256": sha256(ROOT / "docs/V17_PROTOCOL.md"),
            "runner_sha256": sha256(Path(__file__)),
            "model_manifest_sha256": sha256(ROOT / MODEL / "research.json"),
            "model_weight_sha256": sha256(ROOT / MODEL / "adapter_model.safetensors"),
            "d1_cutoff": cutoff, "s1_cutoff": 0.5,
            "candidate_count": candidates, "development_recall_floor_met": eligible,
            "d1_report": d1_report,
            "full_report": evaluate_pairs(rows, decisions(rows, scored, cutoff)),
            "scores": scored,
        }
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"d1_cutoff": cutoff, "development_recall_floor_met": eligible,
                          "d1_pairs": d1_report["exact_pairs"],
                          "false_reviews": record["full_report"]["false_review_on_negatives"]}, indent=2))
    else:
        selection = verify_selection()
        scored = score_rows(rows, backend)
        candidate = decisions(rows, scored, selection["d1_cutoff"])
        comparator = decisions(rows, scored, 0.5)
        candidate_report = evaluate_pairs(rows, candidate)
        comparator_report = evaluate_pairs(rows, comparator)
        categories = {entry["category"]: entry for entry in candidate_report["per_category"]}
        success = (selection["development_recall_floor_met"] and
                   candidate_report["exact_pairs"] >= 42 and
                   candidate_report["false_review_on_negatives"] <= 4 and
                   all(categories[code]["tp"] >= 21 for code in ("D1", "S1")) and
                   candidate_report["exact_pairs"] > comparator_report["exact_pairs"])
        record = {
            "status": "frozen_v17_cutoff_test", "batch": batch,
            "batch_sha256": sha256(ROOT / f"data/synthetic/{batch}.jsonl"),
            "attestation_sha256": sha256(ROOT / f"data/synthetic/{batch}.review.json"),
            "protocol_sha256": sha256(ROOT / "docs/V17_PROTOCOL.md"),
            "runner_sha256": sha256(Path(__file__)),
            "selection_sha256": sha256(ROOT / SELECTION),
            "model_manifest_sha256": sha256(ROOT / MODEL / "research.json"),
            "model_weight_sha256": sha256(ROOT / MODEL / "adapter_model.safetensors"),
            "d1_cutoff": selection["d1_cutoff"], "s1_cutoff": 0.5,
            "scores": scored, "candidate_predictions": candidate,
            "comparator_predictions": comparator,
            "candidate_report": candidate_report, "comparator_report": comparator_report,
            "preregistered_success": success,
            "limitation": "Abstract Romanian field descriptions only; no authentic child-language validity.",
        }
        output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"candidate_pairs": candidate_report["exact_pairs"],
                          "comparator_pairs": comparator_report["exact_pairs"],
                          "candidate_false_reviews": candidate_report["false_review_on_negatives"],
                          "preregistered_success": success}, indent=2))


if __name__ == "__main__":
    main()
