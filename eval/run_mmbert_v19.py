"""Score the sealed abstract batch 0036 with frozen V19, V18 and V16 adapters."""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
from pathlib import Path

from roguard import (D1FactorResearchClassifier, JointD1ResearchClassifier,
                     MMBertResearchClassifier)
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v18 import FIELDS, decision
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
BATCH = "batch-0036"
OUTPUT = Path("eval/runs/ro-mmbert-v19-test-0036.json")
MODEL = Path("models/ro-mmbert-v19-joint-abstract")
FACTOR = Path("models/ro-mmbert-v18-factor-abstract")
DIRECT = Path("models/ro-mmbert-v16-abstract")


def committed(path: Path) -> None:
    saved = subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], cwd=ROOT,
                           check=True, capture_output=True).stdout
    if (ROOT / path).read_bytes() != saved:
        raise ValueError(f"Commit frozen V19 test input before inference: {path}")


def from_scores(row: dict, scores: dict, field_key: str) -> dict:
    judgments = {field: scores[field] >= 0.5 for field in FIELDS}
    return {"id": row["id"], "expected": row["labels"], "strict_parse": True,
            field_key: scores, "predicted": ["D1"] if decision(judgments) else []}


def field_metrics(raw: list[dict], truth: list[dict]) -> dict:
    result = {field: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for field in FIELDS}
    for item, facts in zip(raw, truth):
        for field in FIELDS:
            predicted = item["joint_scores"][field] >= 0.5
            label = facts[field]
            key = "tp" if predicted and label else "fp" if predicted else "fn" if label else "tn"
            result[field][key] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    inputs = [Path("docs/V19_PROTOCOL.md"), Path("training/generate_joint_factor_holdout.py"),
              Path("src/roguard/joint_adapter.py"), Path("src/roguard/prompt_v18.py"),
              Path("training/train_mmbert_joint_factors.py"), Path(__file__).relative_to(ROOT),
              Path(f"data/synthetic/{BATCH}.jsonl"),
              Path(f"data/synthetic/{BATCH}.review.json"),
              Path(f"data/synthetic/{BATCH}.audit.json"),
              Path("eval/runs/ro-mmbert-v19-dev-selection.json")]
    for directory in (MODEL, FACTOR, DIRECT):
        inputs.extend((directory / "research.json", directory / "adapter_model.safetensors"))
    for path in inputs:
        committed(path)
    selected = runpy.run_path(str(ROOT / "eval/verify_committed_v19_selection.py"))["verify"](ROOT)
    if (ROOT / OUTPUT).exists():
        raise FileExistsError("Preserve prior V19 sealed test")
    rows = load_approved(ROOT, [BATCH])
    if len(rows) != 96:
        raise ValueError("V19 test requires 96 sealed abstract cards")
    generator = runpy.run_path(str(ROOT / "training/generate_joint_factor_holdout.py"))
    truth = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
              "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
             for item in generator["build_d1_facts"](BATCH)]
    if len(truth) != len(rows) or any(
            row["labels"] != (["D1"] if decision(facts) else [])
            for row, facts in zip(rows, truth)):
        raise ValueError("V19 test field facts differ from attested labels")
    joint = JointD1ResearchClassifier(args.base_model_path, ROOT / MODEL)
    factor = D1FactorResearchClassifier(args.base_model_path, ROOT / FACTOR)
    direct = MMBertResearchClassifier(args.base_model_path, ROOT / DIRECT)
    if direct.thresholds["D1"] != 0.5:
        raise ValueError("Frozen V16 cutoff differs")
    raw, candidate, factor_items, direct_items = [], [], [], []
    for row in rows:
        scores = joint.score(row["text"])
        old_scores = factor.score(row["text"])
        old_direct = direct.score(row["text"], "ro", "message")["D1"]
        item = {"id": row["id"], "expected": row["labels"], "strict_parse": True,
                "joint_scores": scores, "v18_scores": old_scores, "v16_score": old_direct}
        raw.append(item)
        candidate.append({**item, "predicted": from_scores(row, scores, "joint_scores")["predicted"]})
        factor_items.append({**item, "predicted": from_scores(row, old_scores, "v18_scores")["predicted"]})
        direct_items.append({**item, "predicted": ["D1"] if old_direct >= 0.5 else []})
    reports = {"v19": evaluate_pairs(rows, candidate),
               "v18": evaluate_pairs(rows, factor_items),
               "v16": evaluate_pairs(rows, direct_items)}
    fields = field_metrics(raw, truth)
    metrics = reports["v19"]["per_category"][0]
    success = (reports["v19"]["exact_pairs"] >= 42 and metrics["tp"] >= 42 and
               reports["v19"]["false_review_on_negatives"] <= 6 and
               reports["v19"]["exact_pairs"] > max(reports["v18"]["exact_pairs"],
                                                      reports["v16"]["exact_pairs"]))
    record = {
        "status": "frozen_v19_joint_test", "batch": BATCH,
        "batch_sha256": sha256(ROOT / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(ROOT / f"data/synthetic/{BATCH}.review.json"),
        "audit_sha256": sha256(ROOT / f"data/synthetic/{BATCH}.audit.json"),
        "protocol_sha256": sha256(ROOT / "docs/V19_PROTOCOL.md"),
        "generator_sha256": sha256(ROOT / "training/generate_joint_factor_holdout.py"),
        "trainer_sha256": sha256(ROOT / "training/train_mmbert_joint_factors.py"),
        "runner_sha256": sha256(Path(__file__)),
        "selection_sha256": sha256(ROOT / "eval/runs/ro-mmbert-v19-dev-selection.json"),
        "model_hashes": {name: {
            "manifest_sha256": sha256(ROOT / directory / "research.json"),
            "weight_sha256": sha256(ROOT / directory / "adapter_model.safetensors")}
            for name, directory in (("v19", MODEL), ("v18", FACTOR), ("v16", DIRECT))},
        "selected_development": selected,
        "scores": raw, "candidate_predictions": candidate,
        "v18_predictions": factor_items, "v16_predictions": direct_items,
        "reports": reports, "field_metrics": fields,
        "preregistered_success": success,
        "limitation": "Invented Romanian state descriptions only; no child-language or Ukrainian validity.",
    }
    (ROOT / OUTPUT).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pairs": {name: item["exact_pairs"] for name, item in reports.items()},
        "v19_false_reviews": reports["v19"]["false_review_on_negatives"],
        "v19_D1_tp": metrics["tp"],
        "field_metrics": fields,
        "preregistered_success": success,
    }, indent=2))


if __name__ == "__main__":
    main()
