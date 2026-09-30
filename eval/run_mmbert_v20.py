"""Score sealed batch 0037 with frozen V20 and three prior D1 adapters."""

from __future__ import annotations

import argparse
import json
import runpy
import subprocess
from pathlib import Path

from roguard import (BalancedJointD1ResearchClassifier, D1FactorResearchClassifier,
                     JointD1ResearchClassifier, MMBertResearchClassifier)
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v18 import FIELDS, decision
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
BATCH = "batch-0037"
OUTPUT = Path("eval/runs/ro-mmbert-v20-test-0037.json")
MODELS = {
    "v20": Path("models/ro-mmbert-v20-balanced-joint-abstract"),
    "v19": Path("models/ro-mmbert-v19-joint-abstract"),
    "v18": Path("models/ro-mmbert-v18-factor-abstract"),
    "v16": Path("models/ro-mmbert-v16-abstract"),
}


def committed(path: Path) -> None:
    saved = subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], cwd=ROOT,
                           check=True, capture_output=True).stdout
    if (ROOT / path).read_bytes() != saved:
        raise ValueError(f"Commit frozen V20 test input before inference: {path}")


def field_metrics(raw: list[dict], truth: list[dict]) -> dict:
    result = {field: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for field in FIELDS}
    for item, facts in zip(raw, truth):
        for field in FIELDS:
            predicted = item["v20_scores"][field] >= 0.5
            label = facts[field]
            key = "tp" if predicted and label else "fp" if predicted else "fn" if label else "tn"
            result[field][key] += 1
    return result


def decisions(row: dict, item: dict) -> dict[str, dict]:
    result = {}
    for name in MODELS:
        if name == "v16":
            predicted = item["v16_score"] >= 0.5
        else:
            scores = item[f"{name}_scores"]
            predicted = decision({field: scores[field] >= 0.5 for field in FIELDS})
        result[name] = {**item, "predicted": ["D1"] if predicted else []}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    inputs = [Path("docs/V20_PROTOCOL.md"), Path("training/generate_balanced_joint_holdout.py"),
              Path("training/train_mmbert_balanced_joint.py"), Path(__file__).relative_to(ROOT),
              Path("src/roguard/balanced_adapter.py"), Path("src/roguard/prompt_v18.py"),
              Path(f"data/synthetic/{BATCH}.jsonl"),
              Path(f"data/synthetic/{BATCH}.review.json"),
              Path(f"data/synthetic/{BATCH}.audit.json"),
              Path("eval/runs/ro-mmbert-v20-dev-selection.json")]
    for directory in MODELS.values():
        inputs.extend((directory / "research.json", directory / "adapter_model.safetensors"))
    for path in inputs:
        committed(path)
    selected = runpy.run_path(str(ROOT / "eval/verify_committed_v20_selection.py"))["verify"](ROOT)
    if (ROOT / OUTPUT).exists():
        raise FileExistsError("Preserve prior V20 sealed test")
    rows = load_approved(ROOT, [BATCH])
    generator = runpy.run_path(str(ROOT / "training/generate_balanced_joint_holdout.py"))
    truth = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
              "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
             for item in generator["build_d1_facts"](BATCH)]
    if (len(rows) != 96 or len(truth) != len(rows) or any(
            row["labels"] != (["D1"] if decision(facts) else [])
            for row, facts in zip(rows, truth))):
        raise ValueError("V20 sealed test differs from typed reference states")
    v20 = BalancedJointD1ResearchClassifier(args.base_model_path, ROOT / MODELS["v20"])
    v19 = JointD1ResearchClassifier(args.base_model_path, ROOT / MODELS["v19"])
    v18 = D1FactorResearchClassifier(args.base_model_path, ROOT / MODELS["v18"])
    v16 = MMBertResearchClassifier(args.base_model_path, ROOT / MODELS["v16"])
    if v16.thresholds["D1"] != 0.5:
        raise ValueError("V16 comparison cutoff differs")
    raw = []
    outputs = {name: [] for name in MODELS}
    for row in rows:
        item = {"id": row["id"], "expected": row["labels"], "strict_parse": True,
                "v20_scores": v20.score(row["text"]),
                "v19_scores": v19.score(row["text"]),
                "v18_scores": v18.score(row["text"]),
                "v16_score": v16.score(row["text"], "ro", "message")["D1"]}
        raw.append(item)
        predicted = decisions(row, item)
        for name in MODELS:
            outputs[name].append(predicted[name])
    reports = {name: evaluate_pairs(rows, outputs[name]) for name in MODELS}
    fields = field_metrics(raw, truth)
    d1 = reports["v20"]["per_category"][0]
    success = (reports["v20"]["exact_pairs"] >= 42 and d1["tp"] >= 42 and
               reports["v20"]["false_review_on_negatives"] <= 6 and
               reports["v20"]["exact_pairs"] > max(reports[name]["exact_pairs"]
                                                      for name in ("v19", "v18", "v16")))
    record = {
        "status": "frozen_v20_balanced_joint_test", "batch": BATCH,
        "batch_sha256": sha256(ROOT / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(ROOT / f"data/synthetic/{BATCH}.review.json"),
        "audit_sha256": sha256(ROOT / f"data/synthetic/{BATCH}.audit.json"),
        "protocol_sha256": sha256(ROOT / "docs/V20_PROTOCOL.md"),
        "generator_sha256": sha256(ROOT / "training/generate_balanced_joint_holdout.py"),
        "trainer_sha256": sha256(ROOT / "training/train_mmbert_balanced_joint.py"),
        "runner_sha256": sha256(Path(__file__)),
        "selection_sha256": sha256(ROOT / "eval/runs/ro-mmbert-v20-dev-selection.json"),
        "model_hashes": {name: {
            "manifest_sha256": sha256(ROOT / directory / "research.json"),
            "weight_sha256": sha256(ROOT / directory / "adapter_model.safetensors")}
            for name, directory in MODELS.items()},
        "selected_development": selected,
        "scores": raw, "predictions": outputs, "reports": reports,
        "field_metrics": fields, "preregistered_success": success,
        "limitation": "Invented Romanian state descriptions only; no child-language or Ukrainian validity.",
    }
    (ROOT / OUTPUT).write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pairs": {name: item["exact_pairs"] for name, item in reports.items()},
        "v20_false_reviews": reports["v20"]["false_review_on_negatives"],
        "v20_D1_tp": d1["tp"], "field_metrics": fields,
        "preregistered_success": success,
    }, indent=2))


if __name__ == "__main__":
    main()
