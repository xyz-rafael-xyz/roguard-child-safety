"""Score the once-sealed V21 abstract transfer batch with frozen adapters."""

from __future__ import annotations

import argparse
import json
import math
import runpy
import subprocess
from pathlib import Path

from roguard import (BalancedJointD1ResearchClassifier, D1FactorResearchClassifier,
                     MMBertResearchClassifier, NLIResearchClassifier)
from roguard.nli_v21 import FIELDS, decision
from roguard.pair_eval import evaluate_pairs
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
BATCH = "batch-0038"
OUTPUT = ROOT / "eval/runs/ro-nli-v21-test-0038.json"
MODELS = {
    "v21": Path("models/ro-nli-v21-abstract"),
    "v20": Path("models/ro-mmbert-v20-balanced-joint-abstract"),
    "v18": Path("models/ro-mmbert-v18-factor-abstract"),
    "v16": Path("models/ro-mmbert-v16-abstract"),
}


def committed(path: Path) -> None:
    saved = subprocess.run(["git", "show", f"HEAD:{path.as_posix()}"], cwd=ROOT,
                           check=True, capture_output=True).stdout
    if (ROOT / path).read_bytes() != saved:
        raise ValueError(f"Commit frozen V21 test input before inference: {path}")


def pair_predictions(row: dict, item: dict) -> dict[str, dict]:
    outputs = {}
    for name in MODELS:
        predicted = (item["v16_score"] >= 0.5 if name == "v16" else
                     decision({field: item[f"{name}_scores"][field] >= 0.5
                               for field in FIELDS}))
        outputs[name] = {"id": row["id"], "expected": row["labels"],
                         "strict_parse": True, "predicted": ["D1"] if predicted else []}
    return outputs


def field_metrics(raw: list[dict], truth: list[dict]) -> dict:
    result = {field: {key: 0 for key in ("tp", "fp", "fn", "tn")} for field in FIELDS}
    for item, facts in zip(raw, truth):
        for field in FIELDS:
            predicted, expected = item["v21_scores"][field] >= 0.5, facts[field]
            key = "tp" if predicted and expected else "fp" if predicted else "fn" if expected else "tn"
            result[field][key] += 1
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nli-base-model-path", type=Path, required=True)
    parser.add_argument("--mmbert-base-model-path", type=Path, required=True)
    args = parser.parse_args()
    inputs = [Path("docs/V21_PROTOCOL.md"), Path("training/generate_nli_transfer_holdout.py"),
              Path("training/train_nli_v21.py"), Path("training/validate_generated.py"),
              Path("src/roguard/review.py"), Path("src/roguard/nli_v21.py"),
              Path("src/roguard/nli_adapter.py"), Path("src/roguard/prompt_v18.py"),
              Path("src/roguard/pair_eval.py"), Path("eval/verify_committed_v21_selection.py"),
              Path(__file__).relative_to(ROOT), Path("eval/runs/ro-nli-v21-dev-selection.json")]
    inputs.extend(Path(f"data/synthetic/{BATCH}{extension}") for extension in
                  (".jsonl", ".review.json", ".audit.json"))
    for directory in MODELS.values():
        inputs.extend(directory / name for name in
                      ("research.json", "adapter_config.json", "adapter_model.safetensors"))
    for path in inputs:
        committed(path)
    if OUTPUT.exists():
        raise FileExistsError("Preserve prior sealed V21 test")
    selected = runpy.run_path(str(ROOT / "eval/verify_committed_v21_selection.py"))["verify"](ROOT)
    rows = load_approved(ROOT, [BATCH])
    facts = runpy.run_path(str(ROOT / "training/generate_nli_transfer_holdout.py"))["build_d1_facts"](BATCH)
    truth = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
              "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
             for item in facts]
    if len(rows) != 96 or len(truth) != len(rows) or any(
            row["labels"] != (["D1"] if decision(item) else [])
            for row, item in zip(rows, truth)):
        raise ValueError("V21 sealed cards differ from typed reference states")
    nli = NLIResearchClassifier(args.nli_base_model_path, ROOT / MODELS["v21"])
    v20 = BalancedJointD1ResearchClassifier(args.mmbert_base_model_path, ROOT / MODELS["v20"])
    v18 = D1FactorResearchClassifier(args.mmbert_base_model_path, ROOT / MODELS["v18"])
    v16 = MMBertResearchClassifier(args.mmbert_base_model_path, ROOT / MODELS["v16"])
    if v16.thresholds["D1"] != 0.5:
        raise ValueError("V16 comparison cutoff differs")
    raw, outputs = [], {name: [] for name in MODELS}
    for row in rows:
        item = {"id": row["id"], "expected": row["labels"], "strict_parse": True,
                "v21_scores": nli.score(row["text"]),
                "v20_scores": v20.score(row["text"]),
                "v18_scores": v18.score(row["text"]),
                "v16_score": v16.score(row["text"], "ro", "message")["D1"]}
        if any(type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1
               for name in ("v21", "v20", "v18") for score in item[f"{name}_scores"].values()):
            raise ValueError("Invalid field score")
        raw.append(item)
        predictions = pair_predictions(row, item)
        for name in MODELS:
            outputs[name].append(predictions[name])
    reports = {name: evaluate_pairs(rows, outputs[name]) for name in MODELS}
    fields = field_metrics(raw, truth)
    d1 = reports["v21"]["per_category"][0]
    success = (reports["v21"]["exact_pairs"] >= 42 and d1["tp"] >= 42 and
               reports["v21"]["false_review_on_negatives"] <= 6 and
               reports["v21"]["exact_pairs"] > max(reports[name]["exact_pairs"]
                                                   for name in ("v20", "v18", "v16")))
    record = {
        "status": "frozen_v21_nli_transfer_test", "batch": BATCH,
        "batch_sha256": sha256(ROOT / f"data/synthetic/{BATCH}.jsonl"),
        "attestation_sha256": sha256(ROOT / f"data/synthetic/{BATCH}.review.json"),
        "audit_sha256": sha256(ROOT / f"data/synthetic/{BATCH}.audit.json"),
        "protocol_sha256": sha256(ROOT / "docs/V21_PROTOCOL.md"),
        "generator_sha256": sha256(ROOT / "training/generate_nli_transfer_holdout.py"),
        "trainer_sha256": sha256(ROOT / "training/train_nli_v21.py"),
        "runner_sha256": sha256(Path(__file__)),
        "selection_sha256": sha256(ROOT / "eval/runs/ro-nli-v21-dev-selection.json"),
        "model_hashes": {name: {
            "manifest_sha256": sha256(ROOT / directory / "research.json"),
            "config_sha256": sha256(ROOT / directory / "adapter_config.json"),
            "weight_sha256": sha256(ROOT / directory / "adapter_model.safetensors")}
            for name, directory in MODELS.items()},
        "selected_development": selected, "scores": raw, "predictions": outputs,
        "reports": reports, "field_metrics": fields,
        "preregistered_success": success,
        "limitation": "Invented Romanian abstract cards only; no child-language or Ukrainian validity.",
    }
    OUTPUT.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "pairs": {name: report["exact_pairs"] for name, report in reports.items()},
        "v21_false_reviews": reports["v21"]["false_review_on_negatives"],
        "v21_D1_tp": d1["tp"], "v21_field_metrics": fields,
        "development_gate_passed": selected["development_gate_passed"],
        "preregistered_success": success,
    }, indent=2))


if __name__ == "__main__":
    main()
