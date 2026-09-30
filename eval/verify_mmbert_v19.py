"""Recompute the frozen V19 test from saved scores without loading a model."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v18 import FIELDS, decision
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-v19-test-0036.json")
MODEL = Path("models/ro-mmbert-v19-joint-abstract")
FACTOR = Path("models/ro-mmbert-v18-factor-abstract")
DIRECT = Path("models/ro-mmbert-v16-abstract")


def verify(root: Path = ROOT) -> dict:
    selected = runpy.run_path(str(root / "eval/verify_committed_v19_selection.py"))["verify"](root)
    runner = runpy.run_path(str(root / "eval/run_mmbert_v19.py"))
    generator = runpy.run_path(str(root / "training/generate_joint_factor_holdout.py"))
    rows = load_approved(root, ["batch-0036"])
    raw_truth = generator["build_d1_facts"]("batch-0036")
    truth = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
              "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
             for item in raw_truth]
    record = json.loads((root / RUN).read_text(encoding="utf-8"))
    hashes = {name: {
        "manifest_sha256": sha256(root / directory / "research.json"),
        "weight_sha256": sha256(root / directory / "adapter_model.safetensors")}
        for name, directory in (("v19", MODEL), ("v18", FACTOR), ("v16", DIRECT))}
    if (len(rows) != 96 or len(truth) != len(rows) or
            record.get("status") != "frozen_v19_joint_test" or record.get("batch") != "batch-0036" or
            record.get("batch_sha256") != sha256(root / "data/synthetic/batch-0036.jsonl") or
            record.get("attestation_sha256") != sha256(root / "data/synthetic/batch-0036.review.json") or
            record.get("audit_sha256") != sha256(root / "data/synthetic/batch-0036.audit.json") or
            record.get("protocol_sha256") != sha256(root / "docs/V19_PROTOCOL.md") or
            record.get("generator_sha256") != sha256(root / "training/generate_joint_factor_holdout.py") or
            record.get("trainer_sha256") != sha256(root / "training/train_mmbert_joint_factors.py") or
            record.get("runner_sha256") != sha256(root / "eval/run_mmbert_v19.py") or
            record.get("selection_sha256") != sha256(root / "eval/runs/ro-mmbert-v19-dev-selection.json") or
            record.get("model_hashes") != hashes or
            record.get("selected_development") != selected or
            not isinstance(record.get("scores"), list) or len(record["scores"]) != len(rows)):
        raise ValueError("V19 frozen test inputs differ")
    outputs = {"v19": [], "v18": [], "v16": []}
    for row, facts, item in zip(rows, truth, record["scores"]):
        joint, factor, direct = item.get("joint_scores"), item.get("v18_scores"), item.get("v16_score")
        if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                item.get("strict_parse") is not True or
                not isinstance(joint, dict) or set(joint) != set(FIELDS) or
                not isinstance(factor, dict) or set(factor) != set(FIELDS) or
                any(type(value) not in (int, float) or not math.isfinite(value) or
                    not 0 <= value <= 1 for value in (*joint.values(), *factor.values(), direct)) or
                row["labels"] != (["D1"] if decision(facts) else [])):
            raise ValueError(f"V19 saved test score differs: {row['id']}")
        for name, scores in (("v19", joint), ("v18", factor)):
            predicted = decision({field: scores[field] >= 0.5 for field in FIELDS})
            outputs[name].append({**item, "predicted": ["D1"] if predicted else []})
        outputs["v16"].append({**item, "predicted": ["D1"] if direct >= 0.5 else []})
    reports = {name: evaluate_pairs(rows, items) for name, items in outputs.items()}
    fields = runner["field_metrics"](record["scores"], truth)
    d1 = reports["v19"]["per_category"][0]
    passed = (reports["v19"]["exact_pairs"] >= 42 and d1["tp"] >= 42 and
              reports["v19"]["false_review_on_negatives"] <= 6 and
              reports["v19"]["exact_pairs"] > max(reports["v18"]["exact_pairs"],
                                                     reports["v16"]["exact_pairs"]))
    if (record.get("candidate_predictions") != outputs["v19"] or
            record.get("v18_predictions") != outputs["v18"] or
            record.get("v16_predictions") != outputs["v16"] or
            record.get("reports") != reports or record.get("field_metrics") != fields or
            record.get("preregistered_success") != passed):
        raise ValueError("V19 saved decisions or reports differ")
    return {"pairs": {name: value["exact_pairs"] for name, value in reports.items()},
            "v19_false_reviews": reports["v19"]["false_review_on_negatives"],
            "v19_D1_tp": d1["tp"], "field_metrics": fields,
            "preregistered_success": passed}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
