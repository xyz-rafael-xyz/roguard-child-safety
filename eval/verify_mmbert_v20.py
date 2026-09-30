"""Recompute frozen V20 and comparator decisions from saved scores."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v18 import FIELDS, decision
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-mmbert-v20-test-0037.json")
MODELS = {
    "v20": Path("models/ro-mmbert-v20-balanced-joint-abstract"),
    "v19": Path("models/ro-mmbert-v19-joint-abstract"),
    "v18": Path("models/ro-mmbert-v18-factor-abstract"),
    "v16": Path("models/ro-mmbert-v16-abstract"),
}


def verify(root: Path = ROOT) -> dict:
    selected = runpy.run_path(str(root / "eval/verify_committed_v20_selection.py"))["verify"](root)
    runner = runpy.run_path(str(root / "eval/run_mmbert_v20.py"))
    generator = runpy.run_path(str(root / "training/generate_balanced_joint_holdout.py"))
    rows = load_approved(root, ["batch-0037"])
    raw_truth = generator["build_d1_facts"]("batch-0037")
    truth = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
              "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
             for item in raw_truth]
    record = json.loads((root / RUN).read_text(encoding="utf-8"))
    hashes = {name: {
        "manifest_sha256": sha256(root / directory / "research.json"),
        "weight_sha256": sha256(root / directory / "adapter_model.safetensors")}
        for name, directory in MODELS.items()}
    if (len(rows) != 96 or len(truth) != len(rows) or
            record.get("status") != "frozen_v20_balanced_joint_test" or
            record.get("batch") != "batch-0037" or
            record.get("batch_sha256") != sha256(root / "data/synthetic/batch-0037.jsonl") or
            record.get("attestation_sha256") != sha256(root / "data/synthetic/batch-0037.review.json") or
            record.get("audit_sha256") != sha256(root / "data/synthetic/batch-0037.audit.json") or
            record.get("protocol_sha256") != sha256(root / "docs/V20_PROTOCOL.md") or
            record.get("generator_sha256") != sha256(root / "training/generate_balanced_joint_holdout.py") or
            record.get("trainer_sha256") != sha256(root / "training/train_mmbert_balanced_joint.py") or
            record.get("runner_sha256") != sha256(root / "eval/run_mmbert_v20.py") or
            record.get("selection_sha256") != sha256(root / "eval/runs/ro-mmbert-v20-dev-selection.json") or
            record.get("model_hashes") != hashes or
            record.get("selected_development") != selected or
            not isinstance(record.get("scores"), list) or len(record["scores"]) != len(rows)):
        raise ValueError("V20 frozen test provenance differs")
    predictions = {name: [] for name in MODELS}
    for row, facts, item in zip(rows, truth, record["scores"]):
        if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                item.get("strict_parse") is not True or
                row["labels"] != (["D1"] if decision(facts) else [])):
            raise ValueError(f"V20 saved test row differs: {row['id']}")
        for name in MODELS:
            scores = item.get(f"{name}_scores") if name != "v16" else item.get("v16_score")
            if name == "v16":
                if type(scores) not in (int, float) or not math.isfinite(scores) or not 0 <= scores <= 1:
                    raise ValueError(f"V16 score differs: {row['id']}")
                predicted = scores >= 0.5
            else:
                if (not isinstance(scores, dict) or set(scores) != set(FIELDS) or
                        any(type(value) not in (int, float) or not math.isfinite(value) or
                            not 0 <= value <= 1 for value in scores.values())):
                    raise ValueError(f"{name} field score differs: {row['id']}")
                predicted = decision({field: scores[field] >= 0.5 for field in FIELDS})
            predictions[name].append({**item, "predicted": ["D1"] if predicted else []})
    reports = {name: evaluate_pairs(rows, predictions[name]) for name in MODELS}
    fields = runner["field_metrics"](record["scores"], truth)
    d1 = reports["v20"]["per_category"][0]
    passed = (reports["v20"]["exact_pairs"] >= 42 and d1["tp"] >= 42 and
              reports["v20"]["false_review_on_negatives"] <= 6 and
              reports["v20"]["exact_pairs"] > max(reports[name]["exact_pairs"]
                                                     for name in ("v19", "v18", "v16")))
    if (record.get("predictions") != predictions or record.get("reports") != reports or
            record.get("field_metrics") != fields or record.get("preregistered_success") != passed):
        raise ValueError("V20 saved decisions or metrics differ")
    return {"pairs": {name: item["exact_pairs"] for name, item in reports.items()},
            "v20_false_reviews": reports["v20"]["false_review_on_negatives"],
            "v20_D1_tp": d1["tp"], "field_metrics": fields,
            "preregistered_success": passed}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
