"""Recompute the sealed V21 failure from saved scores and attested states."""

from __future__ import annotations

import json
import math
import runpy
from pathlib import Path

from roguard.nli_v21 import FIELDS, decision
from roguard.pair_eval import evaluate_pairs
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = Path("eval/runs/ro-nli-v21-test-0038.json")
MODELS = {
    "v21": Path("models/ro-nli-v21-abstract"),
    "v20": Path("models/ro-mmbert-v20-balanced-joint-abstract"),
    "v18": Path("models/ro-mmbert-v18-factor-abstract"),
    "v16": Path("models/ro-mmbert-v16-abstract"),
}


def verify(root: Path = ROOT) -> dict:
    selected = runpy.run_path(str(root / "eval/verify_committed_v21_selection.py"))["verify"](root)
    runner = runpy.run_path(str(root / "eval/run_nli_v21.py"))
    generator = runpy.run_path(str(root / "training/generate_nli_transfer_holdout.py"))
    rows = load_approved(root, ["batch-0038"])
    raw_truth = generator["build_d1_facts"]("batch-0038")
    truth = [{"minor_source": item["role"] == "minor", "support_anchor": item["anchor"],
              "indirect_pattern": item["pattern"], "explicit_request": item["explicit"]}
             for item in raw_truth]
    record = json.loads((root / RUN).read_text(encoding="utf-8"))
    hashes = {name: {
        "manifest_sha256": sha256(root / directory / "research.json"),
        "config_sha256": sha256(root / directory / "adapter_config.json"),
        "weight_sha256": sha256(root / directory / "adapter_model.safetensors")}
        for name, directory in MODELS.items()}
    if (len(rows) != 96 or len(truth) != len(rows) or
            record.get("status") != "frozen_v21_nli_transfer_test" or
            record.get("batch") != "batch-0038" or
            record.get("batch_sha256") != sha256(root / "data/synthetic/batch-0038.jsonl") or
            record.get("attestation_sha256") != sha256(root / "data/synthetic/batch-0038.review.json") or
            record.get("audit_sha256") != sha256(root / "data/synthetic/batch-0038.audit.json") or
            record.get("protocol_sha256") != sha256(root / "docs/V21_PROTOCOL.md") or
            record.get("generator_sha256") != sha256(root / "training/generate_nli_transfer_holdout.py") or
            record.get("trainer_sha256") != sha256(root / "training/train_nli_v21.py") or
            record.get("runner_sha256") != sha256(root / "eval/run_nli_v21.py") or
            record.get("selection_sha256") != sha256(root / "eval/runs/ro-nli-v21-dev-selection.json") or
            record.get("model_hashes") != hashes or
            record.get("selected_development") != selected or
            not isinstance(record.get("scores"), list) or len(record["scores"]) != len(rows)):
        raise ValueError("V21 sealed test provenance differs")
    predictions = {name: [] for name in MODELS}
    for row, facts, item in zip(rows, truth, record["scores"]):
        if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                item.get("strict_parse") is not True or
                row["labels"] != (["D1"] if decision(facts) else [])):
            raise ValueError(f"V21 saved test row differs: {row['id']}")
        for name in MODELS:
            scores = item.get(f"{name}_scores") if name != "v16" else item.get("v16_score")
            if name == "v16":
                if type(scores) not in (int, float) or not math.isfinite(scores) or not 0 <= scores <= 1:
                    raise ValueError(f"V16 score differs: {row['id']}")
            elif (not isinstance(scores, dict) or set(scores) != set(FIELDS) or
                  any(type(value) not in (int, float) or not math.isfinite(value) or
                      not 0 <= value <= 1 for value in scores.values())):
                raise ValueError(f"{name} field score differs: {row['id']}")
        predictions_for_card = runner["pair_predictions"](row, item)
        for name in MODELS:
            predictions[name].append(predictions_for_card[name])
    reports = {name: evaluate_pairs(rows, predictions[name]) for name in MODELS}
    fields = runner["field_metrics"](record["scores"], truth)
    d1 = next(item for item in reports["v21"]["per_category"] if item["category"] == "D1")
    passed = (reports["v21"]["exact_pairs"] >= 42 and d1["tp"] >= 42 and
              reports["v21"]["false_review_on_negatives"] <= 6 and
              reports["v21"]["exact_pairs"] > max(reports[name]["exact_pairs"]
                                                  for name in ("v20", "v18", "v16")))
    if (record.get("predictions") != predictions or record.get("reports") != reports or
            record.get("field_metrics") != fields or record.get("preregistered_success") != passed):
        raise ValueError("V21 saved decisions or metrics differ")
    return {"pairs": {name: report["exact_pairs"] for name, report in reports.items()},
            "v21_false_reviews": reports["v21"]["false_review_on_negatives"],
            "v21_D1_tp": d1["tp"], "field_metrics": fields,
            "development_gate_passed": selected["development_gate_passed"],
            "preregistered_success": passed}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
