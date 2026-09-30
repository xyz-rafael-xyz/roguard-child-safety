"""Verify V16 development selection and packaged weights without opening test."""

from __future__ import annotations

import json
import math
from pathlib import Path

from roguard.mmbert_study import MODEL_ID, REVISION
from roguard.pair_eval import evaluate_pairs
from roguard.prompt_v4 import applicable_codes
from roguard.review import load_approved, sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
SELECTION = Path("eval/runs/ro-mmbert-v16-dev-selection.json")
DEV = Path("eval/runs/ro-mmbert-v16-epoch-1-dev.json")
ARTIFACT = Path("models/ro-mmbert-v16-abstract")


def verify(root: Path = ROOT) -> dict:
    selection = json.loads((root / SELECTION).read_text(encoding="utf-8"))
    saved = json.loads((root / DEV).read_text(encoding="utf-8"))
    artifact = root / ARTIFACT
    manifest = json.loads((artifact / "research.json").read_text(encoding="utf-8"))
    config = json.loads((artifact / "adapter_config.json").read_text(encoding="utf-8"))
    rows = load_approved(root, ["batch-0029"])
    candidates = selection.get("candidates")
    if (selection.get("status") != "development_selection" or
            selection.get("study") != "v16_advisory_prose_repair" or
            selection.get("model") != MODEL_ID or selection.get("revision") != REVISION or
            selection.get("train_batch") != "batch-0028" or
            selection.get("development_batch") != "batch-0029" or
            selection.get("train_batch_sha256") != sha256(root / "data/synthetic/batch-0028.jsonl") or
            selection.get("development_sha256") != sha256(root / "data/synthetic/batch-0029.jsonl") or
            selection.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            selection.get("protocol_sha256") != sha256(root / "docs/V16_PROTOCOL.md") or
            selection.get("prefit_amendment_sha256") != sha256(root / "docs/V16_PREFIT_AMENDMENT.md") or
            selection.get("trainer_sha256") != sha256(root / "training/train_mmbert_advisory_repair.py") or
            selection.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v4.py") or
            selection.get("source_adapter_weight_sha256") != sha256(
                root / "models/ro-mmbert-v11-abstract/adapter_model.safetensors") or
            not isinstance(candidates, list) or len(candidates) != 3 or
            sorted(item.get("epoch") for item in candidates) != [1, 2, 3]):
        raise ValueError("V16 registered selection inputs differ")
    best = max(candidates, key=lambda c: (
        c["report"]["exact_pairs"], c["report"]["exact_cards"],
        -c["report"]["false_review_on_negatives"],
        -abs(c["threshold"] - 0.5), -c["epoch"], -c["threshold"]))
    if (selection.get("selected_epoch") != best["epoch"] or
            selection.get("selected_threshold") != best["threshold"] or
            selection.get("selected_weight_sha256") != best["weight_sha256"] or
            best["dev_predictions_sha256"] != sha256(root / DEV)):
        raise ValueError("V16 selected candidate differs from development rule")
    if (manifest.get("artifact_kind") != "experimental_romanian_abstract_card_adapter" or
            manifest.get("study") != "v16_advisory_prose_repair" or
            manifest.get("base_model") != MODEL_ID or manifest.get("base_revision") != REVISION or
            manifest.get("adapter_weight_sha256") != sha256(artifact / "adapter_model.safetensors") or
            manifest.get("adapter_weight_sha256") != best["weight_sha256"] or
            manifest.get("adapter_config_sha256") != sha256(artifact / "adapter_config.json") or
            manifest.get("selection_record_sha256") != sha256(root / SELECTION) or
            manifest.get("development_scores_sha256") != sha256(root / DEV) or
            manifest.get("prompt_sha256") != selection["prompt_sha256"] or
            manifest.get("shared_threshold") != best["threshold"] or
            manifest.get("evaluation_status") != "selected_development_only" or
            config.get("base_model_name_or_path") != MODEL_ID or
            config.get("r") != 16 or config.get("lora_alpha") != 32):
        raise ValueError("V16 committed adapter differs from selection")
    items = saved.get("predictions")
    if (saved.get("batches") != ["batch-0029"] or
            saved.get("threshold") != best["threshold"] or
            not isinstance(items, list) or len(items) != len(rows)):
        raise ValueError("V16 development predictions differ")
    for row, item in zip(rows, items):
        codes = applicable_codes(row["source_kind"])
        scores = item.get("scores")
        if (item.get("id") != row["id"] or item.get("expected") != row["labels"] or
                item.get("strict_parse") is not True or
                not isinstance(scores, dict) or set(scores) != set(codes) or
                any(type(scores[code]) not in (int, float) or
                    not math.isfinite(scores[code]) or not 0 <= scores[code] <= 1
                    for code in codes) or
                item.get("predicted") != [code for code in codes
                                          if scores[code] >= best["threshold"]]):
            raise ValueError(f"V16 invalid development decision: {row['id']}")
    if saved.get("report") != evaluate_pairs(rows, items) or saved["report"] != best["report"]:
        raise ValueError("V16 development metrics differ")
    return {"selected_epoch": best["epoch"], "threshold": best["threshold"],
            "development_exact_pairs": saved["report"]["exact_pairs"],
            "selected_weight_sha256": best["weight_sha256"]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
