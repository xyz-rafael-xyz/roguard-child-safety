"""Verify the pre-author V16 comparator and abstract-study code freeze."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from roguard.human_eval import ABSTRACT_TARGET
from roguard.mmbert_study import MAX_LENGTH, MODEL_ID, REVISION
from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
RECORD = Path("eval/prospective/ro-v16-baseline-freeze.json")
FILES = frozenset({
    "models/ro-mmbert-v16-abstract/adapter_model.safetensors",
    "models/ro-mmbert-v16-abstract/adapter_config.json",
    "models/ro-mmbert-v16-abstract/research.json",
    "src/roguard/prompt_v4.py",
    "src/roguard/mmbert_adapter.py",
    "src/roguard/mmbert_study.py",
    "src/roguard/prospective_predict.py",
    "src/roguard/annotate_packet.py",
    "src/roguard/human_eval.py",
    "docs/HUMAN_VALIDATION_PROTOCOL.md",
})


def verify(root: Path = ROOT) -> dict:
    record = json.loads((root / RECORD).read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "status": "registered_baseline_before_independent_batch",
        "registered_on": "2026-09-30",
        "pre_author_git_commit": "7b5f507ab136b1d89a08a83bd9186d808e2e1267",
        "role": "fixed_research_comparator_not_deployment_model",
        "language": "ro",
        "categories": ["D1", "S1"],
        "input_scope": "independently_authored_abstract_cards_only",
        "adapter_dir": "models/ro-mmbert-v16-abstract",
        "model_id": MODEL_ID,
        "base_revision": REVISION,
        "cutoff": 0.5,
        "max_length": MAX_LENGTH,
        "independent_batch_received": False,
        "human_review_completed": False,
        "fresh_test_result_available": False,
        "real_child_language_accuracy_established": False,
    }
    if (not isinstance(record, dict) or set(record) != {*expected, "sha256"} or
            any(record.get(key) != value for key, value in expected.items()) or
            date.fromisoformat(record["registered_on"]) > date.today() or
            not isinstance(record["sha256"], dict) or set(record["sha256"]) != FILES):
        raise ValueError("Prospective comparator registration differs")
    for name in FILES:
        if sha256(root / name) != record["sha256"][name]:
            raise ValueError(f"Prospective comparator file differs: {name}")
    adapter = root / record["adapter_dir"]
    manifest = json.loads((adapter / "research.json").read_text(encoding="utf-8"))
    if (manifest.get("adapter_weight_sha256") != record["sha256"]["models/ro-mmbert-v16-abstract/adapter_model.safetensors"] or
            manifest.get("adapter_config_sha256") != record["sha256"]["models/ro-mmbert-v16-abstract/adapter_config.json"] or
            manifest.get("prompt_sha256") != record["sha256"]["src/roguard/prompt_v4.py"] or
            manifest.get("base_model") != MODEL_ID or manifest.get("base_revision") != REVISION or
            manifest.get("shared_threshold") != record["cutoff"] or
            manifest.get("trained_domain") != "abstract_symbolic_cards_only" or
            ABSTRACT_TARGET != {
                "minimum_valid_one_fact_pairs": 44,
                "minimum_valid_pairs_per_author_surface": 22,
                "minimum_valid_pairs_per_d1_factor": 10,
                "minimum_valid_pairs_per_s1_factor": 5,
                "minimum_recall": 0.90,
                "minimum_specificity": 0.90,
                "minimum_precision": 0.90,
                "minimum_exact_valid_pair_rate": 0.90,
                "minimum_exact_valid_pair_rate_per_author_surface": 0.85,
                "minimum_exact_valid_pair_rate_per_factor": 0.80,
                "required_parse_coverage": 1.0,
            }):
        raise ValueError("Prospective comparator manifest or target differs")
    return {"status": record["status"], "baseline": record["adapter_dir"],
            "files_verified": len(FILES), "categories": record["categories"],
            "independent_batch_received": False, "fresh_test_result_available": False}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
