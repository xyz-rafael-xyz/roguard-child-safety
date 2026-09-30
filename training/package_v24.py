"""Preserve the V24 native-encoder development result and selected adapter."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from roguard.mmbert_study import verify_adapter_head
from roguard.review import sha256

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("checkpoints/ro-bert-v24")
SELECTION = Path("eval/runs/ro-bert-v24-dev-selection.json")
TARGET = Path("models/ro-bert-v24-abstract")
MODEL_ID = "dumitrescustefan/bert-base-romanian-cased-v1"
REVISION = "37fb0ffb4bc4f7c4cde429626775685fb18f234f"


def package(root: Path = ROOT) -> Path:
    record_path = root / SOURCE / "development-selection.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record.get("study") != "v24_native_romanian_bert_d1" or
            record.get("model") != MODEL_ID or record.get("revision") != REVISION or
            record.get("protocol_sha256") != sha256(root / "docs/V24_PROTOCOL.md") or
            record.get("trainer_sha256") != sha256(root / "training/train_robert_v24.py") or
            record.get("prompt_sha256") != sha256(root / "src/roguard/prompt_v4.py") or
            record.get("development_gate_passed") is not False):
        raise ValueError("V24 source differs from preregistration or failure status")
    candidates = record.get("candidates")
    if not isinstance(candidates, list) or [item.get("epoch") for item in candidates] != list(range(1, 7)):
        raise ValueError("V24 epoch record is incomplete")
    best = max(candidates, key=lambda item: (
        item["summary"]["worst_surface_pairs"], item["summary"]["total_pairs"],
        -item["summary"]["false_reviews"], -item["epoch"]))
    if (record.get("selected_epoch") != best["epoch"] or
            record.get("selected_weight_sha256") != best["weight_sha256"]):
        raise ValueError("V24 selected epoch differs from preregistered rule")
    for item in candidates:
        epoch = item["epoch"]
        checkpoint = root / SOURCE / f"epoch-{epoch}"
        raw = root / SOURCE / f"epoch-{epoch}-dev.json"
        if (sha256(checkpoint / "adapter_model.safetensors") != item["weight_sha256"] or
                sha256(checkpoint / "adapter_config.json") != item["adapter_config_sha256"] or
                sha256(raw) != item["dev_predictions_sha256"]):
            raise ValueError(f"V24 epoch {epoch} artifact differs")
    destinations = [root / SELECTION]
    destinations.extend(root / f"eval/runs/ro-bert-v24-epoch-{epoch}-dev.json"
                        for epoch in range(1, 7))
    target = root / TARGET
    if target.exists() or any(path.exists() for path in destinations):
        raise FileExistsError("Preserve existing V24 package and raw score files")
    source_adapter = root / SOURCE / f"epoch-{best['epoch']}"
    verify_adapter_head(source_adapter)
    shutil.copyfile(record_path, destinations[0])
    for epoch, path in enumerate(destinations[1:], 1):
        shutil.copyfile(root / SOURCE / f"epoch-{epoch}-dev.json", path)
    target.mkdir(parents=True)
    shutil.copyfile(source_adapter / "adapter_model.safetensors",
                    target / "adapter_model.safetensors")
    config = json.loads((source_adapter / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = MODEL_ID
    (target / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "artifact_kind": "experimental_romanian_abstract_card_adapter",
        "study": "v24_native_romanian_bert_d1", "base_model": MODEL_ID,
        "base_revision": REVISION, "base_files_sha256": record["base_files_sha256"],
        "adapter_weight_sha256": sha256(target / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(target / "adapter_config.json"),
        "selection_record_sha256": sha256(root / SELECTION),
        "development_scores_sha256": sha256(destinations[best["epoch"]]),
        "selected_epoch": best["epoch"], "shared_threshold": 0.5,
        "development_gate_passed": False,
        "evaluation_status": "selected_development_only",
        "prompt_sha256": record["prompt_sha256"], "trained_language": "ro",
        "trained_domain": "abstract_symbolic_cards_only",
    }
    (target / "research.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        "# V24 Romanian BERT abstract D1 adapter\n\n"
        "Private research candidate selected on consumed development batches 0037 and 0038. "
        "Its registered development gate failed at 25/96 complete pairs with 0/24 on two "
        "surfaces. It has no new held-out test and must not screen child messages. "
        "Load only with the pinned Romanian BERT base.\n", encoding="utf-8")
    return target


if __name__ == "__main__":
    print(package())
