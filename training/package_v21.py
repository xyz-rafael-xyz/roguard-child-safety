"""Package the development-selected V21 NLI adapter and every epoch score file."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from roguard.mmbert_study import verify_adapter_head
from roguard.nli_v21 import BASE_HASHES, MODEL_ID, REVISION
from roguard.review import sha256, taxonomy_sha256

ROOT = Path(__file__).resolve().parents[1]
TARGET = Path("models/ro-nli-v21-abstract")
RECORD = Path("eval/runs/ro-nli-v21-dev-selection.json")


def package(root: Path = ROOT) -> Path:
    source = root / "checkpoints/ro-nli-v21"
    record = json.loads((source / "development-selection.json").read_text(encoding="utf-8"))
    if (record.get("study") != "v21_nli_adapted_d1" or
            record.get("model") != MODEL_ID or record.get("revision") != REVISION or
            record.get("base_files_sha256") != BASE_HASHES or
            record.get("taxonomy_sha256") != taxonomy_sha256(root, "ro") or
            record.get("generator_sha256") != sha256(root / "training/generate_nli_transfer_holdout.py") or
            record.get("trainer_sha256") != sha256(root / "training/train_nli_v21.py") or
            record.get("prompt_sha256") != sha256(root / "src/roguard/nli_v21.py") or
            record.get("protocol_sha256") != sha256(root / "docs/V21_PROTOCOL.md") or
            record.get("development_gate_passed") is not False or
            len(record.get("candidates", [])) != 3):
        raise ValueError("V21 selection record differs from preregistered inputs")
    candidates = record["candidates"]
    selected = max(candidates, key=lambda item: (
        item["report"]["exact_pairs"], -item["report"]["false_review_on_negatives"],
        item["field_correct"], -item["epoch"]))
    epoch = selected["epoch"]
    if (record.get("selected_epoch") != epoch or
            record.get("selected_weight_sha256") != selected["weight_sha256"]):
        raise ValueError("V21 selected checkpoint differs from fixed rule")
    source_adapter = source / f"epoch-{epoch}"
    if sha256(source_adapter / "adapter_model.safetensors") != selected["weight_sha256"]:
        raise ValueError("V21 selected weight differs from training record")
    for item in candidates:
        dev_file = source / f"epoch-{item['epoch']}-dev.json"
        if sha256(dev_file) != item["dev_predictions_sha256"]:
            raise ValueError("V21 epoch development scores differ from training record")
    target = root / TARGET
    record_target = root / RECORD
    dev_targets = [root / f"eval/runs/ro-nli-v21-epoch-{item['epoch']}-dev.json"
                   for item in candidates]
    if any(path.exists() for path in (target, record_target, *dev_targets)):
        raise FileExistsError("Preserve prior V21 packaged selection")
    target.mkdir(parents=True)
    shutil.copyfile(source_adapter / "adapter_model.safetensors",
                    target / "adapter_model.safetensors")
    config = json.loads((source_adapter / "adapter_config.json").read_text(encoding="utf-8"))
    config["base_model_name_or_path"] = MODEL_ID
    (target / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    verify_adapter_head(target, num_labels=3, require_pooler=True)
    record_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / "development-selection.json", record_target)
    for item, destination in zip(candidates, dev_targets):
        shutil.copyfile(source / f"epoch-{item['epoch']}-dev.json", destination)
    manifest = {
        "artifact_kind": "experimental_romanian_abstract_d1_nli_adapter",
        "study": "v21_nli_adapted_d1", "base_model": MODEL_ID,
        "base_revision": REVISION,
        "adapter_weight_sha256": sha256(target / "adapter_model.safetensors"),
        "adapter_config_sha256": sha256(target / "adapter_config.json"),
        "selection_record_sha256": sha256(record_target),
        "development_scores_sha256": sha256(dev_targets[epoch - 1]),
        "selected_epoch": epoch, "field_cutoff": 0.5,
        "development_gate_passed": False,
        "evaluation_status": "selected_development_failed_gate",
        "trained_language": "ro", "trained_domain": "abstract_symbolic_cards_only",
        "prompt_sha256": sha256(root / "src/roguard/nli_v21.py"),
    }
    (target / "research.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(
        "# V21 Romanian abstract D1 NLI adapter\n\n"
        "Private research artifact selected despite a failed development gate. "
        "It applies four Romanian NLI hypotheses to invented state descriptions "
        "and the fixed D1 Boolean rule. It is not a detector for child messages "
        "or a calibrated risk model. Load only with the pinned MIT-licensed "
        "mDeBERTa NLI base.\n",
        encoding="utf-8")
    return target


if __name__ == "__main__":
    print(package())
