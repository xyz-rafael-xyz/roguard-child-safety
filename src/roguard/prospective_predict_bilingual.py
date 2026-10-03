"""Freeze RO/UK V25 comparator decisions from an opaque, label-free packet."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

from .annotate_packet import _packet_rows
from .review import sha256

ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = re.compile(r"[A-Za-z0-9._-]{1,80}\Z")


def check_cell_wrapper(root: Path, wrapper: Path, language: str, category: str) -> tuple[Path, float, str]:
    """Bind one V25 cell to its copied frozen weight and source artifact."""
    if wrapper.is_symlink():
        raise ValueError("Cell wrapper must be a regular directory")
    root, wrapper = root.resolve(), wrapper.resolve()
    if not wrapper.is_relative_to(root / "models"):
        raise ValueError("Cell wrapper must be under repository models/")
    record = json.loads((wrapper / "research.json").read_text(encoding="utf-8"))
    cell = f"{language}:{category}"
    if (not isinstance(record, dict) or set(record) != {
            "schema_version", "study", "status", "input_scope",
            "trained_language", "category", "source_adapter_dir",
            "shared_threshold", "adapter_weight_sha256",
            "adapter_config_sha256", "source_manifest_sha256",
            "eligible_for_live_child_message_screening"} or
            record.get("schema_version") != 1 or
            record.get("study") != "v25_independent_abstract_comparator" or
            record.get("status") != "source_artifact_frozen_for_independent_study" or
            record.get("input_scope") != "independently_authored_abstract_cards_only" or
            record.get("eligible_for_live_child_message_screening") is not False or
            record.get("trained_language") != language or
            record.get("category") != category or
            record.get("source_adapter_dir") != "models/bi-mmbert-v25-abstract" or
            type(record.get("shared_threshold")) not in (int, float) or
            not math.isfinite(record["shared_threshold"]) or
            not 0 <= record["shared_threshold"] <= 1):
        raise ValueError("Cell wrapper scope or cutoff differs")
    weight = wrapper / "adapter_model.safetensors"
    config = wrapper / "adapter_config.json"
    source = root / record["source_adapter_dir"]
    original = json.loads((source / "research.json").read_text(encoding="utf-8"))
    if (weight.is_symlink() or config.is_symlink() or
            record.get("adapter_weight_sha256") != sha256(weight) or
            record.get("adapter_config_sha256") != sha256(config) or
            record.get("source_manifest_sha256") != sha256(source / "research.json") or
            original.get("weight_sha256") != sha256(source / "adapter_model.safetensors") or
            original.get("adapter_config_sha256") != sha256(source / "adapter_config.json") or
            record["adapter_weight_sha256"] != original["weight_sha256"] or
            record["adapter_config_sha256"] != original["adapter_config_sha256"] or
            not isinstance(original.get("thresholds"), dict) or
            original["thresholds"].get(cell) != record["shared_threshold"]):
        raise ValueError("Cell wrapper differs from the frozen V25 artifact")
    model_id = "mmbert_" + record["adapter_weight_sha256"]
    return source, record["shared_threshold"], model_id


def freeze_bilingual_packet_predictions(packet_path: Path, backend, model_id: str,
                                        output: Path) -> dict:
    """Score only packet cards and write an exclusive, packet-hash-bound stream."""
    packet = packet_path.expanduser().resolve()
    rows, language, category = _packet_rows(packet)
    if not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
        raise ValueError("Prediction freeze needs a short model ID")
    destination = output.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Frozen prediction output already exists")
    if destination.resolve() == packet:
        raise ValueError("Packet and prediction destination must differ")
    cutoff = backend.thresholds.get(f"{language}:{category}")
    if type(cutoff) not in (int, float) or not math.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Research adapter has no finite frozen cutoff for this cell")
    packet_hash = sha256(packet)
    cards = [{"language": language, "source_kind": row["source_kind"],
              "text": row["abstract_card"]} for row in rows]
    scores = backend.score_cards(cards)
    if (not isinstance(scores, list) or len(scores) != len(rows) or
            any(type(score) not in (int, float) or not math.isfinite(score) or
                not 0 <= score <= 1 for score in scores)):
        raise ValueError("Research adapter returned invalid scores")
    if sha256(packet) != packet_hash:
        raise ValueError("Blinded packet changed during inference")
    predictions = [{"item_id": row["item_id"],
                    "predicted": [category] if score >= cutoff else [],
                    "strict_parse": True}
                   for row, score in zip(rows, scores)]
    document = {"schema_version": 2, "packet_sha256": packet_hash,
                "category": category, "language": language,
                "model_id": model_id, "predictions": predictions}
    destination.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    descriptor = os.open(destination, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(document, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    return {"output": str(destination), "category": category, "language": language,
            "cards": len(rows), "model_id": model_id, "packet_sha256": packet_hash,
            "predictions_sha256": sha256(destination),
            "source_batch_or_author_labels_read": False,
            "packet_origin_verified_by_software": False,
            "independent_human_review_verified_by_software": False,
            "model_frozen_before_batch_verified_by_software": False}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze RO/UK V25 decisions from a shuffled abstract packet")
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    try:
        _, language, category = _packet_rows(args.packet)
        if args.output.expanduser().exists() or args.output.expanduser().is_symlink():
            raise FileExistsError("Frozen prediction output already exists")
        source, cutoff, model_id = check_cell_wrapper(args.root, args.wrapper,
                                                      language, category)
        from .v25_abstract import V25SyntheticResearchClassifier
        backend = V25SyntheticResearchClassifier(args.base_model_path, source)
        if backend.thresholds.get(f"{language}:{category}") != cutoff:
            raise ValueError("Model cutoff differs from the cell wrapper")
        result = freeze_bilingual_packet_predictions(args.packet, backend, model_id,
                                                     args.output)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
