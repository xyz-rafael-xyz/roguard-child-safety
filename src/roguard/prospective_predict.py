"""Freeze Romanian abstract-card decisions from an opaque, label-free packet."""

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


def freeze_packet_predictions(packet_path: Path, backend, model_id: str,
                              output: Path) -> dict:
    """Score only shuffled packet cards; source IDs and intended labels are unavailable."""
    packet = packet_path.expanduser().resolve()
    rows, language, category = _packet_rows(packet)
    if language != "ro" or not isinstance(model_id, str) or not MODEL_ID.fullmatch(model_id):
        raise ValueError("Prediction freeze requires a Romanian packet and a short model ID")
    destination = output.expanduser().absolute()
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Frozen prediction output already exists")
    if destination == packet:
        raise ValueError("Packet and prediction destination must differ")
    threshold = backend.thresholds.get(category)
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Research adapter has no finite frozen cutoff for this category")
    packet_hash = sha256(packet)
    predictions = []
    for row in rows:
        scores = backend.score(row["abstract_card"], language, row["source_kind"])
        score = scores.get(category) if isinstance(scores, dict) else None
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Research adapter returned an invalid category score")
        predictions.append({"item_id": row["item_id"],
                            "predicted": [category] if score >= threshold else [],
                            "strict_parse": True})
    if sha256(packet) != packet_hash:
        raise ValueError("Blinded packet changed during inference")
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
        description="Freeze Romanian mmBERT decisions from a shuffled, label-free abstract packet")
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, default=ROOT / "models/ro-mmbert-v16-abstract")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        # Reject malformed or unsupported packets before loading the large base.
        _, language, _ = _packet_rows(args.packet)
        if language != "ro":
            raise ValueError("Research adapter accepts Romanian abstract packets only")
        if args.output.expanduser().exists() or args.output.expanduser().is_symlink():
            raise FileExistsError("Frozen prediction output already exists")
        from .mmbert_adapter import MMBertResearchClassifier
        backend = MMBertResearchClassifier(args.base_model_path, args.adapter)
        model_id = "mmbert_" + sha256(args.adapter / "adapter_model.safetensors")
        result = freeze_packet_predictions(args.packet, backend, model_id, args.output)
    except (OSError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
