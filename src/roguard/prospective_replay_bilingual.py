"""Reproduce one frozen RO/UK comparator prediction stream from its adapter."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .annotate_packet import _packet_rows
from .human_eval import _read_blind_predictions
from .prospective_predict_bilingual import (
    check_cell_wrapper, freeze_bilingual_packet_predictions)
from .review import sha256
from .study_freeze import verify_study_freeze


def replay_bilingual_predictions(root: Path, packet: Path, predictions: Path,
                                 wrapper: Path, freeze_path: Path, backend) -> dict:
    """Verify exact pre-author bytes, then independently rerun every blind decision."""
    root = root.resolve()
    if predictions.is_symlink() or not predictions.is_file():
        raise ValueError("Predictions must be a regular file")
    packet = packet.resolve()
    predictions = predictions.resolve()
    rows, language, category = _packet_rows(packet)
    source, cutoff, model_id = check_cell_wrapper(root, wrapper, language, category)
    relative_wrapper = wrapper.resolve().relative_to(root)
    freeze = verify_study_freeze(root, freeze_path, language, category)
    record_path = root / freeze["path"]
    record = json.loads(record_path.read_text(encoding="utf-8"))
    if (record["adapter_dir"] != relative_wrapper.as_posix() or
            record["cutoff"] != cutoff or
            freeze["prediction_model_id"] != model_id or
            backend.thresholds.get(f"{language}:{category}") != cutoff):
        raise ValueError("Replay model differs from the registered cell freeze")
    packet_hash = sha256(packet)
    identity = {row["item_id"]: row["item_id"] for row in rows}
    saved_model_id, _ = _read_blind_predictions(
        predictions, packet, packet_hash, language, category, identity)
    if saved_model_id != model_id:
        raise ValueError("Saved predictions name another model")
    saved_hash = sha256(predictions)
    with tempfile.TemporaryDirectory(prefix="roguard-replay-") as temporary:
        replay_file = Path(temporary) / "predictions.json"
        freeze_bilingual_packet_predictions(packet, backend, model_id, replay_file)
        saved = json.loads(predictions.read_text(encoding="utf-8"))
        reproduced = json.loads(replay_file.read_text(encoding="utf-8"))
    if sha256(packet) != packet_hash or sha256(predictions) != saved_hash:
        raise ValueError("Replay input changed during inference")
    if saved != reproduced:
        raise ValueError("Frozen predictions do not reproduce from the registered adapter")
    return {"schema_version": 1, "language": language, "category": category,
            "cards": len(rows), "packet_sha256": packet_hash,
            "predictions_sha256": saved_hash, "model_id": model_id,
            "model_freeze_sha256": freeze["sha256"],
            "all_decisions_reproduced_from_registered_adapter": True,
            "human_authorship_or_label_validity_verified": False,
            "real_child_language_accuracy_established": False,
            "source_adapter_dir": str(source.resolve().relative_to(root))}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a frozen RO/UK abstract comparator prediction packet")
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    args = parser.parse_args()
    try:
        rows, language, category = _packet_rows(args.packet)
        del rows
        source, cutoff, _ = check_cell_wrapper(args.root, args.wrapper,
                                               language, category)
        verify_study_freeze(args.root, args.freeze, language, category)
        from .v25_abstract import V25SyntheticResearchClassifier
        backend = V25SyntheticResearchClassifier(args.base_model_path, source)
        if backend.thresholds.get(f"{language}:{category}") != cutoff:
            raise ValueError("Model cutoff differs from the cell wrapper")
        result = replay_bilingual_predictions(
            args.root, args.packet, args.predictions, args.wrapper,
            args.freeze, backend)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
