"""Replay the packaged V24 adapter on its consumed development cards."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from roguard.mmbert_study import predict_rows
from roguard.review import load_approved, sha256

ROOT = Path(__file__).resolve().parents[1]
ADAPTER = ROOT / "models/ro-bert-v24-abstract"


def replay(base_model_path: Path, tolerance: float = 1e-5) -> dict:
    checked = runpy.run_path(str(ROOT / "eval/verify_committed_v24_selection.py"))["verify"](ROOT)
    base = base_model_path.resolve()
    manifest = json.loads((ADAPTER / "research.json").read_text(encoding="utf-8"))
    for name, digest in manifest["base_files_sha256"].items():
        if sha256(base / name) != digest:
            raise ValueError(f"Romanian BERT base hash differs: {name}")
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
    foundation = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=2, use_safetensors=True, local_files_only=True)
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    model = PeftModel.from_pretrained(foundation, ADAPTER, is_trainable=False).to(device).eval()
    epoch = checked["selected_epoch"]
    saved = json.loads((ROOT / f"eval/runs/ro-bert-v24-epoch-{epoch}-dev.json").read_text(
        encoding="utf-8"))["predictions"]
    maximum, count = 0.0, 0
    for batch in ("batch-0037", "batch-0038"):
        rows = [row for row in load_approved(ROOT, [batch]) if row["source_kind"] == "message"]
        current = predict_rows(model, tokenizer, rows, device)
        expected = saved[batch]
        if len(current) != 96 or len(current) != len(expected):
            raise ValueError("V24 development replay count differs")
        for row, item in zip(current, expected):
            if row["id"] != item["id"]:
                raise ValueError("V24 development replay row differs")
            delta = abs(row["scores"]["D1"] - item["scores"]["D1"])
            maximum = max(maximum, delta)
            count += 1
            if delta > tolerance or row["predicted"] != item["predicted"]:
                raise ValueError(f"V24 saved score or decision differs: {row['id']}")
    return {"cards_replayed": count, "max_absolute_score_delta": maximum,
            "selected_epoch": epoch,
            "development_gate_passed": checked["development_gate_passed"],
            "new_held_out_test_exists": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model-path", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(replay(args.base_model_path), indent=2))


if __name__ == "__main__":
    main()
