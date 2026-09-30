"""Optional-dependency inference utilities for the frozen v9 symbolic study."""

from __future__ import annotations

import math
import json
import struct
from pathlib import Path

from .prompt_v4 import applicable_codes, make_prompt_v4
from .review import sha256

MODEL_ID = "jhu-clsp/mmBERT-base"
REVISION = "c5955035435e2bf121cde7f3c8863ef52ff35d82"
BASE_HASHES = {
    "config.json": "47b40fd2e1df8299426dd5f4bb18c28f028cfafcb51b73645f83e596d187eb37",
    "pytorch_model.bin": "8ea64ec1ea4eb8fca0fc14b69a2ae571de6bfbc25fd214bb932dd4aba6a3a04e",
    "tokenizer.json": "197d4cc5406ee12cc50c8b5511f2393cc32d9db321545979ce041c1199178356",
    "tokenizer_config.json": "1d2f82c1341a79748e00efe82e67690f99d00b3c2a894f2b23128fd9d3519da3",
}
MAX_LENGTH = 256
THRESHOLD = 0.5


def verify_base(path: Path) -> None:
    for name, expected in BASE_HASHES.items():
        if sha256(path / name) != expected:
            raise ValueError(f"mmBERT base hash differs: {name}")


def verify_adapter_head(path: Path, num_labels: int = 2,
                        require_pooler: bool = False) -> None:
    """Reject an adapter lacking the fitted sequence-classification head."""
    if num_labels not in (2, 3, 4):
        raise ValueError("Unsupported fitted classifier shape")
    weight_path = path / "adapter_model.safetensors"
    size = weight_path.stat().st_size
    with weight_path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Adapter safetensors header is missing")
        length = struct.unpack("<Q", prefix)[0]
        if not 0 < length <= min(16_777_216, size - 8):
            raise ValueError("Adapter safetensors header length is invalid")
        try:
            header = json.loads(stream.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Adapter safetensors header is invalid") from exc
    if not isinstance(header, dict):
        raise ValueError("Adapter safetensors header must be a mapping")
    data_size = size - 8 - length
    required = [
        ("base_model.model.classifier.weight", [num_labels, 768], num_labels * 768 * 4),
        ("base_model.model.classifier.bias", [num_labels], num_labels * 4),
    ]
    if require_pooler:
        required.extend((
            ("base_model.model.pooler.dense.weight", [768, 768], 768 * 768 * 4),
            ("base_model.model.pooler.dense.bias", [768], 768 * 4),
        ))
    for name, shape, byte_count in required:
        entry = header.get(name)
        if not isinstance(entry, dict) or entry.get("dtype") != "F32" or entry.get("shape") != shape:
            raise ValueError(f"Adapter is missing its fitted classifier tensor: {name}")
        offsets = entry.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(value) is not int for value in offsets) or
                not 0 <= offsets[0] <= offsets[1] <= data_size or
                offsets[1] - offsets[0] != byte_count):
            raise ValueError(f"Adapter classifier tensor has invalid offsets: {name}")


def predict_rows(model, tokenizer, rows: list[dict], device: str, batch_size: int = 8,
                 threshold: float = THRESHOLD) -> list[dict]:
    """Return ordered decisions with one finite positive-class score per applicable rule."""
    import torch

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be within [0, 1]")
    tasks = [(index, code, make_prompt_v4(row, code))
             for index, row in enumerate(rows)
             for code in applicable_codes(row["source_kind"])]
    scores: dict[tuple[int, str], float] = {}
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(tasks), batch_size):
            part = tasks[start:start + batch_size]
            encoded = tokenizer([task[2] for task in part], padding=True, truncation=True,
                                max_length=MAX_LENGTH, return_tensors="pt").to(device)
            probabilities = torch.softmax(model(**encoded).logits.float(), dim=-1)[:, 1].cpu().tolist()
            for (index, code, _), score in zip(part, probabilities):
                if not math.isfinite(score):
                    raise ValueError("Nonfinite encoder score")
                scores[index, code] = score
    return [{"id": row["id"], "expected": row["labels"],
             "predicted": [code for code in applicable_codes(row["source_kind"])
                           if scores[index, code] >= threshold],
             "strict_parse": True,
             "scores": {code: scores[index, code] for code in applicable_codes(row["source_kind"])}}
            for index, row in enumerate(rows)]
