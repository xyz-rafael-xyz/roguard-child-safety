"""Pure-Python inference for an optional fictional-message research model."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from importlib import resources
from pathlib import Path

from .cli import _strict_json_loads


class SyntheticMessageModel:
    """Local advisory score; never an authorization or real-world probability."""

    def __init__(self, data: bytes):
        self.sha256 = hashlib.sha256(data).hexdigest()
        payload = json.loads(gzip.decompress(data).decode("utf-8"))
        if (payload.get("format") != "roguard-character-tfidf-v1" or
                payload.get("language") not in ("ro", "uk") or
                not isinstance(payload.get("features"), list) or
                not isinstance(payload.get("ngram_range"), list) or
                len(payload["ngram_range"]) != 2):
            raise ValueError("Unsupported message model")
        self.low, self.high = payload["ngram_range"]
        self.language = payload["language"]
        self.cutoff = float(payload["cutoff"])
        self.intercept = float(payload["intercept"])
        if not (2 <= self.low <= self.high <= 6 and 0 <= self.cutoff <= 1):
            raise ValueError("Invalid model settings")
        self.features = {name: (float(idf), float(coefficient))
                         for name, idf, coefficient in payload["features"]}
        if len(self.features) != len(payload["features"]):
            raise ValueError("Duplicate model feature")

    @classmethod
    @lru_cache(maxsize=2)
    def packaged(cls, language: str) -> "SyntheticMessageModel":
        if language not in ("ro", "uk"):
            raise ValueError("Language must be ro or uk")
        filename = f"message-model-{language}.json.gz"
        try:
            data = resources.read_binary("roguard", filename)
        except FileNotFoundError as exc:
            raise ValueError("Optional fictional-message model is not packaged") from exc
        return cls(data)

    def score(self, text: str) -> float:
        if not isinstance(text, str) or not text.strip() or len(text) > 2000:
            raise ValueError("Provide nonempty text under 2001 characters")
        normalized = unicodedata.normalize("NFC", text).lower()
        counts = Counter(normalized[index:index + n]
                         for n in range(self.low, self.high + 1)
                         for index in range(len(normalized) - n + 1))
        weighted = []
        for name, count in counts.items():
            entry = self.features.get(name)
            if entry is not None:
                idf, coefficient = entry
                weighted.append(((1 + math.log(count)) * idf, coefficient))
        norm = math.sqrt(sum(value * value for value, _ in weighted))
        logit = self.intercept
        if norm:
            logit += sum(value * coefficient for value, coefficient in weighted) / norm
        return 1 / (1 + math.exp(-logit))


def assess_message(payload: object, model: SyntheticMessageModel | None = None) -> dict:
    if (not isinstance(payload, dict) or set(payload) !=
            {"language", "declared_source_role", "text"}):
        raise ValueError("Expected language, declared_source_role, and text")
    language = payload["language"]
    if language not in ("ro", "uk") or payload["declared_source_role"] != "minor":
        raise ValueError("Declare ro or uk and source_role minor")
    selected = model or SyntheticMessageModel.packaged(language)
    if selected.language != language:
        raise ValueError("Message model language mismatch")
    score = selected.score(payload["text"])
    return {"schema_version": "3.0", "language": language,
            "declared_source_role": "minor", "advisory_code": (
                "REVIEW_SUPPORT_SIGNAL" if score >= selected.cutoff else "NO_MODEL_SIGNAL"),
            "synthetic_score": round(score, 6), "synthetic_cutoff": round(selected.cutoff, 6),
            "model_sha256": selected.sha256, "real_world_accuracy_established": False,
            "caller_facts_unverified": True, "message_echoed": False,
            "external_action_taken": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Experimental fictional-message D1 advisory")
    parser.add_argument("input", nargs="?", type=Path,
                        help="JSON card path; omit to read standard input")
    args = parser.parse_args()
    try:
        raw = args.input.read_text(encoding="utf-8") if args.input else sys.stdin.read()
        report = assess_message(_strict_json_loads(raw))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
