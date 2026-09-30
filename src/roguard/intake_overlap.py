"""Content-free lexical overlap audit for new abstract test cards."""

from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from .review import ReviewError

NEAR_OVERLAP_DICE = 0.8


def _grams(text: str) -> frozenset[str]:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return frozenset(normalized[index:index + 5]
                     for index in range(max(0, len(normalized) - 4)))


def _cross_pair_overlap(grams_by_card: list[frozenset[str]]) -> dict:
    """Flag similar cards from different adjacent pairs, excluding each pair's mate."""
    flagged = []
    maximum = 0.0
    comparisons = 0
    for index, grams in enumerate(grams_by_card):
        best = None
        best_score = 0.0
        for other_index, other in enumerate(grams_by_card):
            if index // 2 == other_index // 2:
                continue
            if index < other_index:
                comparisons += 1
            score = (2 * len(grams & other) / (len(grams) + len(other))
                     if grams or other else 0.0)
            if score > best_score:
                best, best_score = other_index, score
        maximum = max(maximum, best_score)
        if best is not None and best_score >= NEAR_OVERLAP_DICE:
            flagged.append({"card_number": index + 1,
                            "nearest_other_card_number": best + 1,
                            "similarity": round(best_score, 4)})
    return {"cross_pair_comparisons": comparisons,
            "near_overlap_cards": len(flagged),
            "maximum_similarity": round(maximum, 4) if comparisons else None,
            "flagged": flagged,
            "semantic_independence_verified": False}


def audit_prior_overlap(root: Path, rows: list[dict], batch: str) -> dict:
    """Flag prior and cross-pair reuse without returning text or claiming independence."""
    if not rows or len({(row.get("language"), row.get("source_kind")) for row in rows}) != 1:
        raise ValueError("Overlap audit needs one language and source kind")
    language, kind = rows[0]["language"], rows[0]["source_kind"]
    if any(not isinstance(row.get("text"), str) for row in rows):
        raise ValueError("Overlap audit needs abstract card text")
    candidate_grams = [_grams(row["text"]) for row in rows]
    cross_pair = _cross_pair_overlap(candidate_grams)
    prior = []
    index = defaultdict(list)
    prior_hashes = {}
    for path in sorted((root / "data/synthetic").glob("batch-*.jsonl")):
        if path.stem == batch:
            continue
        if path.is_symlink():
            raise ReviewError("Symlinked prior batch is not accepted")
        raw = path.read_bytes()
        prior_hashes[path.stem] = hashlib.sha256(raw).hexdigest()
        for line in raw.decode("utf-8").splitlines():
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ReviewError(f"Invalid prior batch while checking overlap: {path.name}") from exc
            if (not isinstance(item, dict) or item.get("language") != language or
                    item.get("source_kind") != kind or not isinstance(item.get("text"), str)):
                continue
            grams = _grams(item["text"])
            if not grams:
                continue
            position = len(prior)
            prior.append((path.stem, grams))
            for gram in grams:
                index[gram].append(position)
    flagged = []
    maximum = 0.0
    for card_number, grams in enumerate(candidate_grams, 1):
        overlaps = Counter(position for gram in grams for position in index.get(gram, ()))
        if not overlaps:
            continue
        best = max(overlaps, key=lambda position: (
            2 * overlaps[position] / (len(grams) + len(prior[position][1])),
            -position))
        score = 2 * overlaps[best] / (len(grams) + len(prior[best][1]))
        maximum = max(maximum, score)
        if score >= NEAR_OVERLAP_DICE:
            flagged.append({"card_number": card_number, "prior_batch": prior[best][0],
                            "similarity": round(score, 4)})
    return {
        "language": language, "source_kind": kind,
        "prior_cards_compared": len(prior), "candidate_cards": len(rows),
        "prior_batch_sha256": prior_hashes,
        "metric": "unique_normalized_character_5gram_dice",
        "near_overlap_threshold": NEAR_OVERLAP_DICE,
        "near_overlap_cards": len(flagged),
        "maximum_similarity": round(maximum, 4) if prior else None,
        "flagged": flagged,
        "within_candidate_cross_pair": cross_pair,
        "semantic_independence_verified": False,
    }
