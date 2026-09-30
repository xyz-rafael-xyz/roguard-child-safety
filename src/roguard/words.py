"""Local, deterministic word counting for a declared response-length contract."""

from __future__ import annotations

import unicodedata

_JOINERS = frozenset({"'", "’", "-", "‑"})


def count_words(text: str) -> int:
    """Count Unicode letter/number runs; retain internal apostrophes and hyphens.

    This is a published measurement convention, not a reading-level estimate.
    Combining marks stay with the preceding run, including decomposed accents.
    """
    if not isinstance(text, str):
        raise ValueError("Response text must be a string")
    count = 0
    in_word = False
    for index, character in enumerate(text):
        category = unicodedata.category(character)
        if category[0] in {"L", "N"}:
            if not in_word:
                count += 1
            in_word = True
        elif category[0] == "M" and in_word:
            continue
        elif (character in _JOINERS and in_word and index + 1 < len(text) and
              unicodedata.category(text[index + 1])[0] in {"L", "N"}):
            continue
        else:
            in_word = False
    return count
