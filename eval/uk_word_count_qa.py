"""Narrow numeral-agreement check for Ukrainian abstract A1 word-count cards.

The two recognized frames use nominative or inanimate accusative quantity phrases.
Other Ukrainian grammatical cases require a fluent reviewer and are not checked here.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

COUNT_FRAME = re.compile(r"(?:межа\s*[—-]\s*|виміряно\s+)(\d+)\s+(слово|слова|слів)\b")


def word_form(number: int) -> str:
    """Return the noun form after a cardinal count in the supported frames."""
    if number < 0:
        raise ValueError("Word count must be nonnegative")
    if number % 100 in (11, 12, 13, 14):
        return "слів"
    ending = number % 10
    return "слово" if ending == 1 else "слова" if ending in (2, 3, 4) else "слів"


def audit_card(text: str) -> tuple[dict, ...]:
    """Report mismatches without retaining or returning the card's text."""
    findings = []
    for match in COUNT_FRAME.finditer(text):
        number, actual = int(match.group(1)), match.group(2)
        expected = word_form(number)
        if actual != expected:
            findings.append({"number": number, "actual": actual, "expected": expected})
    return tuple(findings)


def audit_file(path: Path) -> dict:
    """Audit only Ukrainian abstract A1 response rows in a JSONL file."""
    cards = 0
    frames = 0
    problems = []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        if row.get("language") != "uk" or row.get("source_kind") != "response":
            continue
        text = row.get("text")
        if not isinstance(text, str) or not isinstance(row.get("id"), str):
            raise ValueError("Malformed Ukrainian A1 abstract card")
        if "A1" not in row.get("labels", ()) and not COUNT_FRAME.search(text):
            continue
        cards += 1
        frames += len(COUNT_FRAME.findall(text))
        for finding in audit_card(text):
            problems.append({"id": row["id"], **finding})
    if cards == 0 or frames != cards * 2:
        raise ValueError("Expected two supported count frames per Ukrainian A1 card")
    return {"cards": cards, "frames": frames, "errors": len(problems), "problems": problems,
            "scope": "abstract A1 cards; two fixed quantity frames only"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Ukrainian abstract A1 count agreement")
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    try:
        report = audit_file(args.path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
