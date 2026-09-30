"""Collect content-free answers while showing one blinded abstract card at a time."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path

from .agreement_batch import ITEM_ID, _one_category, _read_jsonl
from .blind_packets import KINDS
from .cli import assess_json
from .review_audit import _read_packet
from .review_card import collect_disclosure_card, collect_support_card


def _packet_rows(path: Path) -> tuple[list[dict], str, str]:
    rows = _read_packet(path)
    if len(rows) < 2:
        raise ValueError("Blinded packet needs at least two abstract cards")
    first = rows[0]
    language, category = first["language"], first["category"]
    if language not in ("ro", "uk") or category not in KINDS:
        raise ValueError("Blinded packet has an unsupported language or category")
    ids = []
    for row in rows:
        item_id = row["item_id"]
        if (not isinstance(item_id, str) or not ITEM_ID.fullmatch(item_id) or
                row["language"] != language or row["category"] != category or
                row["source_kind"] != KINDS[category] or
                not isinstance(row["abstract_card"], str) or
                not row["abstract_card"].strip()):
            raise ValueError("Blinded packet has an invalid abstract-card row")
        ids.append(item_id)
    if len(set(ids)) != len(ids):
        raise ValueError("Blinded packet has duplicate opaque item IDs")
    return rows, language, category


def _completed_prefix(path: Path, rows: list[dict], language: str,
                      category: str) -> int:
    if not path.exists():
        return 0
    if path.is_symlink() or not path.is_file():
        raise ValueError("Answer destination must be an ordinary file")
    if stat.S_IMODE(path.stat().st_mode) & 0o077:
        raise ValueError("Existing answer file must be private (mode 0600)")
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if len(lines) > len(rows) or any(not line.endswith("\n") for line in lines):
        raise ValueError("Existing answers are incomplete or exceed the packet")
    existing = _read_jsonl(path) if lines else []
    for index, item in enumerate(existing):
        kind, card = _one_category(item)
        if (item["item_id"] != rows[index]["item_id"] or
                kind != ("disclosure" if category == "D1" else "support") or
                card.get("language") != language):
            raise ValueError("Existing answers differ from the packet prefix")
        assess_json(card)
    return len(existing)


def annotate_packet(
    packet_path: Path,
    answers_path: Path,
    read_line: Callable[[], str],
    show: Callable[[str], None],
) -> dict:
    """Resume an ordered packet locally; write one validated JSONL row per card."""
    packet, answers = packet_path.expanduser().resolve(), answers_path.expanduser()
    if packet == answers.resolve() or (answers.exists() and os.path.samefile(packet, answers)):
        raise ValueError("Packet and answer destination must differ")
    rows, language, category = _packet_rows(packet)
    completed = _completed_prefix(answers, rows, language, category)
    if completed == len(rows):
        return {"rows_total": len(rows), "rows_completed": completed,
                "resumed_from": completed, "category": category,
                "language": language, "answer_text_stored": False}
    answers.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND
    flags |= os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    flags |= 0 if answers.exists() else os.O_CREAT | os.O_EXCL
    descriptor = os.open(answers, flags, 0o600)
    collector = collect_disclosure_card if category == "D1" else collect_support_card
    with os.fdopen(descriptor, "a", encoding="utf-8") as output:
        for index in range(completed, len(rows)):
            item = rows[index]
            show(f"[{index + 1}/{len(rows)}] {item['item_id']}\n{item['abstract_card']}")
            card = collector(language, read_line, show)
            output.write(json.dumps({"item_id": item["item_id"], "card": card},
                                    ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    return {"rows_total": len(rows), "rows_completed": len(rows),
            "resumed_from": completed, "category": category,
            "language": language, "answer_text_stored": False}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Annotate a blinded abstract packet locally with numeric D1/S1 choices")
    parser.add_argument("packet", type=Path)
    parser.add_argument("answers", type=Path)
    args = parser.parse_args()
    try:
        result = annotate_packet(
            args.packet, args.answers, sys.stdin.readline,
            lambda message: print(message, file=sys.stderr, flush=True),
        )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
