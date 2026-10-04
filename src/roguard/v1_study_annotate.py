"""Collect private, content-free v1 D1/S1 judgments from blind packets."""

from __future__ import annotations

import argparse
import json
import os
import stat
import sys
from pathlib import Path

from .v1_study_eval import D1_FIELDS, S1_ALLOWED, _decision, _packet, _read_jsonl

ROOT = Path(__file__).resolve().parents[2]

D1_LABELS = {
    "ro": {
        "source_role": "Sursa declarată: 1 minor / 2 adult / 3 altă sursă / 0 necunoscut",
        "safety_or_support_anchor": "Reper contextual despre siguranță ori sprijin: 1 da / 2 nu / 0 neclar",
        "indirect_or_repeated_support_pattern": "Tipar indirect sau reluat: 1 da / 2 nu / 0 neclar",
        "explicit_support_request": "Cerere explicită de sprijin: 1 da / 2 nu / 0 neclar",
        "direct_safety_statement": "Afirmație directă despre siguranță: 1 da / 2 nu / 0 neclar",
        "retracted_or_denied": "Retragere ori negare ulterioară: 1 da / 2 nu / 0 neclar",
    },
    "uk": {
        "source_role": "Зазначене джерело: 1 дитина / 2 дорослий / 3 інше / 0 невідомо",
        "safety_or_support_anchor": "Контекстна підстава щодо безпеки чи підтримки: 1 так / 2 ні / 0 невідомо",
        "indirect_or_repeated_support_pattern": "Непряма або повторена ознака: 1 так / 2 ні / 0 невідомо",
        "explicit_support_request": "Пряме прохання про підтримку: 1 так / 2 ні / 0 невідомо",
        "direct_safety_statement": "Пряме твердження про безпеку: 1 так / 2 ні / 0 невідомо",
        "retracted_or_denied": "Подальше відкликання чи заперечення: 1 так / 2 ні / 0 невідомо",
    },
}


def _choice(prompt: str, choices: dict[str, object]) -> object:
    while True:
        print(prompt, file=sys.stderr, flush=True)
        answer = sys.stdin.readline()
        if not answer:
            raise EOFError("Annotation stopped before the next answer")
        value = answer.strip()
        if value in choices:
            return choices[value]
        print("Use one of: " + ", ".join(choices), file=sys.stderr, flush=True)


def _collect(language: str, category: str) -> dict:
    if category == "D1":
        facts = {}
        for name in (
                "source_role", "safety_or_support_anchor",
                "indirect_or_repeated_support_pattern", "explicit_support_request",
                "direct_safety_statement", "retracted_or_denied"):
            if name == "source_role":
                facts[name] = _choice(D1_LABELS[language][name],
                                      {"1": "minor", "2": "adult", "3": "other", "0": None})
            else:
                facts[name] = _choice(D1_LABELS[language][name],
                                      {"1": True, "2": False, "0": None})
        assert set(facts) == D1_FIELDS
        return {"language": language, "disclosure": facts}
    applicable, passed = [], []
    unknown = False
    for field in sorted(S1_ALLOWED):
        caption = ("Câmp" if language == "ro" else "Поле") + f" {field}: "
        applies_prompt = ("1 se aplică / 2 nu se aplică / 0 neclar" if language == "ro"
                          else "1 застосовне / 2 незастосовне / 0 невідомо")
        passes_prompt = ("1 îndeplinit / 2 neîndeplinit / 0 neclar" if language == "ro"
                         else "1 виконано / 2 не виконано / 0 невідомо")
        applies = _choice(caption + applies_prompt,
                          {"1": True, "2": False, "0": None})
        if applies is None:
            unknown = True
            continue
        if not applies:
            continue
        applicable.append(field)
        fulfills = _choice(caption + passes_prompt,
                           {"1": True, "2": False, "0": None})
        if fulfills is None:
            unknown = True
        elif fulfills:
            passed.append(field)
    return {"language": language,
            "support": {"applicable_fields": None if unknown else applicable,
                        "passed_fields": None if unknown else passed}}


def annotate_v1_packet(packet_path: Path, answers_path: Path) -> dict:
    """Resume in packet order; save only opaque IDs and structured decisions."""
    packet_path = packet_path.expanduser().resolve()
    answers_path = answers_path.expanduser().absolute()
    resolved_answers = answers_path.resolve()
    if resolved_answers.is_relative_to(ROOT) and not resolved_answers.is_relative_to(ROOT / "review_runs"):
        raise ValueError("Private answers must stay outside Git or under review_runs/")
    raw = _read_jsonl(packet_path)
    if not raw:
        raise ValueError("Blind packet is empty")
    language, category = raw[0].get("language"), raw[0].get("category")
    if language not in ("ro", "uk") or category not in ("D1", "S1"):
        raise ValueError("Unsupported v1 packet language or category")
    rows = _packet(packet_path, language, category)
    if (answers_path == packet_path or
            (answers_path.exists() and os.path.samefile(packet_path, answers_path))):
        raise ValueError("Answer file must differ from the packet")
    if answers_path.exists():
        if answers_path.is_symlink() or not answers_path.is_file():
            raise ValueError("Answer file must be regular")
        if stat.S_IMODE(answers_path.stat().st_mode) & 0o077:
            raise ValueError("Existing answers must be private (mode 0600)")
        existing = _read_jsonl(answers_path)
        if len(existing) > len(rows):
            raise ValueError("Answer file exceeds packet length")
        for index, item in enumerate(existing):
            if (set(item) != {"item_id", "card"} or
                    item["item_id"] != rows[index]["item_id"]):
                raise ValueError("Answer prefix differs from packet order")
            _decision(language, category, item["card"])
    else:
        existing = []
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_APPEND
    flags |= os.O_NOFOLLOW if hasattr(os, "O_NOFOLLOW") else 0
    if not answers_path.exists():
        flags |= os.O_CREAT | os.O_EXCL
    descriptor = os.open(answers_path, flags, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as output:
        for index in range(len(existing), len(rows)):
            row = rows[index]
            print(f"[{index + 1}/96] {row['item_id']}\n{row['abstract_card']}",
                  file=sys.stderr, flush=True)
            card = _collect(language, category)
            _decision(language, category, card)
            output.write(json.dumps({"item_id": row["item_id"], "card": card},
                                    ensure_ascii=False, sort_keys=True) + "\n")
            output.flush()
            os.fsync(output.fileno())
    return {"language": language, "category": category,
            "rows_total": len(rows), "rows_completed": len(rows),
            "resumed_from": len(existing), "card_text_stored_in_answers": False}


def main() -> None:
    parser = argparse.ArgumentParser(description="Privately annotate a v1 blind packet")
    parser.add_argument("packet", type=Path)
    parser.add_argument("answers", type=Path)
    args = parser.parse_args()
    try:
        report = annotate_v1_packet(args.packet, args.answers)
    except (OSError, TypeError, ValueError, json.JSONDecodeError, EOFError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
