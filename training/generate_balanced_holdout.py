"""Fresh abstract Romanian holdout for equal-category task sampling."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "batch-0018"
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILDERS = REFERENCE["BUILDERS"]
FIELD_NAMES = REFERENCE["FIELD_NAMES"]
NAMES = {
    "role": "rol declarat la origine", "anchor": "există un reper de siguranță sau sprijin",
    "pattern": "există un parcurs indirect ori repetat", "explicit": "cererea de sprijin este directă",
    "allowed_role": "rol autorizat de regulă", "rule_purpose": "scop înscris în regulă",
    "recipient": "destinatarul încercat", "purpose": "scopul încercării",
    "age": "vârsta din scenariu", "cap": "pragul numeric de cuvinte",
    "measured": "cuvinte numărate", "support_ok": "celelalte cerințe de sprijin sunt îndeplinite",
    "last_event": "ultimul act din istoricul acordului", "proposed_use": "elementul este folosit",
    "required_use": "folosirea este cerută", "protected_included": "câmpul protejat apare",
    "required_included": "câmpul necesar rămâne",
    "recurrence": "există corecție urmată de reluarea solicitării",
    "owner": "există un responsabil de revizuire",
    "accept_unparsed": "un rezultat neparsabil poate fi acceptat",
    "external": "fluxul are ieșire externă",
    "approval": "ieșirea externă are nevoie de aprobare umană distinctă",
    "field": "element de sprijin verificat", "field_passed": "elementul trece verificarea",
}
EVENTS = {"grant": "acord nou", "revoke": "acord retras",
          "pause": "acord suspendat", "resume": "acord reluat"}
FRAMES = ("Exercițiu abstract de reguli", "Inventar pentru simularea deciziei")


def render(facts: dict, seed: int, symbol: str) -> str:
    rng = random.Random(seed)
    parts = []
    for key, value in facts.items():
        shown = (FIELD_NAMES[value] if key == "field" else EVENTS[value] if key == "last_event"
                 else "prezent" if value is True else "absent" if value is False else str(value))
        parts.append(f"{NAMES[key]} — {shown}")
    rng.shuffle(parts)
    return (f"{rng.choice(FRAMES)} pentru {symbol}: " + " / ".join(parts) +
            ". Sunt descrise numai proprietăți inventate, fără mesaje ori răspunsuri.")


def build_rows(batch_id: str = BATCH_ID) -> list[dict]:
    if batch_id != BATCH_ID:
        raise ValueError("Unknown balanced holdout")
    rng = random.Random(880018)
    rows = []
    for code, builder in BUILDERS.items():
        for index in range(12):
            symbol = f"variabilă-{code.lower()}-{index + 1:02d}-18"
            kind, negative, positive, judge = builder(index + 96, rng, symbol)
            if (set(negative) != set(positive) or
                    sum(negative[key] != positive[key] for key in negative) != 1 or
                    judge(negative) or not judge(positive)):
                raise AssertionError(f"Invalid balanced-study pair: {code}/{index}")
            seed = rng.randrange(1 << 30)
            for variant, facts in enumerate((negative, positive)):
                rows.append({"id": f"{batch_id}-ro-{len(rows) + 1:04d}", "language": "ro",
                             "source_kind": kind, "split": "test",
                             "text": render(facts, seed, symbol),
                             "labels": [code] if variant else [], "origin": "abstract_balanced_holdout_v1"})
    assert len(rows) == 144
    return rows


def main() -> None:
    stem = ROOT / "data" / "synthetic" / BATCH_ID
    paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Balanced holdout already exists")
    rows = build_rows()
    paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                for row in rows), encoding="utf-8")
    table = ["# batch-0018 — Romanian balanced-sampling holdout", "",
             "Stare: în așteptarea validării automate. Split: `test`. 144 de fișe simbolice fără comunicări reale ori realiste.", "",
             "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
    table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                 for row in rows)
    table.extend(("", "Fiecare pereche schimbă un singur fapt verificat printr-o regulă tipizată.", ""))
    paths[1].write_text("\n".join(table), encoding="utf-8")
    paths[2].write_text(json.dumps(candidate_manifest(ROOT, BATCH_ID), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"{BATCH_ID}: {len(rows)} test cards")


if __name__ == "__main__":
    main()
