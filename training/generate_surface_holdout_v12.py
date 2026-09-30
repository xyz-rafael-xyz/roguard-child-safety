"""Sealed Romanian abstract-card surface for the v12 threshold-transfer study."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "batch-0024"
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILDERS = REFERENCE["BUILDERS"]
FIELDS = REFERENCE["FIELD_NAMES"]
LABELS = {
    "role": "calitatea sursei", "anchor": "legătura cu protecția ori sprijinul",
    "pattern": "indiciul indirect sau repetat", "explicit": "ajutorul cerut direct",
    "allowed_role": "rolul admis în regulă", "rule_purpose": "scopul înscris ca admis",
    "recipient": "rolul care ar primi elementul", "purpose": "scopul folosirii propuse",
    "age": "vârsta consemnată", "cap": "limita de cuvinte fixată",
    "measured": "lungimea exprimată în cuvinte", "support_ok": "condițiile suplimentare îndeplinite",
    "last_event": "actul final privind acordul", "proposed_use": "folosirea avută în vedere",
    "required_use": "folosirea necesară pentru sarcină",
    "protected_included": "prezența câmpului rezervat",
    "required_included": "păstrarea câmpului obligatoriu", "recurrence": "repetarea după corectare",
    "owner": "responsabilul de verificare", "accept_unparsed": "admiterea unui rezultat invalid",
    "external": "activarea unei ieșiri externe", "approval": "aprobarea distinctă a unei persoane",
    "field": "condiția de sprijin examinată", "field_passed": "condiția examinată îndeplinită",
}
EVENTS = {"grant": "acord acordat", "revoke": "acord revocat",
          "pause": "acord suspendat", "resume": "acord reactivat"}
FRAMES = ("Inventar fictiv al regulii", "Fișă simbolică de comparație")


def render(facts: dict, seed: int, symbol: str) -> str:
    rng = random.Random(seed)
    pieces = []
    for key in sorted(facts, reverse=True):
        value = facts[key]
        shown = (FIELDS[value] if key == "field" else EVENTS[value] if key == "last_event"
                 else "se aplică" if value is True else "nu se aplică" if value is False else str(value))
        pieces.append(f"{LABELS[key]}: {shown}")
    return (f"{rng.choice(FRAMES)} {symbol} — " + "; ".join(pieces) +
            ". Numai proprietăți inventate, fără mesaj personal.")


def build_rows(batch_id: str = BATCH_ID) -> list[dict]:
    if batch_id != BATCH_ID:
        raise ValueError("Unknown v12 holdout")
    rng = random.Random(724024)
    rows = []
    for code, builder in BUILDERS.items():
        for index in range(12):
            symbol = f"probă-{batch_id[-4:]}-{len(rows) // 2 + 1:03d}"
            kind, negative, positive, judge = builder(index + 480, rng, symbol)
            if (set(negative) != set(positive) or
                    sum(negative[key] != positive[key] for key in negative) != 1 or
                    judge(negative) or not judge(positive)):
                raise AssertionError(f"Invalid independent pair: {symbol}")
            seed = rng.randrange(1 << 30)
            for variant, facts in enumerate((negative, positive)):
                rows.append({"id": f"{batch_id}-ro-{len(rows) + 1:04d}", "language": "ro",
                             "source_kind": kind, "split": "test", "text": render(facts, seed, symbol),
                             "labels": [code] if variant else [], "origin": "abstract_surface_independent_v2"})
    assert len(rows) == 144
    return rows


def main() -> None:
    stem = ROOT / "data/synthetic" / BATCH_ID
    paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("V12 holdout already exists")
    rows = build_rows()
    paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                for row in rows), encoding="utf-8")
    table = ["# batch-0024 — Romanian v12 independent surface holdout", "",
             "Stare: în așteptarea validării automate. Split: `test`. 144 de fișe simbolice fără comunicări reale ori realiste.", "",
             "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
    table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                 for row in rows)
    table.extend(("", "Fiecare pereche schimbă un singur fapt verificat de regula tipizată.", ""))
    paths[1].write_text("\n".join(table), encoding="utf-8")
    paths[2].write_text(json.dumps(candidate_manifest(ROOT, BATCH_ID), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"{BATCH_ID}: {len(rows)} test cards")


if __name__ == "__main__":
    main()
