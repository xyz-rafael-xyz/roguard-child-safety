"""Sealed Romanian symbolic-card transfer set for the encoder study.

No child utterances, response text, cases, or realistic paraphrases are made.
"""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "batch-0019"
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILDERS = REFERENCE["BUILDERS"]
FIELD_NAMES = REFERENCE["FIELD_NAMES"]

HEADINGS = {
    "role": "statutul sursei", "anchor": "indiciul pentru protecție sau susținere",
    "pattern": "evoluția indirectă ori repetată", "explicit": "solicitarea directă",
    "allowed_role": "statutul prevăzut în condiție", "rule_purpose": "finalitatea permisă",
    "recipient": "rolul către care se îndreaptă operația", "purpose": "finalitatea operației",
    "age": "numărul de ani declarat", "cap": "limita de termeni", "measured": "termeni rezultați la numărare",
    "support_ok": "restul verificărilor de susținere",
    "last_event": "evenimentul de acord cel mai recent", "proposed_use": "folosirea planificată",
    "required_use": "folosirea cerută de condiție", "protected_included": "prezența datei protejate",
    "required_included": "păstrarea datei obligatorii", "recurrence": "repetarea după intervenția corectivă",
    "owner": "persoana desemnată să revadă", "accept_unparsed": "acceptarea rezultatului fără structură validă",
    "external": "activarea operației din afara sistemului", "approval": "avizul uman separat pentru acea operație",
    "field": "criteriul de susținere urmărit", "field_passed": "îndeplinirea criteriului urmărit",
}
EVENTS = {"grant": "activare", "revoke": "retragere", "pause": "suspendare", "resume": "reactivare"}
FRAMES = ("Tabel de condiții inventate", "Înregistrare fictivă pentru verificarea regulii")


def render(facts: dict, seed: int, symbol: str) -> str:
    rng = random.Random(seed)
    entries = []
    for key in sorted(facts, reverse=True):
        value = facts[key]
        shown = (FIELD_NAMES[value] if key == "field" else EVENTS[value] if key == "last_event"
                 else "adevărat" if value is True else "fals" if value is False else str(value))
        entries.append(f"{HEADINGS[key]} = {shown}")
    return (f"{rng.choice(FRAMES)} {symbol}. " + "; ".join(entries) +
            ". Date simbolice, fără comunicarea vreunei persoane.")


def build_rows(batch_id: str = BATCH_ID) -> list[dict]:
    if batch_id != BATCH_ID:
        raise ValueError("Unknown encoder holdout")
    rng = random.Random(190919)
    rows = []
    for builder in BUILDERS.values():
        for index in range(12):
            symbol = f"nr-{len(rows) // 2 + 1:03d}"
            kind, negative, positive, judge = builder(index + 144, rng, symbol)
            if (set(negative) != set(positive) or
                    sum(negative[key] != positive[key] for key in negative) != 1 or
                    judge(negative) or not judge(positive)):
                raise AssertionError(f"Invalid one-fact reference pair: {symbol}")
            seed = rng.randrange(1 << 30)
            for variant, facts in enumerate((negative, positive)):
                code = next(code for code, candidate in BUILDERS.items() if candidate is builder)
                rows.append({"id": f"{batch_id}-ro-{len(rows) + 1:04d}", "language": "ro",
                             "source_kind": kind, "split": "test", "text": render(facts, seed, symbol),
                             "labels": [code] if variant else [], "origin": "abstract_encoder_holdout_v1"})
    assert len(rows) == 144
    return rows


def main() -> None:
    stem = ROOT / "data" / "synthetic" / BATCH_ID
    paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Encoder holdout already exists")
    rows = build_rows()
    paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                for row in rows), encoding="utf-8")
    table = ["# batch-0019 — Romanian encoder holdout", "",
             "Stare: în așteptarea validării automate. Split: `test`. 144 de fișe simbolice, fără comunicări reale ori realiste.", "",
             "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
    table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                 for row in rows)
    table.extend(("", "Fiecare pereche schimbă exact un fapt verificat de regula tipizată.", ""))
    paths[1].write_text("\n".join(table), encoding="utf-8")
    paths[2].write_text(json.dumps(candidate_manifest(ROOT, BATCH_ID), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"{BATCH_ID}: {len(rows)} test cards")


if __name__ == "__main__":
    main()
