"""Separate Romanian abstract holdout renderer for the v11 surface study."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "batch-0023"
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILDERS = REFERENCE["BUILDERS"]
FIELDS = REFERENCE["FIELD_NAMES"]
LABELS = {
    "role": "rolul celui de la origine", "anchor": "referința la protecție sau susținere",
    "pattern": "abordarea repetată fără cerere directă", "explicit": "cererea fără echivoc pentru ajutor",
    "allowed_role": "rolul inclus în permisiunea declarată", "rule_purpose": "întrebuințarea permisă",
    "recipient": "rolul ales ca destinatar", "purpose": "întrebuințarea cerută acum",
    "age": "anii declarați pentru evaluare", "cap": "cuvintele admise cel mult",
    "measured": "cuvintele găsite în ieșire", "support_ok": "alte condiții de sprijin îndeplinite",
    "last_event": "ultima schimbare a acordului", "proposed_use": "folosirea inclusă în cerere",
    "required_use": "folosirea necesară potrivit fișei", "protected_included": "elementul confidențial inclus",
    "required_included": "elementul obligatoriu păstrat", "recurrence": "reapariția după remediere",
    "owner": "persoana însărcinată cu controlul", "accept_unparsed": "admiterea ieșirii fără formă validă",
    "external": "efectul produs în afara fluxului", "approval": "acordul separat al evaluatorului uman",
    "field": "partea de sprijin supusă verificării", "field_passed": "partea verificată respectată",
}
EVENTS = {"grant": "drept conferit", "revoke": "drept retras",
          "pause": "drept pus în pauză", "resume": "drept reactivat"}
FRAMES = ("Proces-verbal fictiv al condițiilor", "Listă de verificare imaginară")


def render(facts: dict, seed: int, symbol: str) -> str:
    rng = random.Random(seed)
    pieces = []
    for key in sorted(facts):
        value = facts[key]
        shown = (FIELDS[value] if key == "field" else EVENTS[value] if key == "last_event"
                 else "se confirmă" if value is True else "nu se confirmă" if value is False else str(value))
        pieces.append(f"{LABELS[key]} — {shown}")
    return (f"{rng.choice(FRAMES)} {symbol}. " + ", ".join(pieces) +
            ". Sunt numai însușiri inventate, fără o relatare personală.")


def build_rows(batch_id: str = BATCH_ID) -> list[dict]:
    if batch_id != BATCH_ID:
        raise ValueError("Unknown v11 holdout")
    rng = random.Random(623023)
    rows = []
    for code, builder in BUILDERS.items():
        for index in range(12):
            symbol = f"nota-simbolică-{len(rows) // 2 + 1:03d}"
            kind, negative, positive, judge = builder(index + 384, rng, symbol)
            if (set(negative) != set(positive) or
                    sum(negative[key] != positive[key] for key in negative) != 1 or
                    judge(negative) or not judge(positive)):
                raise AssertionError(f"Invalid independent pair: {symbol}")
            seed = rng.randrange(1 << 30)
            for variant, facts in enumerate((negative, positive)):
                rows.append({"id": f"{batch_id}-ro-{len(rows) + 1:04d}", "language": "ro",
                             "source_kind": kind, "split": "test", "text": render(facts, seed, symbol),
                             "labels": [code] if variant else [], "origin": "abstract_surface_independent_v1"})
    assert len(rows) == 144
    return rows


def main() -> None:
    stem = ROOT / "data/synthetic" / BATCH_ID
    paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("V11 holdout already exists")
    rows = build_rows()
    paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                for row in rows), encoding="utf-8")
    table = ["# batch-0023 — Romanian independent surface holdout", "",
             "Stare: în așteptarea validării automate. Split: `test`. 144 de fișe simbolice fără comunicări reale ori realiste.", "",
             "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
    table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                 for row in rows)
    table.extend(("", "Perechile adiacente au o singură diferență tipizată, validată de regulile de referință.", ""))
    paths[1].write_text("\n".join(table), encoding="utf-8")
    paths[2].write_text(json.dumps(candidate_manifest(ROOT, BATCH_ID), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"{BATCH_ID}: {len(rows)} test cards")


if __name__ == "__main__":
    main()
