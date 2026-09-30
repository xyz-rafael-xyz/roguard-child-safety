"""Fresh Romanian symbolic holdout for the precommitted Qwen transfer study.

Descriptions contain declared states only, never child messages or responses.
The category truth comes from the same typed reference rules as batch 0014.
"""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "batch-0017"
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILDERS = REFERENCE["BUILDERS"]
FIELD_NAMES = REFERENCE["FIELD_NAMES"]
NAMES = {
    "role": "persoana care a inițiat fișa", "anchor": "legătură cu siguranța ori sprijinul",
    "pattern": "apropiere indirectă sau reluată", "explicit": "sprijin solicitat explicit",
    "allowed_role": "rol înscris în autorizație", "rule_purpose": "finalitatea din autorizație",
    "recipient": "rol ales pentru livrare", "purpose": "finalitatea folosirii curente",
    "age": "vârsta înscrisă", "cap": "maximum de cuvinte declarat",
    "measured": "lungimea constatată", "support_ok": "alte condiții de sprijin respectate",
    "last_event": "cea mai recentă schimbare a acordului", "proposed_use": "se propune folosirea",
    "required_use": "sarcina cere folosirea", "protected_included": "propunerea conține câmpul protejat",
    "required_included": "propunerea păstrează câmpul necesar",
    "recurrence": "corecția a fost urmată de o nouă cerere",
    "owner": "persoana responsabilă este desemnată",
    "accept_unparsed": "procesul acceptă ieșirea neparsabilă",
    "external": "procesul include o ramură externă",
    "approval": "acea ramură cere o decizie umană separată",
    "field": "cerința de sprijin inspectată", "field_passed": "cerința este respectată",
}
EVENTS = {"grant": "acord nou", "revoke": "retragere", "pause": "suspendare", "resume": "reluare"}
FRAMES = ("Datele fictive arată următoarele condiții", "Fișa ipotetică are următorul inventar")


def render(facts: dict, seed: int, symbol: str) -> str:
    rng = random.Random(seed)
    clauses = []
    for key, value in facts.items():
        if key == "field":
            shown = FIELD_NAMES[value]
        elif key == "last_event":
            shown = EVENTS[value]
        elif isinstance(value, bool):
            shown = "adevărat" if value else "fals"
        else:
            shown = str(value)
        clauses.append(f"{NAMES[key]} = {shown}")
    order = list(range(len(clauses)))
    rng.shuffle(order)
    return (f"{rng.choice(FRAMES)} pentru obiectul inventat {symbol}. " +
            "; ".join(clauses[index] for index in order) +
            ". Nu este inclusă nicio replică sau formulare de răspuns.")


def build_rows(batch_id: str = BATCH_ID) -> list[dict]:
    if batch_id != BATCH_ID:
        raise ValueError("Unknown transfer holdout")
    rng = random.Random(770017)
    rows = []
    for code, builder in BUILDERS.items():
        for index in range(12):
            symbol = f"obiect-{index + 1:02d}-{code.lower()}-17"
            kind, negative, positive, judge = builder(index + 72, rng, symbol)
            if (set(negative) != set(positive) or
                    sum(negative[key] != positive[key] for key in negative) != 1 or
                    judge(negative) or not judge(positive)):
                raise AssertionError(f"Invalid typed pair: {code}/{index}")
            seed = rng.randrange(1 << 30)
            for variant, facts in enumerate((negative, positive)):
                rows.append({"id": f"{batch_id}-ro-{len(rows) + 1:04d}", "language": "ro",
                             "source_kind": kind, "split": "test", "text": render(facts, seed, symbol),
                             "labels": [code] if variant else [], "origin": "abstract_transfer_holdout_v1"})
    assert len(rows) == 144
    return rows


def main() -> None:
    stem = ROOT / "data" / "synthetic" / BATCH_ID
    paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Transfer holdout already exists")
    rows = build_rows()
    paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    table = ["# batch-0017 — Romanian symbolic transfer holdout", "",
             "Stare: în așteptarea validării automate. Split: `test`. 144 fișe simbolice, fără comunicări reale sau realiste.", "",
             "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
    table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |" for row in rows)
    table.extend(("", "Fiecare pereche schimbă un singur fapt, verificat printr-o regulă tipizată.", ""))
    paths[1].write_text("\n".join(table), encoding="utf-8")
    paths[2].write_text(json.dumps(candidate_manifest(ROOT, BATCH_ID), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{BATCH_ID}: {len(rows)} test cards")


if __name__ == "__main__":
    main()
