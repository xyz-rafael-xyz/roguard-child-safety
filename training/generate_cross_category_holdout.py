"""Fresh abstract Romanian one-fact test for cross-category response learning."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "batch-0020"
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILDERS = REFERENCE["BUILDERS"]
FIELD_NAMES = REFERENCE["FIELD_NAMES"]

TERMS = {
    "role": "sursa consemnată", "anchor": "indicatorul legat de protecție ori ajutor",
    "pattern": "succesiunea de abordări indirecte", "explicit": "cererea formulată direct",
    "allowed_role": "rolul care apare în condiția de acces", "rule_purpose": "utilizarea admisă de condiție",
    "recipient": "rolul către care se propune transferul", "purpose": "utilizarea propusă acum",
    "age": "ani în fișă", "cap": "număr maxim de cuvinte", "measured": "număr efectiv de cuvinte",
    "support_ok": "criteriile de sprijin rămase",
    "last_event": "actul final din șirul permisiunilor", "proposed_use": "elementul utilizat în proiect",
    "required_use": "elementul cerut pentru rezultat", "protected_included": "intrarea protejată în rezultat",
    "required_included": "intrarea necesară în rezultat", "recurrence": "reluare după remediere",
    "owner": "revizorul desemnat", "accept_unparsed": "admiterea unui răspuns cu structură invalidă",
    "external": "ramura care produce un efect extern", "approval": "confirmarea umană distinctă",
    "field": "cerința de sprijin examinată", "field_passed": "cerința examinată satisfăcută",
}
EVENTS = {"grant": "acord emis", "revoke": "acord anulat", "pause": "acord pus în așteptare",
          "resume": "acord repus în vigoare"}
FRAMES = ("Registru simbolic de proprietăți", "Condiții inventate pentru un control")


def render(facts: dict, seed: int, symbol: str) -> str:
    rng = random.Random(seed)
    entries = []
    for key, value in facts.items():
        shown = (FIELD_NAMES[value] if key == "field" else EVENTS[value] if key == "last_event"
                 else "bifat" if value is True else "nebifat" if value is False else str(value))
        entries.append(f"{TERMS[key]}: {shown}")
    rng.shuffle(entries)
    return (f"{rng.choice(FRAMES)} {symbol}: " + " · ".join(entries) +
            ". Notația descrie doar stări fictive, nu conversații.")


def build_rows(batch_id: str = BATCH_ID) -> list[dict]:
    if batch_id != BATCH_ID:
        raise ValueError("Unknown cross-category holdout")
    rng = random.Random(202020)
    rows = []
    for code, builder in BUILDERS.items():
        for index in range(12):
            symbol = f"caz-simbolic-{len(rows) // 2 + 1:03d}"
            kind, negative, positive, judge = builder(index + 192, rng, symbol)
            if (set(negative) != set(positive) or
                    sum(negative[key] != positive[key] for key in negative) != 1 or
                    judge(negative) or not judge(positive)):
                raise AssertionError(f"Invalid one-fact reference pair: {symbol}")
            seed = rng.randrange(1 << 30)
            for variant, facts in enumerate((negative, positive)):
                rows.append({"id": f"{batch_id}-ro-{len(rows) + 1:04d}", "language": "ro",
                             "source_kind": kind, "split": "test", "text": render(facts, seed, symbol),
                             "labels": [code] if variant else [], "origin": "abstract_cross_category_holdout_v1"})
    assert len(rows) == 144
    return rows


def main() -> None:
    stem = ROOT / "data" / "synthetic" / BATCH_ID
    paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Cross-category holdout already exists")
    rows = build_rows()
    paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                for row in rows), encoding="utf-8")
    table = ["# batch-0020 — Romanian cross-category holdout", "",
             "Stare: în așteptarea validării automate. Split: `test`. 144 de fișe simbolice fără comunicări reale ori realiste.", "",
             "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
    table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                 for row in rows)
    table.extend(("", "Perechile alăturate diferă într-un singur fapt verificat printr-o regulă tipizată.", ""))
    paths[1].write_text("\n".join(table), encoding="utf-8")
    paths[2].write_text(json.dumps(candidate_manifest(ROOT, BATCH_ID), ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8")
    print(f"{BATCH_ID}: {len(rows)} test cards")


if __name__ == "__main__":
    main()
