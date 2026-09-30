"""Expanded Romanian abstract-card train and development surfaces.

The training lexicon deliberately incorporates wording families from consumed
synthetic tests 0018–0020. No earlier test rows or labels enter this generator.
"""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILDERS = REFERENCE["BUILDERS"]
FIELD_NAMES = REFERENCE["FIELD_NAMES"]
OLD_ALIASES = REFERENCE["ALIASES"]
PRIOR_LEXICONS = tuple(runpy.run_path(str(Path(__file__).with_name(name)))[key] for name, key in (
    ("generate_balanced_holdout.py", "NAMES"),
    ("generate_encoder_holdout.py", "HEADINGS"),
    ("generate_cross_category_holdout.py", "TERMS"),
))
SPECS = {"batch-0021": ("train", 48, 4), "batch-0022": ("dev", 12, 1)}
TRAIN_LABELS = {key: tuple(dict.fromkeys((*OLD_ALIASES[key], *(lexicon[key] for lexicon in PRIOR_LEXICONS))))
                for key in OLD_ALIASES}
DEV_LABELS = {
    "role": "identitatea rolului sursă", "anchor": "legătura cu nevoia de siguranță ori ajutor",
    "pattern": "revenirea indirectă la tema ajutorului", "explicit": "sprijinul solicitat fără ocol",
    "allowed_role": "rol acceptat de norma fișei", "rule_purpose": "motivul înscris în norma fișei",
    "recipient": "rol vizat de distribuire", "purpose": "motivul distribuției propuse",
    "age": "numărul de ani din contract", "cap": "pragul admis pentru cuvinte",
    "measured": "cuvintele din rezultatul măsurat", "support_ok": "restul condițiilor de sprijin trec",
    "last_event": "actul de permisiune cu efect final", "proposed_use": "folosirea din propunere",
    "required_use": "folosirea impusă de regulă", "protected_included": "includerea elementului rezervat",
    "required_included": "includerea elementului obligatoriu", "recurrence": "o nouă cerere după corecție",
    "owner": "rol responsabil de revizuire", "accept_unparsed": "admiterea ieșirii imposibil de interpretat",
    "external": "activarea traseului spre exterior", "approval": "decizia distinctă a unui om",
    "field": "obligația de sprijin urmărită", "field_passed": "obligația urmărită respectată",
}
EVENTS = {"grant": "acord nou", "revoke": "acord retras", "pause": "acord suspendat",
          "resume": "acord reluat"}
BOOL_TRAIN = (("da", "nu"), ("prezent", "absent"), ("adevărat", "fals"),
              ("bifat", "nebifat"), ("confirmat", "neconfirmat"), ("activ", "inactiv"))
BOOL_DEV = ("îndeplinit", "neîndeplinit")
FRAMES = {
    "train": ("Model de proprietăți inventate", "Tabel fictiv de control",
              "Stări ipotetice pentru clasificare", "Date simbolice pentru o regulă"),
    "dev": ("Registru nou de verificare", "Fișă ipotetică pentru compararea condițiilor"),
}
TAILS = {"train": "Nicio comunicare personală nu este inclusă.",
         "dev": "Sunt consemnate numai proprietăți inventate."}


def render(facts: dict, split: str, seed: int, symbol: str, style: int) -> str:
    rng = random.Random(seed)
    bool_true, bool_false = (BOOL_TRAIN[(style + rng.randrange(len(BOOL_TRAIN))) % len(BOOL_TRAIN)]
                             if split == "train" else BOOL_DEV)
    clauses = []
    for key, value in facts.items():
        title = rng.choice(TRAIN_LABELS[key]) if split == "train" else DEV_LABELS[key]
        shown = (FIELD_NAMES[value] if key == "field" else EVENTS[value] if key == "last_event"
                 else bool_true if value is True else bool_false if value is False else str(value))
        clauses.append(f"{title} = {shown}")
    rng.shuffle(clauses)
    separator = rng.choice(("; ", " / ", " · ")) if split == "train" else " • "
    return (f"{rng.choice(FRAMES[split])} {symbol}: " + separator.join(clauses) +
            f". {TAILS[split]}")


def build_rows(batch_id: str) -> list[dict]:
    if batch_id not in SPECS:
        raise ValueError("Unknown surface study batch")
    split, pair_count, surfaces = SPECS[batch_id]
    rng = random.Random(520021 + int(batch_id[-4:]))
    rows = []
    offset = 264 if split == "train" else 360
    for code, builder in BUILDERS.items():
        for index in range(pair_count):
            for style in range(surfaces):
                symbol = f"unitate-{batch_id[-4:]}-{len(rows) // 2 + 1:04d}"
                # Keep one typed state pair while varying its independent rendering.
                state_rng = random.Random(7341 + index * 101 + list(BUILDERS).index(code))
                kind, negative, positive, judge = builder(index + offset, state_rng, symbol)
                if (set(negative) != set(positive) or
                        sum(negative[key] != positive[key] for key in negative) != 1 or
                        judge(negative) or not judge(positive)):
                    raise AssertionError(f"Invalid surface pair: {symbol}")
                seed = rng.randrange(1 << 30)
                for variant, facts in enumerate((negative, positive)):
                    rows.append({"id": f"{batch_id}-ro-{len(rows) + 1:05d}", "language": "ro",
                                 "source_kind": kind, "split": split,
                                 "text": render(facts, split, seed, symbol, style),
                                 "labels": [code] if variant else [], "origin": "abstract_surface_matrix_v1"})
    assert len(rows) == 12 * pair_count * surfaces
    return rows


def main() -> None:
    paths = [ROOT / "data/synthetic" / f"{batch_id}{suffix}"
             for batch_id in SPECS for suffix in (".jsonl", ".preview.md", ".review.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Surface study batch already exists")
    for batch_id, (split, _, _) in SPECS.items():
        rows = build_rows(batch_id)
        stem = ROOT / "data/synthetic" / batch_id
        path = stem.with_suffix(".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                for row in rows), encoding="utf-8")
        table = [f"# {batch_id} — variații abstracte românești", "",
                 f"Stare: în așteptarea validării automate. Split: `{split}`. {len(rows)} fișe simbolice fără comunicări reale ori realiste.", "",
                 "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
        table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                     for row in rows)
        table.extend(("", "Perechile adiacente schimbă un singur fapt verificat de regula tipizată.", ""))
        stem.with_suffix(".preview.md").write_text("\n".join(table), encoding="utf-8")
        stem.with_suffix(".review.json").write_text(
            json.dumps(candidate_manifest(ROOT, batch_id), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8")
        print(f"{batch_id}: {len(rows)} {split} cards")


if __name__ == "__main__":
    main()
