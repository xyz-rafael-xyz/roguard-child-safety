"""Abstract Romanian D1/S1 prose families for a separately tested repair study."""

from __future__ import annotations

import json
import random
import runpy
from pathlib import Path

from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = runpy.run_path(str(Path(__file__).with_name("generate_compositional.py")))
BUILDERS = {code: REFERENCE["BUILDERS"][code] for code in ("D1", "S1")}
FIELD_NAMES = REFERENCE["FIELD_NAMES"]
SPECS = {
    "batch-0028": ("train", 24, 4, 804),
    "batch-0029": ("dev", 12, 1, 911),
    "batch-0030": ("test", 24, 1, 1031),
}
INTRO = (
    "Inventar de relații ipotetice", "Controlul proprietăților unei fișe fictive",
    "Rezumat al atributelor declarate", "Schemă de verificare a semnalelor",
    "Notă de dezvoltare pentru câmpuri abstracte", "Nouă fișă de control conceptual",
)
TAIL = (
    "Nicio formulare personală nu este reprodusă.",
    "Sunt numai stări inventate, fără dialog.",
    "Fișa nu include o comunicare propriu-zisă.",
    "Sunt consemnate exclusiv atribute de test.",
    "Nu există mesaj sau răspuns redat.",
    "Textul descrie reguli, nu vorbe ale unei persoane.",
)
ROLE = (
    ("rolul de origine este minor", "rolul de origine este adult"),
    ("sursa a fost marcată minoră", "sursa a fost marcată adultă"),
    ("proveniența minoră este confirmată", "proveniența este adultă"),
    ("fișa atribuie sursa unui minor", "fișa atribuie sursa unui adult"),
    ("identitatea rolului sursă: minor", "identitatea rolului sursă: adult"),
    ("pentru sursă figurează statutul minor", "pentru sursă figurează statutul adult"),
)
ANCHOR = (
    ("reperul de siguranță există", "reperul de siguranță lipsește"),
    ("legătura cu sprijinul este prezentă", "legătura cu sprijinul este absentă"),
    ("contextul de ajutor este consemnat", "contextul de ajutor nu este consemnat"),
    ("se confirmă un indiciu privind siguranța", "nu se confirmă un indiciu privind siguranța"),
    ("ancora de sprijin: prezentă", "ancora de sprijin: absentă"),
    ("a fost marcat un reper de protecție", "nu a fost marcat un reper de protecție"),
)
PATTERN = (
    ("tiparul indirect ori reluat există", "tiparul indirect ori reluat lipsește"),
    ("apropierea treptată de sprijin este marcată", "apropierea treptată de sprijin nu este marcată"),
    ("revenirea la tema ajutorului este prezentă", "revenirea la tema ajutorului este absentă"),
    ("se confirmă un semnal indirect de căutare a sprijinului", "nu se confirmă un semnal indirect de căutare a sprijinului"),
    ("traseul indirect: prezent", "traseul indirect: absent"),
    ("fișa notează o apropiere indirectă de ajutor", "fișa nu notează o apropiere indirectă de ajutor"),
)
EXPLICIT = (
    ("cererea directă de sprijin există", "cererea directă de sprijin lipsește"),
    ("solicitarea fără ocol este marcată", "solicitarea fără ocol nu este marcată"),
    ("ajutorul este cerut explicit în fișă", "ajutorul nu este cerut explicit în fișă"),
    ("se confirmă un indicator direct de sprijin", "nu se confirmă un indicator direct de sprijin"),
    ("solicitarea directă: prezentă", "solicitarea directă: absentă"),
    ("fișa notează sprijin cerut direct", "fișa nu notează sprijin cerut direct"),
)
FIELD = (
    "câmpul de răspuns urmărit este {}", "este controlată componenta {}",
    "cerința aplicabilă privește {}", "verificarea vizează {}",
    "elementul de sprijin ales: {}", "pentru această fișă se examinează {}",
)
PASSED = (
    ("cerința urmărită este satisfăcută", "cerința urmărită este încălcată"),
    ("componenta a trecut controlul", "componenta nu a trecut controlul"),
    ("se confirmă îndeplinirea câmpului", "nu se confirmă îndeplinirea câmpului"),
    ("elementul evaluat este prezent corect", "elementul evaluat lipsește ori este greșit"),
    ("starea câmpului: îndeplinit", "starea câmpului: neîndeplinit"),
    ("fișa confirmă respectarea cerinței", "fișa constată nerespectarea cerinței"),
)
OTHER = (
    ("celelalte cerințe sunt respectate", "celelalte cerințe nu sunt respectate"),
    ("restul câmpurilor sunt în regulă", "restul câmpurilor nu sunt în regulă"),
    ("se confirmă restul obligațiilor", "nu se confirmă restul obligațiilor"),
    ("condițiile auxiliare au trecut", "condițiile auxiliare nu au trecut"),
    ("alte verificări: îndeplinite", "alte verificări: neîndeplinite"),
    ("fișa păstrează restul pașilor", "fișa nu păstrează restul pașilor"),
)


def render(code: str, facts: dict, seed: int, symbol: str, style: int) -> str:
    if code == "D1":
        clauses = [
            ROLE[style][facts["role"] != "minor"],
            ANCHOR[style][not facts["anchor"]],
            PATTERN[style][not facts["pattern"]],
            EXPLICIT[style][not facts["explicit"]],
        ]
    else:
        clauses = [
            FIELD[style].format(FIELD_NAMES[facts["field"]]),
            PASSED[style][not facts["field_passed"]],
            OTHER[style][not facts["support_ok"]],
        ]
    random.Random(seed).shuffle(clauses)
    return f"{INTRO[style]} {symbol}. " + "; ".join(clauses) + f". {TAIL[style]}"


def build_rows(batch_id: str) -> list[dict]:
    if batch_id not in SPECS:
        raise ValueError("Unknown advisory repair batch")
    split, pair_count, surfaces, offset = SPECS[batch_id]
    rng = random.Random(928000 + int(batch_id[-4:]))
    style_offset = 0 if split == "train" else 4 if split == "dev" else 5
    rows = []
    for code, builder in BUILDERS.items():
        for index in range(pair_count):
            for variation in range(surfaces):
                symbol = f"ipoteză-{batch_id[-4:]}-{len(rows) // 2 + 1:04d}"
                state_rng = random.Random(offset + index * 17 + (0 if code == "D1" else 271))
                kind, negative, positive, judge = builder(index + offset, state_rng, symbol)
                if (set(negative) != set(positive) or
                        sum(negative[key] != positive[key] for key in negative) != 1 or
                        judge(negative) or not judge(positive)):
                    raise AssertionError(f"Invalid advisory repair pair: {symbol}")
                seed = rng.randrange(1 << 30)
                for variant, facts in enumerate((negative, positive)):
                    rows.append({
                        "id": f"{batch_id}-ro-{len(rows) + 1:05d}", "language": "ro",
                        "source_kind": kind, "split": split,
                        "text": render(code, facts, seed, symbol, style_offset + variation),
                        "labels": [code] if variant else [], "origin": "abstract_advisory_repair_v1",
                    })
    assert len(rows) == 4 * pair_count * surfaces
    return rows


def main() -> None:
    for batch_id, (split, _, _, _) in SPECS.items():
        stem = ROOT / "data/synthetic" / batch_id
        paths = [stem.with_suffix(suffix) for suffix in (".jsonl", ".preview.md", ".review.json")]
        if any(path.exists() for path in paths):
            raise FileExistsError("Preserve existing advisory repair batch")
        rows = build_rows(batch_id)
        paths[0].write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                                    for row in rows), encoding="utf-8")
        table = [f"# {batch_id} — fișe abstracte pentru controlul transferului", "",
                 f"Stare: în așteptarea validării automate. Split: `{split}`. {len(rows)} fișe simbolice fără comunicări reale ori realiste.", "",
                 "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
        table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |"
                     for row in rows)
        table.extend(("", "Perechile adiacente au o singură diferență verificată prin regula tipizată.", ""))
        paths[1].write_text("\n".join(table), encoding="utf-8")
        paths[2].write_text(json.dumps(candidate_manifest(ROOT, batch_id), ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
        print(f"{batch_id}: {len(rows)} {split} cards")


if __name__ == "__main__":
    main()
