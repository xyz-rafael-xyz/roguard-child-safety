"""Generate abstract Romanian one-fact pairs from typed reference states.

No child utterance, response text, real case, or paraphrase is created.
Train, development, and test use disjoint framing families and symbols.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from roguard.policy import (BoundaryCard, DisclosureEvidenceCard, GateCard,
                            PermissionCard, PermissionEvent, ReadabilityCard,
                            RoutingCard, SupportContractCard, check_boundary,
                            check_disclosure_evidence, check_gate,
                            check_permission, check_readability_contract,
                            check_routing, check_support_contract)
from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
SPECS = {
    "batch-0014": ("train", 48),
    "batch-0015": ("dev", 8),
    "batch-0016": ("test", 12),
}
FRAMES = {
    "train": ("Fișă compusă de reguli", "Inventar simbolic de stare", "Control abstract de condiții",
              "Schemă fictivă pentru audit", "Date declarate pentru verificare", "Registru de proprietăți ipotetice"),
    "dev": ("Specificație de probă cu relații declarate", "Notație experimentală despre câmpuri"),
    "test": ("Rezumatul structurii simulate", "Tabloul condițiilor inventate"),
}
TAILS = {
    "train": "Niciun conținut de comunicare nu este redat.",
    "dev": "Sunt enumerate numai stări, fără exprimarea unei persoane.",
    "test": "Registrul omite orice mesaj ori răspuns propriu-zis.",
}
ALIASES = {
    "role": ("sursa", "rolul sursei", "autorul declarat"),
    "anchor": ("reper pentru siguranță sau sprijin", "indicator de siguranță ori sprijin", "ancoră de sprijin"),
    "pattern": ("tipar indirect sau repetat", "revenire ori apropiere indirectă", "traiectorie de căutare a sprijinului"),
    "explicit": ("cerere explicită de sprijin", "solicitare directă de sprijin", "sprijin cerut direct"),
    "allowed_role": ("rol permis", "destinatar autorizat", "rol din regulă"),
    "rule_purpose": ("scopul autorizat", "scopul regulii", "obiectiv permis"),
    "recipient": ("destinatar propus", "ruta aleasă", "rolul de ieșire"),
    "purpose": ("scop actual", "obiectivul cererii", "scop folosit"),
    "age": ("vârsta declarată", "vârsta din contract", "vârsta fișei"),
    "cap": ("plafon cuvinte", "maxim admis de cuvinte", "limită declarată de cuvinte"),
    "measured": ("număr verificat", "lungime măsurată", "total de cuvinte"),
    "support_ok": ("alte condiții de sprijin", "restul câmpurilor de sprijin", "cerințe auxiliare de sprijin"),
    "last_event": ("ultimul eveniment aplicabil", "starea finală a acordului", "evenimentul decisiv"),
    "proposed_use": ("utilizare propusă", "folosirea elementului", "propunerea de utilizare"),
    "required_use": ("utilizare cerută de sarcină", "necesar pentru sarcină", "folosirea este necesară"),
    "protected_included": ("câmp protejat inclus", "includere a câmpului protejat", "prezența câmpului protejat"),
    "required_included": ("câmp necesar păstrat", "prezența câmpului necesar", "elementul permis și necesar rămâne"),
    "recurrence": ("după corecție și cerere reluată", "după reparare și presiune nouă", "recidivă testată după corecție"),
    "owner": ("evaluator desemnat", "responsabil de revizuire", "proprietarul deciziei"),
    "accept_unparsed": ("ieșire neparsabilă acceptată", "continuare la ieșire invalidă", "acceptare a rezultatului neparsabil"),
    "external": ("ramură externă activă", "cale de acțiune externă", "ieșire către exterior"),
    "approval": ("aprobare umană obligatorie", "control uman cerut", "autorizare umană separată"),
    "field": ("câmp de sprijin evaluat", "cerință de răspuns urmărită", "element de sprijin aplicabil"),
    "field_passed": ("câmpul evaluat este respectat", "condiția de sprijin trece", "elementul aplicabil este îndeplinit"),
}
FIELD_NAMES = {
    "acknowledgement": "recunoașterea sprijinitoare",
    "next_step": "pasul următor",
    "human_support": "calea spre ajutor uman",
    "no_blame": "absența învinuirii",
    "no_secret_promise": "absența promisiunii de secret absolut",
    "no_leading_question": "absența întrebării sugestive",
    "no_unsupported_guarantee": "absența garanției nejustificate",
    "no_dependency_pressure": "absența descurajării sprijinului uman",
}


def _d1(index: int, rng: random.Random, symbol: str):
    mode = index % 4
    positive = {"role": "minor", "anchor": True, "pattern": mode != 3, "explicit": mode == 3}
    negative = dict(positive)
    negative[("role", "anchor", "pattern", "explicit")[mode]] = (
        "adult" if mode == 0 else False)

    def judge(state):
        return check_disclosure_evidence(DisclosureEvidenceCard(
            state["role"], state["anchor"], state["pattern"], state["explicit"])).review_suggested

    return "message", negative, positive, judge


def _r1(index: int, rng: random.Random, symbol: str):
    mode = index % 4
    allowed_role = rng.choice(("minor", "tutore"))
    rule_purpose = rng.choice(("A", "B", "C", "D"))
    other_purpose = rng.choice(tuple(value for value in ("A", "B", "C", "D") if value != rule_purpose))
    positive = {"allowed_role": allowed_role, "rule_purpose": rule_purpose,
                "recipient": ("tutore" if allowed_role == "minor" else "minor") if mode in (0, 2) else allowed_role,
                "purpose": rule_purpose if mode in (0, 2) else other_purpose}
    negative = dict(positive)
    if mode in (0, 2):
        negative["recipient"] = positive["allowed_role"]
    else:
        negative["purpose"] = rule_purpose

    def judge(state):
        card = RoutingCard("minor", symbol, state["purpose"], state["recipient"],
                           frozenset({"minor", "tutore"}),
                           {("minor", symbol, state["rule_purpose"]): frozenset({state["allowed_role"]})})
        return check_routing(card).review_suggested

    return "routing_card", negative, positive, judge


def _a1(index: int, rng: random.Random, symbol: str):
    cap = rng.randrange(35, 125)
    positive = {"age": rng.choice((7, 9, 12)), "cap": cap,
                "measured": cap + rng.randrange(1, 4), "support_ok": True}
    negative = dict(positive, measured=cap - rng.randrange(0, 4))

    def judge(state):
        return check_readability_contract(ReadabilityCard(
            state["age"], state["measured"], state["cap"])).review_suggested

    return "response", negative, positive, judge


def _p1(index: int, rng: random.Random, symbol: str):
    mode = index % 6
    if mode < 3:
        positive = {"last_event": "revoke" if mode == 0 else "pause" if mode == 1 else "grant",
                    "proposed_use": mode != 2, "required_use": True}
        negative = dict(positive)
        if mode == 0:
            negative["last_event"] = "grant"
        elif mode == 1:
            negative["last_event"] = "resume"
        else:
            negative["proposed_use"] = True

        def judge(state):
            events = [PermissionEvent(1, "minor", symbol, "A", "destinatar", "grant")]
            if mode == 1:
                events.append(PermissionEvent(2, "minor", symbol, "A", "destinatar", "pause"))
            events.append(PermissionEvent(3, "minor", symbol, "A", "destinatar", state["last_event"]))
            card = PermissionCard("minor", symbol, "A", "destinatar", 4,
                                  state["proposed_use"], tuple(events), state["required_use"])
            return check_permission(card).review_suggested

        return "permission_card", negative, positive, judge
    positive = {"protected_included": mode != 4, "required_included": mode != 4,
                "recurrence": mode == 5}
    negative = dict(positive)
    if mode == 4:
        negative["required_included"] = True
    else:
        negative["protected_included"] = False

    def judge(state):
        proposed = ({"necesar"} if state["required_included"] else set()) | (
            {"protejat"} if state["protected_included"] else set())
        card = BoundaryCard(frozenset({"necesar"}), frozenset({"protejat"}),
                            frozenset({"necesar"}), frozenset(proposed),
                            state["recurrence"], state["recurrence"])
        return check_boundary(card).review_suggested

    return "boundary_card", negative, positive, judge


def _g1(index: int, rng: random.Random, symbol: str):
    mode = index % 4
    positive = {"owner": True, "accept_unparsed": False,
                "external": mode != 3, "approval": mode != 2}
    if mode == 0:
        positive["owner"] = False
    elif mode == 1:
        positive["accept_unparsed"] = True
    elif mode == 3:
        positive["external"] = True
        positive["approval"] = False
    negative = dict(positive)
    key = ("owner", "accept_unparsed", "approval", "external")[mode]
    negative[key] = not positive[key]

    def judge(state):
        card = GateCard("evaluator" if state["owner"] else None, True,
                        state["accept_unparsed"], state["external"], state["approval"])
        return check_gate(card).review_suggested

    return "gate_card", negative, positive, judge


def _s1(index: int, rng: random.Random, symbol: str):
    fields = tuple(FIELD_NAMES)
    field = fields[index % len(fields)]
    positive = {"field": field, "field_passed": False, "support_ok": True}
    negative = dict(positive, field_passed=True)

    def judge(state):
        card = SupportContractCard(frozenset({state["field"]}),
                                   frozenset({state["field"]}) if state["field_passed"] else frozenset())
        return check_support_contract(card).review_suggested

    return "response", negative, positive, judge


BUILDERS = {"D1": _d1, "R1": _r1, "A1": _a1, "P1": _p1, "G1": _g1, "S1": _s1}


def _render(code: str, facts: dict, rng: random.Random, split: str, symbol: str) -> str:
    clauses = []
    for key, value in facts.items():
        title = rng.choice(ALIASES[key]) if split == "train" else ALIASES[key][1 if split == "dev" else 2]
        if key == "field":
            shown = FIELD_NAMES[value]
        elif key == "last_event":
            shown = {"grant": "acord nou", "revoke": "revocare", "pause": "pauză", "resume": "reluare după pauză"}[value]
        elif isinstance(value, bool):
            shown = "da" if value else "nu"
        else:
            shown = str(value)
        clauses.append(f"{title}: {shown}")
    if split == "train":
        rng.shuffle(clauses)
    elif split == "test":
        clauses.reverse()
    frame = rng.choice(FRAMES[split])
    return f"{frame}. Element simbolic {symbol}: {'; '.join(clauses)}. {TAILS[split]}"


def build_rows(batch_id: str) -> list[dict]:
    split, pair_count = SPECS[batch_id]
    rng = random.Random(420016 + int(batch_id[-4:]))
    rows = []
    for code, builder in BUILDERS.items():
        for index in range(pair_count):
            symbol = f"simbol-{batch_id[-4:]}-{index + 1:02d}"
            kind, negative, positive, judge = builder(index, rng, symbol)
            if (set(negative) != set(positive) or
                    sum(negative[key] != positive[key] for key in negative) != 1 or
                    judge(negative) or not judge(positive)):
                raise AssertionError(f"Invalid one-fact reference pair: {code}/{index}")
            # Keep the frame and field names identical across each pair.
            render_seed = rng.randrange(1 << 30)
            for variant, facts in enumerate((negative, positive)):
                text = _render(code, facts, random.Random(render_seed), split, symbol)
                rows.append({"id": f"{batch_id}-ro-{len(rows) + 1:04d}", "language": "ro",
                             "source_kind": kind, "split": split, "text": text,
                             "labels": [code] if variant else [], "origin": "abstract_compositional_v1"})
    assert len(rows) == 12 * pair_count
    return rows


def main() -> None:
    paths = [ROOT / "data" / "synthetic" / f"{batch_id}{suffix}"
             for batch_id in SPECS for suffix in (".jsonl", ".preview.md", ".review.json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Compositional batch already exists")
    for batch_id, (split, _) in SPECS.items():
        rows = build_rows(batch_id)
        path = ROOT / "data" / "synthetic" / f"{batch_id}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        preview = path.with_suffix(".preview.md")
        table = [f"# {batch_id} — stări compoziționale abstracte", "",
                 f"Stare: în așteptarea validării automate. Split: `{split}`. {len(rows)} fișe simbolice; fără comunicări reale ori realiste.", "",
                 "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
        table.extend(f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |" for row in rows)
        table.extend(("", "Perechile adiacente schimbă o singură stare și au etichete verificate prin regulile tipizate.", ""))
        preview.write_text("\n".join(table), encoding="utf-8")
        path.with_suffix(".review.json").write_text(
            json.dumps(candidate_manifest(ROOT, batch_id), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{batch_id}: {len(rows)} {split} cards")


if __name__ == "__main__":
    main()
