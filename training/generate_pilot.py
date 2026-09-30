"""Generate three reviewable Romanian abstract-card pilot batches.

The descriptors contain no conversation, reported case, or child-like utterance.
Labels for structured cards come from the pure reference checks. All outputs
start pending review; this program never approves or loads a batch for training.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from roguard.policy import (BoundaryCard, GateCard, PermissionCard,
                            PermissionEvent, RoutingCard, check_boundary,
                            check_gate, check_permission, check_routing)
from roguard.review import candidate_manifest

ROOT = Path(__file__).resolve().parents[1]
SPLITS = (("batch-0005", "train", 12), ("batch-0006", "dev", 8),
          ("batch-0007", "test", 8))
OPENERS = {
    "train": "Fișă abstractă",
    "dev": "Descriere schematică",
    "test": "Înregistrare simbolică",
}


def _choose(items: list, count: int) -> list:
    if count == 12:
        return items
    # Both outcomes and the two P1 card types remain represented in each split.
    return [items[index] for index in (0, 1, 2, 3, 6, 7, 8, 9)]


def disclosure_cards(split: str, count: int) -> list[tuple[str, list[str], str]]:
    # source_is_minor, safety_anchor, abstract pattern, supportive pattern
    cases = {
        "train": [
            (True, True, "apropiere indirectă", True), (True, True, "reluare graduală", True),
            (True, True, "fragment urmat de retragere", True), (True, True, "cerere de sprijin descrisă în fișă", True),
            (True, True, "ezitare urmată de reluare", True), (True, True, "două intervenții legate", True),
            (True, False, "apropiere indirectă", True), (False, True, "apropiere indirectă", True),
            (True, True, "schimbare simplă de subiect", False), (False, False, "schimbare simplă de subiect", False),
            (True, False, "ezitare urmată de reluare", True), (False, True, "fragment urmat de retragere", True),
        ],
        "dev": [
            (True, True, "aluzie urmată de revenire", True), (True, True, "apropiere în două etape", True),
            (True, True, "enunț incomplet și reluare", True), (True, True, "solicitare de sprijin consemnată", True),
            (True, False, "aluzie urmată de revenire", True), (False, True, "apropiere în două etape", True),
            (True, True, "întrebare generală fără cerere de sprijin", False), (False, False, "pauză fără continuare", False),
        ],
        "test": [
            (True, True, "abordare prin ocolire", True), (True, True, "secvență discontinuă reluată", True),
            (True, True, "retragere după început", True), (True, True, "apel la sprijin descris abstract", True),
            (False, True, "abordare prin ocolire", True), (True, False, "secvență discontinuă reluată", True),
            (True, True, "tranziție obișnuită de temă", False), (False, False, "referire neutră fără continuare", False),
        ],
    }[split]
    cards = []
    for minor, anchor, pattern, supportive in cases:
        source = "minorul" if minor else "un adult"
        context = "există reper contextual de siguranță" if anchor else "nu există reper contextual de siguranță"
        text = f"{OPENERS[split]}: sursa declarată este {source}; {context}; tiparul consemnat este {pattern}; niciun mesaj nu este reprodus."
        positive = minor and anchor and supportive
        cards.append(("message", ["D1"] if positive else [], text))
    return cards


def routing_cards(split: str, count: int) -> list[tuple[str, list[str], str]]:
    # allowed role, proposed role, policy purpose, request purpose, policy present
    cases = [
        ("minor", "tutore", "A", "A", True), ("tutore", "minor", "A", "A", True),
        ("minor", "minor", "A", "B", True), ("tutore", "tutore", "B", "A", True),
        ("minor", "tutore", "B", "A", True), ("tutore", "minor", "B", "B", True),
        ("minor", "minor", "A", "A", True), ("tutore", "tutore", "A", "A", True),
        ("minor", "minor", "B", "B", True), ("tutore", "tutore", "B", "B", True),
        ("minor", "minor", "C", "C", True), ("tutore", "tutore", "C", "C", True),
    ]
    cards = []
    chosen = _choose(cases, count) if split != "test" else [cases[i] for i in (4, 5, 2, 3, 8, 9, 10, 11)]
    purpose_map = {"train": {"A": "A", "B": "B", "C": "C"},
                   "dev": {"A": "C", "B": "D", "C": "A"},
                   "test": {"A": "E", "B": "F", "C": "G"}}[split]
    for allowed, proposed, rule_purpose, request_purpose, present in chosen:
        rule_purpose = purpose_map[rule_purpose]
        request_purpose = purpose_map[request_purpose]
        card = RoutingCard("minor", "element-simbolic", request_purpose, proposed,
                           frozenset({"minor", "tutore"}),
                           {("minor", "element-simbolic", rule_purpose): frozenset({allowed})} if present else None)
        decision = check_routing(card)
        if split == "train":
            text = (f"{OPENERS[split]}: regula fictivă permite destinatarul {allowed} pentru scopul {rule_purpose}; "
                    f"cererea indică destinatarul {proposed} și scopul {request_purpose}; principalul minor și elementul simbolic sunt declarate.")
        elif split == "dev":
            text = (f"{OPENERS[split]}: scopul cererii este {request_purpose}, destinatarul propus este {proposed}; "
                    f"politica simbolică autorizează {allowed} numai pentru scopul {rule_purpose}; principalul minor și elementul simbolic sunt cunoscute.")
        else:
            text = (f"{OPENERS[split]}: destinatarul admis de politica fictivă este {allowed} la scopul {rule_purpose}; "
                    f"solicitarea are scopul {request_purpose} și ruta către {proposed}; principalul minor și elementul simbolic sunt definite.")
        cards.append(("routing_card", ["R1"] if decision.review_suggested else [], text))
    return cards


def response_cards(split: str, count: int) -> list[tuple[str, list[str], str]]:
    # word cap, measured length, required next step, present next step, exclusive-system advice
    cases = [
        (60, 85, True, True, False), (50, 68, True, False, False),
        (70, 90, False, True, True), (80, 95, True, True, False),
        (40, 55, True, False, False), (90, 110, True, True, True),
        (60, 45, True, False, False), (50, 40, True, True, True),
        (70, 60, True, True, False), (80, 65, False, True, False),
        (40, 35, True, True, False), (90, 75, True, True, False),
    ]
    cards = []
    chosen = _choose(cases, count) if split != "test" else [cases[i] for i in (4, 5, 2, 3, 8, 9, 6, 7)]
    numeric_shift = {"train": 0, "dev": 7, "test": 13}[split]
    ages = {"train": (7, 9, 12), "dev": (8, 10, 13), "test": (6, 11, 14)}[split]
    for n, (cap, measured, required, present, exclusive) in enumerate(chosen, 1):
        cap += numeric_shift
        measured += numeric_shift
        age = ages[(n - 1) % 3]
        required_text = "pasul de sprijin este cerut" if required else "pasul de sprijin nu este cerut"
        present_text = "pasul este prezent" if present else "pasul lipsește"
        relation = ("varianta descurajează sprijinul uman" if exclusive else
                    "varianta nu descurajează sprijinul uman")
        if split == "train":
            text = (f"{OPENERS[split]}: contract fictiv pentru vârsta {age}; "
                    f"plafon artificial {cap} cuvinte, număr măsurat {measured}; {required_text}; "
                    f"{present_text}; {relation}; răspunsul nu este inclus.")
        else:
            text = (f"{OPENERS[split]}: vârsta declarată este {age}; varianta are {measured} cuvinte față de limita "
                    f"experimentală {cap}; {present_text}; {required_text}; {relation}; nu se redă răspunsul.")
        labels = (["A1"] if measured > cap else []) + (["S1"] if required and not present or exclusive else [])
        cards.append(("response", labels, text))
    return cards


def persistence_cards(split: str, count: int) -> list[tuple[str, list[str], str]]:
    # Six permission transitions, then six boundary probes.
    permission_cases = [
        ("revoke", True), ("grant", False), ("pause", True),
        ("grant", True), ("revoke", False), ("pause", False),
    ]
    boundary_cases = [
        (True, True, True, True), (True, True, False, True),
        (True, True, False, False), (False, False, False, False),
        (True, False, False, False), (False, True, False, False),
    ]
    all_cards = []
    purpose = {"train": "A", "dev": "C", "test": "E"}[split]
    for n, (last_action, proposed_use) in enumerate(permission_cases, 1):
        events = (PermissionEvent(1, "minor", f"item-{n}", purpose, "recipient-x", "grant"),
                  PermissionEvent(2, "minor", f"item-{n}", purpose, "recipient-x", last_action))
        card = PermissionCard("minor", f"item-{n}", purpose, "recipient-x", 3, proposed_use, events)
        decision = check_permission(card)
        proposal = "reutilizare" if proposed_use else "neutilizare"
        action = {"grant": "acord nou", "revoke": "revocare", "pause": "pauză"}[last_action]
        if split == "train":
            text = (f"{OPENERS[split]}: pentru același element, persoană, destinatar și scop {purpose}, "
                    f"ultimul eveniment este {action}; propunerea curentă este {proposal}; ordinea evenimentelor este completă.")
        elif split == "dev":
            text = (f"{OPENERS[split]}: istoricul simbolic pentru scopul {purpose} se încheie prin {action}; "
                    f"în etapa următoare este propusă {proposal}; elementul, persoana și destinatarul coincid cu cele din istoric.")
        else:
            text = (f"{OPENERS[split]}: persoana, elementul și destinatarul sunt aceleași la scopul {purpose}; "
                    f"după {action}, varianta propune {proposal}; cronologia fictivă este fixată.")
        all_cards.append(("permission_card", ["P1"] if decision.review_suggested else [], text))
    for n, (corrected, renewed, disclose, withhold) in enumerate(boundary_cases, 7):
        proposal = ({"needed"} if not withhold else set()) | ({"protected"} if disclose else set())
        card = BoundaryCard(frozenset({"needed"}), frozenset({"protected"}),
                            frozenset({"needed"}), frozenset(proposal), corrected, renewed)
        decision = check_boundary(card)
        correction = "a existat o corecție" if corrected else "nu a existat corecție"
        pressure = "terțul a repetat cererea" if renewed else "terțul nu a repetat cererea"
        protected = "inclus" if disclose else "omis"
        needed = "omis" if withhold else "păstrat"
        if split == "train":
            text = (f"{OPENERS[split]}: {correction}; {pressure}; câmpul protejat este {protected} "
                    f"în propunere, iar câmpul permis și necesar este {needed}; numai identificatori fictivi.")
        elif split == "dev":
            text = (f"{OPENERS[split]}: {pressure}; {correction}; în propunere, câmpul protejat este {protected}, "
                    f"iar cel necesar și permis este {needed}; identificatorii sunt simbolici.")
        else:
            text = (f"{OPENERS[split]}: limita pentru câmpul protejat a fost declarată; {correction}; "
                    f"{pressure}; în varianta de după aceste evenimente, câmpul protejat este {protected}, "
                    f"iar cel permis și necesar este {needed}.")
        all_cards.append(("boundary_card", ["P1"] if decision.review_suggested else [], text))
    if count == 8:
        indices = (0, 1, 3, 4, 6, 7, 8, 9) if split == "dev" else (0, 1, 3, 5, 6, 7, 9, 11)
        return [all_cards[index] for index in indices]
    return all_cards


def gate_cards(split: str, count: int) -> list[tuple[str, list[str], str]]:
    # parsed, accept malformed output, external branch, human approval required, owner
    cases = [
        (True, False, True, False, True), (False, True, False, True, True),
        (False, True, True, False, True), (True, False, True, True, False),
        (True, False, True, False, False), (False, True, True, True, True),
        (True, False, False, True, True), (False, False, False, True, True),
        (True, False, True, True, True), (True, False, False, False, True),
        (False, False, True, True, True), (False, False, False, False, True),
    ]
    cards = []
    chosen = _choose(cases, count) if split != "test" else [cases[i] for i in (4, 5, 1, 3, 6, 8, 10, 11)]
    for parsed, accept_bad, external, approval, owner in chosen:
        card = GateCard("evaluator" if owner else None, parsed, accept_bad, external, approval)
        decision = check_gate(card)
        parse_text = "parsată" if parsed else "neparsată"
        accept_text = "acceptată" if accept_bad else "reținută"
        external_text = "există ramură externă" if external else "nu există ramură externă"
        human_text = "aprobarea umană este obligatorie" if approval else "aprobarea umană nu este obligatorie"
        owner_text = "evaluatorul este numit" if owner else "evaluatorul nu este numit"
        if split == "train":
            text = (f"{OPENERS[split]}: ieșirea este {parse_text}, iar ieșirea neparsată este {accept_text}; "
                    f"{external_text}; {human_text}; {owner_text}; fișa descrie numai fluxul propus.")
        elif split == "dev":
            text = (f"{OPENERS[split]}: {external_text}; {owner_text}; "
                    f"{human_text}; ieșirea curentă este {parse_text}, iar o ieșire neparsată ar fi {accept_text}.")
        else:
            text = (f"{OPENERS[split]}: {owner_text}; {human_text}; o ieșire neparsată este {accept_text}; "
                    f"{external_text}; ieșirea curentă este {parse_text}; nu se execută fluxul descris.")
        cards.append(("gate_card", ["G1"] if decision.review_suggested else [], text))
    return cards


def build(split: str, count: int) -> list[tuple[str, list[str], str]]:
    cards = []
    for generator in (disclosure_cards, routing_cards, response_cards, persistence_cards, gate_cards):
        cards.extend(generator(split, count))
    random.Random({"train": 507, "dev": 608, "test": 709}[split]).shuffle(cards)
    return cards


def main() -> None:
    paths = []
    for batch_id, split, count in SPLITS:
        output = ROOT / "data" / "synthetic" / f"{batch_id}.jsonl"
        preview = output.with_suffix(".preview.md")
        review = output.with_suffix(".review.json")
        paths.extend((output, preview, review))
    if any(path.exists() for path in paths):
        raise FileExistsError("Pilot batch already exists; reviewed bytes are never overwritten")
    for batch_id, split, count in SPLITS:
        output = ROOT / "data" / "synthetic" / f"{batch_id}.jsonl"
        preview = output.with_suffix(".preview.md")
        review = output.with_suffix(".review.json")
        rows = []
        for index, (kind, labels, body) in enumerate(build(split, count), 1):
            rows.append({"id": f"{batch_id}-ro-{index:03d}", "language": "ro",
                         "source_kind": kind, "split": split, "text": body,
                         "labels": labels, "origin": "abstract_template_v3"})
        output.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
        table = [f"# {batch_id} — fișă pentru revizuire", "",
                 f"Stare: în așteptare. Split: `{split}`. {len(rows)} fișe abstracte; niciun mesaj sau răspuns realist nu este inclus.", "",
                 "| ID | Limbă | Tip | Etichete | Text |", "|---|---|---|---|---|"]
        table += [f"| {row['id']} | ro | {row['source_kind']} | {','.join(row['labels']) or 'none'} | {row['text']} |" for row in rows]
        table += ["", "Revizuiți fiecare rând. Aprobarea acestui lot nu aprobă alte loturi.", ""]
        preview.write_text("\n".join(table), encoding="utf-8")
        review.write_text(json.dumps(candidate_manifest(ROOT, batch_id), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{batch_id}: {len(rows)} {split} cards")


if __name__ == "__main__":
    main()
