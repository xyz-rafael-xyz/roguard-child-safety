"""Factor questions for abstract Romanian D1 evidence, without message content."""

from __future__ import annotations

FIELDS = ("minor_source", "support_anchor", "indirect_pattern", "explicit_request")
QUESTIONS = {
    "minor_source": "Fișa declară că sursa are rol de minor?",
    "support_anchor": "Fișa declară prezent un reper de siguranță ori sprijin?",
    "indirect_pattern": "Fișa declară prezent un tipar indirect ori repetat de căutare a sprijinului?",
    "explicit_request": "Fișa declară prezentă o solicitare directă de sprijin?",
}


def make_factor_prompt(text: str, field: str) -> str:
    if field not in FIELDS or not isinstance(text, str) or not text.strip():
        raise ValueError("Specify one known D1 factor and nonempty abstract text")
    return (
        "Evaluează un singur atribut dintr-o fișă simbolică. Respectă și negațiile. "
        "Răspunde da numai dacă atributul este afirmat în fișă; altfel nu. "
        f"Întrebare: {QUESTIONS[field]}\nFișă: {text}"
    )


def decision(factors: dict[str, bool]) -> bool:
    if set(factors) != set(FIELDS) or any(type(value) is not bool for value in factors.values()):
        raise ValueError("D1 decision requires four Boolean factor judgments")
    return (factors["minor_source"] and factors["support_anchor"] and
            (factors["indirect_pattern"] or factors["explicit_request"]))
