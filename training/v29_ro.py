"""Additional Romanian abstract wording for the V29 prospective test."""

from training.v28_ro import D1_STYLES as PRIOR_D1, FIELDS, S1_STYLES as PRIOR_S1

D1_STYLES = PRIOR_D1 + (
    {
        "open": "Completare ipotetică a câmpurilor de observație",
        "role": ("sursa din fișă este un minor", "sursa din fișă este un adult"),
        "anchor": ("indicele de protecție are valoare confirmată", "indicele de protecție are valoare neconfirmată"),
        "indirect": ("apropierea treptată de tema sprijinului este bifată", "apropierea treptată de tema sprijinului nu este bifată"),
        "explicit": ("solicitarea fără ocol de ajutor este bifată", "solicitarea fără ocol de ajutor nu este bifată"),
        "close": "Este o schemă de valori, fără relatarea vreunei persoane.",
    },
)

S1_STYLES = PRIOR_S1 + (
    {
        "open": "Completare ipotetică a controlului de răspuns",
        "field": "rubrica examinată se referă la {}",
        "applicable": ("rubrica este necesară în evaluare", "rubrica nu este necesară în evaluare"),
        "passed": ("rezultatul rubricii este valid", "rezultatul rubricii este invalid"),
        "close": "Celelalte rubrici necesare sunt valide; nu este prezent un răspuns.",
    },
)
