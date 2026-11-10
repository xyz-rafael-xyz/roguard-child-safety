"""Conservative, auditable scope corrections for an opt-in structured review."""
from __future__ import annotations

import re
import unicodedata

_MODAL = {
    'ro': re.compile(r'\bpoate\s+că\b|\bmă\s+întreb\s+ce\s+s-ar\s+întâmpla\s+dacă\b|\bnu\s+știu\s+dacă\b|\bnu\s+exclud\b|\be\s+posibil\b|\bnu\s+pot\s+stabili\s+dacă\s+această\s+descriere\s+este\s+adevărată\s+despre\s+mine\b'),
    'uk': re.compile(r'\bможливо\b|\bне\s+знаю\s*,?\s*чи\b|\bне\s+впевнен\s*,?\s*чи\b|\bне\s+виключаю\b|\bя\s+не\s+можу\s+визначити\s*,?\s+чи\s+цей\s+опис\s+відповідає\s+дійсності\s+щодо\s+мене\b'),
}
_ROLE = {
    'ro': {
        'self': (r'asta\s+mă\s+privește\s+pe\s+mine', r'mă\s+refer\s+la\s+propria\s+mea\s+situație'),
        'other': (r'povestesc\s+ce\s+mi-a\s+zis\s+o\s+prietenă', r'un\s+prieten\s+spune\s+că'),
        'fiction': (r'compun\s+un\s+dialog\s+pentru\s+un\s+roman',),
    },
    'uk': {
        'self': (r'це\s+стосується\s+мене',),
        'other': (r'розповідаю\s*,?\s*що\s+сказала\s+подруга', r'друг\s+каже\s*,?\s*що'),
        'fiction': (r'складаю\s+діалог\s+для\s+роману',),
    },
}
_TIME = {
    'ro': {
        0: (r'asta\s+e\s+situația\s+de\s+azi', r'vorbesc\s+despre\s+situația\s+de\s+acum'),
        1: (r'vorbesc\s+doar\s+despre\s+situația\s+de\s+anul\s+trecut',),
        2: (r'vorbesc\s+despre\s+o\s+situație\s+posibilă', r'mă\s+întreb\s+ce\s+s-ar\s+întâmpla\s+dacă'),
    },
    'uk': {
        0: (r'це\s+сьогоднішня\s+ситуація', r'я\s+говорю\s+про\s+теперішню\s+ситуацію'),
        1: (r'говорю\s+про\s+минулий\s+період',),
        2: (r'я\s+говорю\s+про\s+можливу\s+ситуацію', r'лише\s+уявляю\s*,?\s*що\s+це\s+могло\s+б\s+статися'),
    },
}
_ROLES = {'self': (0, 0), 'other': (1, 1), 'fiction': (2, 2)}
_SLICES = ((0, 3), (3, 6), (6, 10), (10, 13), (13, 15), (15, 17))


def _outside(text: str) -> str | None:
    value = unicodedata.normalize('NFC', text).casefold()
    for left, right in (('«', '»'), ('“', '”'), ('"', '"')):
        if left != right and value.count(left) != value.count(right):
            return None
        if left == right and value.count(left) % 2:
            return None
        value = re.sub(re.escape(left) + r'[^' + re.escape(right) + r']*' + re.escape(right), ' ', value)
    return value


def cues(text: str, language: str) -> dict:
    """Read only explicit operator cues; abstain on malformed or conflicting frames."""
    if not isinstance(text, str) or language not in _MODAL:
        raise ValueError('Expected Romanian or Ukrainian focal text')
    value = unicodedata.normalize('NFC', text).casefold()
    before_quote = re.split('[«“"]', value, maxsplit=1)[0]
    naming = (language == 'ro' and re.search(r'\bnu\s+știu\s+dacă\s+se\s+numește\b', before_quote)) or (
        language == 'uk' and re.search(r'\bне\s+знаю\s*,?\s*чи\s+це\b', before_quote))
    modal = None if naming else _MODAL[language].search(before_quote)
    outside = _outside(text)
    roles = [] if outside is None else [name for name, patterns in _ROLE[language].items()
                                  if any(re.search(r'\b' + pattern + r'\b', outside) for pattern in patterns)]
    times = [] if outside is None else [time for time, patterns in _TIME[language].items()
                                  if any(re.search(r'\b' + pattern + r'\b', outside) for pattern in patterns)]
    return {'uncertain': None if modal is None else modal.group(),
            'role': roles[0] if len(roles) == 1 else None,
            'time': times[0] if len(times) == 1 else None}


def correct_logits(logits, *, text: str, language: str, spec: dict, score_fn):
    """Return corrected logits and evidence, preserving the parent binary decision."""
    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    if values.shape != (19,) or not np.isfinite(values).all() or spec.get('language') != language:
        raise ValueError('Invalid review logits or language')
    before = values.copy()
    found = cues(text, language)
    candidate = before.copy()

    def force(start, end, target):
        chosen = start + target
        candidate[chosen] = max(float(candidate[start:end].max()) + 20.0, float(candidate[chosen]))
        if not np.isfinite(candidate[chosen]):
            raise ValueError('Correction produced nonfinite logits')

    if found['uncertain'] is not None:
        force(6, 10, 2)
    if found['role'] is not None:
        subject, source = _ROLES[found['role']]
        force(0, 3, subject)
        force(10, 13, source)
    if found['time'] is not None:
        force(3, 6, found['time'])
    threshold = spec['threshold']
    parent_score = float(score_fn(before, spec))
    candidate_score = float(score_fn(candidate, spec))
    parent_binary = int(parent_score >= threshold)
    candidate_binary = int(candidate_score >= threshold)
    rejected = parent_binary != candidate_binary
    result = before if rejected else candidate
    fields = [int(np.argmax(result[start:end])) for start, end in _SLICES]
    return result, {'cues': found, 'binary_guard_rejected': rejected, 'parent_score': parent_score,
                    'score': parent_score if rejected else candidate_score, 'fields': fields}
