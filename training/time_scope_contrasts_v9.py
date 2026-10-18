"""Training-only invariance to whether an experience is current or earlier.

Uses existing training families, never evaluation wording. Time alone does not
remove a concern under the review policy. Variants are correlated exposures.
"""
from training.focus_boundary_contrasts_v9 import CONTENT, FEMININE


FRAMES = {
    'ro': {
        'CURRENT': ['Vorbesc despre ceea ce trăiesc în perioada aceasta.',
                    'Mă refer la propria mea situație din zilele acestea.'],
        'PAST': ['Vorbesc despre ceva trăit cu ceva timp în urmă.',
                 'Mă refer la propria mea situație de acum câțiva ani.'],
    },
    'uk': {
        'CURRENT': ['Йдеться про те, що відбувається зі мною цими днями.',
                    'Розповідаю про власну ситуацію на цей момент.'],
        'PAST': ['Йдеться про те, що відбувалося зі мною раніше.',
                 'Розповідаю про власну ситуацію кілька років тому.'],
    },
}


def generate():
    rows = []
    for language, groups in CONTENT.items():
        for concern, messages in groups.items():
            for family, text in enumerate(messages):
                variants = [text] + ([FEMININE[language][text]] if text in FEMININE[language] else [])
                for message in variants:
                    for time, frames in FRAMES[language].items():
                        for marker in frames:
                            for turns in ([marker+' '+message], [message+' '+marker],
                                          [marker, message], [message, marker],
                                          [message, 'Poți continua.' if language == 'ro' else 'Можеш продовжити.', marker]):
                                values = [dict(role='assistant' if len(turns) == 3 and i == 1 else 'minor', text=t)
                                          for i, t in enumerate(turns)]
                                rows.append(dict(case_id=f'time-scope-v9-train-{language}-{len(rows)}',
                                    language=language, turns=values, focus=len(values)-1,
                                    attributes=dict(subject='SELF', source='DIRECT', assertion='AFFIRMED',
                                                    relation='NEW', time=time, concern=str(concern)),
                                    source='time_scope_contrasts_v9', family=f'focus-v9-{language}-{concern}-{family}',
                                    fictional=True))
    return rows
