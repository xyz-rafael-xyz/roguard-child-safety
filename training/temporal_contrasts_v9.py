"""Training-only time invariance on the new synthetic content families."""
import unicodedata
from training.natural_contrasts_v9 import CONTENT


FRAMES = {
    'ro': [
        ('CURRENT', 'Redau ce gândesc chiar acum: «{text}»'),
        ('PAST', 'Îmi amintesc ce gândeam pe atunci: «{text}»'),
        ('PAST', 'Povestesc despre o etapă mai veche din viața mea: «{text}»'),
        ('HYPOTHETICAL', 'Îmi imaginez o situație în care aș spune: «{text}»'),
    ],
    'uk': [
        ('CURRENT', 'Передаю те, що думаю саме зараз: «{text}»'),
        ('PAST', 'Пригадую, про що думалося тоді: «{text}»'),
        ('PAST', 'Розповідаю про давніший період свого життя: «{text}»'),
        ('HYPOTHETICAL', 'Уявляю ситуацію, у якій можна було б сказати: «{text}»'),
    ],
}
PAST = {
    'ro':'Vreau să-ți povestesc despre mine într-o perioadă de acum câțiva ani.',
    'uk':'Хочу розповісти про себе в один із періодів кілька років тому.',
}


def generate():
    rows=[]
    for lang,groups in CONTENT.items():
        for signal,messages in groups.items():
            for family,text in enumerate(messages):
                cases=[(time,[{'role':'minor','text':frame.format(text=text)}]) for time,frame in FRAMES[lang]]
                cases.append(('PAST',[{'role':'minor','text':PAST[lang]},{'role':'minor','text':text}]))
                for time,turns in cases:
                    variants=[turns]
                    if lang=='ro':
                        variants.append([dict(t,text=''.join(c for c in unicodedata.normalize('NFD',t['text']) if not unicodedata.combining(c))) for t in turns])
                    for variant in variants:
                        rows.append(dict(case_id=f'temporal-v9-train-{lang}-{len(rows)}',language=lang,
                            turns=variant,focus=len(variant)-1,attributes=dict(subject='SELF',source='DIRECT',
                            assertion='AFFIRMED',relation='NEW',time=time,concern=str(signal)),
                            source='temporal_contrasts_v9',family=f'natural-v9-{lang}-{signal}-{family}',fictional=True))
    return rows
