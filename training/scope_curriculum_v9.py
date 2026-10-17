"""Training-only contrasts crossing assertion scope with unrelated prior topics.

Project-authored synthetic wording, not independently reviewed child language.
Content comes exclusively from declared training slices. No evaluation wording
or predicted evaluation labels are used to construct these examples.
"""
import random
from training.context_v7 import CONCERNS
from training.context_v71 import TEXT
from training.disclosure_hesitancy_v8 import POSITIVE

FRAMES = {
    'ro': {
        'self': ['Acum mă refer la situația mea reală.', 'Următorul lucru este despre mine personal.'],
        'other': ['Acum reproduc mărturia unei persoane reale.', 'Următoarele cuvinte sunt ale unui prieten, nu ale unui personaj.'],
        'fiction': ['Tot ce urmează este inventat pentru o povestire.', 'Următoarele cuvinte aparțin exclusiv unui personaj imaginar.'],
        'affirm': ['Susțin că descrierea aceasta este adevărată:', 'Confirm următoarea descriere:'],
        'deny': ['Contest adevărul următoarei descrieri:', 'Dezavuez afirmația care urmează:', 'Refuz să confirm acest lucru, fiindcă este fals:', 'Nu corespunde realității următoarea afirmație:'],
        'uncertain': ['Nu pot confirma încă următoarea descriere:', 'Este posibil să fie adevărat ceea ce urmează:'],
        'retract': ['Îmi retrag confirmarea de adineauri:', 'Renunț la ceea ce am susținut mai înainte:'],
    },
    'uk': {
        'self': ['Тепер ідеться про мою реальну ситуацію.', 'Наступне стосується мене особисто.'],
        'other': ['Тепер передаю свідчення реальної людини.', 'Наступні слова належать другові, а не персонажу.'],
        'fiction': ['Усе наступне вигадане для оповідання.', 'Наступні слова належать лише уявному персонажу.'],
        'affirm': ['Стверджую, що цей опис правдивий:', 'Підтверджую такий опис:'],
        'deny': ['Оспорюю правдивість наступного опису:', 'Спростовую наступну заяву:', 'Відмовляюся підтверджувати це, бо це неправда:', 'Наступне твердження не відповідає реальності:'],
        'uncertain': ['Ще не можу підтвердити наступний опис:', 'Можливо, наступне є правдою:'],
        'retract': ['Відкликаю своє щойно надане підтвердження:', 'Відмовляюся від того, що стверджував перед цим:'],
    },
}


def generate():
    result = []
    rng = random.Random(9017)
    for language in ('ro', 'uk'):
        words = FRAMES[language]
        contents = [(1, text) for text in CONCERNS[language][:12] + POSITIVE[language]]
        contents += [(0, text) for text in TEXT[language][:24]]
        for family, (signal, content) in enumerate(contents):
            for subject, scope in [('SELF', 'self'), ('OTHER', 'other'), ('FICTIONAL', 'fiction')]:
                for assertion, frame in [('AFFIRMED', 'affirm'), ('NEGATED', 'deny'), ('UNCERTAIN', 'uncertain'), ('RETRACTED', 'retract')]:
                    for variant in range(2):
                        scope_text = words[scope][variant]
                        assertion_text = rng.choice(words[frame])
                        utterance = assertion_text + ' «' + content + '»'
                        current = scope_text + ' ' + utterance if variant == 0 else utterance + ' ' + scope_text
                        prior = rng.choice(TEXT[language][:24])
                        if assertion == 'RETRACTED':
                            prior = scope_text + ' ' + words['affirm'][variant] + ' «' + content + '»'
                        result.append({
                            'case_id': f'scope-v9-train-{language}-{len(result)}',
                            'language': language, 'focus': 2,
                            'turns': [{'role': 'minor', 'text': prior},
                                      {'role': 'assistant', 'text': 'Te ascult.' if language == 'ro' else 'Я слухаю.'},
                                      {'role': 'minor', 'text': current}],
                            'attributes': {'subject': subject, 'source': {'SELF': 'DIRECT', 'OTHER': 'REPORTED', 'FICTIONAL': 'FICTION'}[subject],
                                           'time': 'CURRENT', 'assertion': assertion,
                                           'relation': 'RETRACTS' if assertion == 'RETRACTED' else 'NEW', 'concern': str(signal)},
                            'source': 'scope_curriculum_v9', 'family': f'{language}-{family}',
                            'pair_group': f'scope-v9-{language}-{family}-{subject}-{variant}',
                            'fictional': True,
                        })
    return result
