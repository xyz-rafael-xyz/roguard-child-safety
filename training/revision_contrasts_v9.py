"""Training-only withdrawal contrasts using study117's training content only.

Both members withdraw a prior claim. Its content decides whether the unchanged
policy asks for review. Real, reported and fictional contexts are all represented.
No development/test content or framing is used by this generator.
"""
from training.paired_scope_data_v9 import CONTENT, BENIGN

WITHDRAWALS = {
    'ro': [
        'Retrag ce tocmai am spus.',
        'M-am răzgândit: nu mai susțin afirmația de mai devreme.',
        'Aș vrea să iau înapoi cuvintele acelea.',
        'Mai înainte am spus asta, dar acum îmi retrag afirmația.',
    ],
    'uk': [
        'Беру назад те, що щойно сказав.',
        'Я передумав: більше не стверджую того, що казав раніше.',
        'Хочу взяти ті слова назад.',
        'Раніше я так сказав, але тепер відмовляюся від своїх слів.',
    ],
}
CONTEXTS = {
    'ro': {
        'self': 'Vorbesc despre mine: {x}',
        'reported': 'Redau ce mi-a spus o persoană pe care o cunosc: «{x}»',
        'fiction': 'În povestea inventată de mine, personajul spune: «{x}»',
    },
    'uk': {
        'self': 'Розповідаю про себе: {x}',
        'reported': 'Передаю слова знайомої людини: «{x}»',
        'fiction': 'У моїй вигаданій історії герой каже: «{x}»',
    },
}
ADDITIONAL_BENIGN = {
    'ro': [
        'Cum fac un tabel cu trei coloane?',
        'Nu știu ce tastă pune semnul întrebării.',
        'Caut o rețetă simplă de biscuiți.',
        'Vreau să învăț să număr în italiană.',
        'Nu îmi amintesc numele celui mai mare ocean.',
        'Cum se citește ora pe un ceas cu ace?',
        'Mă ajuți să scriu o invitație la un concurs de desen?',
        'Vreau să aflu de ce au norii forme diferite.',
        'Cum organizez pozele în dosare?',
        'Nu pricep diferența dintre un verb și un adjectiv.',
        'Mă ajuți să găsesc un nume pentru robotul din joc?',
        'Încerc să țin ritmul când bat din palme.',
    ],
    'uk': [
        'Як зробити таблицю з трьома стовпцями?',
        'Не знаю, якою клавішею набрати знак питання.',
        'Шукаю простий рецепт печива.',
        'Хочу навчитися рахувати італійською.',
        'Не пам’ятаю назву найбільшого океану.',
        'Як визначати час за стрілками годинника?',
        'Допоможеш написати запрошення на конкурс малюнків?',
        'Хочу дізнатися, чому хмари мають різну форму.',
        'Як розкласти фотографії по папках?',
        'Не розумію різниці між дієсловом і прикметником.',
        'Допоможи придумати ім’я для робота у грі.',
        'Пробую плескати в долоні в ритм музики.',
    ],
}


def generate(language):
    if language not in WITHDRAWALS:
        raise ValueError('Unsupported language')
    rows = []
    for i, concern in enumerate(CONTENT[language]['train']):
        benign = (BENIGN[language]['train'] + ADDITIONAL_BENIGN[language])[i]
        for scope, frame in CONTEXTS[language].items():
            for j, withdrawal in enumerate(WITHDRAWALS[language]):
                for layout in (0, 1):
                    pair = f'revision123-{language}-{i}-{scope}-{j}-{layout}'
                    for gold, content in ((0, benign), (1, concern)):
                        opening = frame.format(x=content)
                        turns = ([dict(role='minor', text=opening), dict(role='minor', text=withdrawal)]
                                 if layout else [dict(role='minor', text=opening + ' ' + withdrawal)])
                        attributes = dict(subject={'self': 'SELF', 'reported': 'OTHER', 'fiction': 'FICTIONAL'}[scope],
                                          time='CURRENT', assertion='RETRACTED',
                                          source={'self': 'DIRECT', 'reported': 'REPORTED', 'fiction': 'FICTION'}[scope],
                                          concern=str(gold), relation='RETRACTS')
                        rows.append(dict(case_id=pair + '-' + str(gold), language=language,
                                         source='revision_contrasts_v9', partition='train',
                                         synthetic_only=True, independently_annotated=False,
                                         pair_group=pair, scope=scope, content_family=str(i),
                                         turns=turns, focus=len(turns)-1, attributes=attributes, review=gold))
    return rows
