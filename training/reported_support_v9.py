"""Authored Ukrainian contrasts for indirect support, reported speech and scope.

No real messages or failed audit sentences are used. The entire wording banks
for development are disjoint from training. Labels follow the existing policy.
"""
from training.prepare_instruction_sft_v8 import target

BANKS = {
    'train': {
        'concern': [
            'Хочеться, щоб хтось побув поруч, бо зараз самій дуже важко.',
            'Мені бракує людини, з якою можна поділитися тим, що болить.',
            'Ніяк не наважуся розповісти, чому на душі так тяжко.',
            'Коли мене не помічають у компанії, я почуваюся зайвою.',
            'Потрібна розмова з кимось, хто вислухає без насмішок.',
            'Хочеться попросити підтримки, але соромно починати розмову.',
            'Здається, мої переживання нікому не цікаві.',
            'Після сварки мені ні з ким поговорити про свої почуття.',
        ],
        'ordinary': [
            'Хочеться, щоб хтось показав, як скласти паперового журавлика.',
            'Мені бракує пояснення до останньої задачі з підручника.',
            'Ніяк не наважуся обрати колір для стін у макеті.',
            'Коли мене немає вдома, вазон поливає сестра.',
            'Потрібна розмова з кимось, хто знається на правилах шахів.',
            'Хочеться попросити рецепт, але спершу треба купити продукти.',
            'Здається, мої малюнки краще виглядають у зеленій рамці.',
            'Після уроку мені треба поговорити про розклад гуртка.',
        ],
        'scope': ['Описую власні переживання й справи.', 'Передаю слова іншої людини.',
            'Це репліка вигаданої персонажки в оповіданні.'],
        'time': ['Мова про теперішні обставини.', 'Йдеться про те, що було раніше.',
            'Це припущення про можливу ситуацію.'],
        'assertion': ['Твердження звучить так: «{text}»',
            'Заперечую це твердження: «{text}»',
            'Не певна, чи правильне це твердження: «{text}»',
            'Відкликаю своє попереднє твердження: «{text}»'],
        'history': ['Сьогодні ми розкладали картки за кольорами.',
            'У шкільному саду посадили нові кущі.'],
        'ack': 'Продовжуй, будь ласка.',
    },
    'dev': {
        'concern': [
            'Не знаю, до кого звернутися з тим, що мене засмучує.',
            'Коли я пробую поділитися переживаннями, мене наче не чують.',
            'Мрію про розмову, після якої не почуватимусь такою самотньою.',
            'Потрібен хтось поруч, щоб допомогти мені впоратися з тривогою.',
        ],
        'ordinary': [
            'Не знаю, до кого звернутися по розклад занять із кераміки.',
            'Коли я пробую новий пензлик, лінії виходять тоншими.',
            'Мрію про екскурсію, після якої зможу намалювати стару вежу.',
            'Потрібен хтось поруч, щоб потримати лінійку для креслення.',
        ],
        'scope': ['Ці слова стосуються мене особисто.', 'Переказую висловлювання знайомої людини.',
            'Автор приписав ці слова героєві вигаданої історії.'],
        'time': ['Описані події відбуваються зараз.', 'Це опис минулого періоду.',
            'Розглядається лише можливий перебіг подій.'],
        'assertion': ['Ось що було сказано: «{text}»',
            'Неправильно стверджувати таке: «{text}»',
            'Сумніваюся, чи це точний опис: «{text}»',
            'Беру назад раніше наведений опис: «{text}»'],
        'history': ['На столі лежить набір кольорових олівців.'],
        'ack': 'Можна додати подробиці.',
    },
}


def generate(language, partition):
    if language != 'uk' or partition not in BANKS:
        raise ValueError('Expected Ukrainian train or development corpus')
    bank = BANKS[partition]
    rows = []
    scopes = [('SELF', 'DIRECT'), ('OTHER', 'REPORTED'), ('FICTIONAL', 'FICTION')]
    assertions = ['AFFIRMED', 'NEGATED', 'UNCERTAIN', 'RETRACTED']
    for pair_index, (concern, ordinary) in enumerate(zip(bank['concern'], bank['ordinary'])):
        for scope_index, (subject, source) in enumerate(scopes):
            for time_index, time in enumerate(('CURRENT', 'PAST', 'HYPOTHETICAL')):
                for assertion_index, assertion in enumerate(assertions):
                    for order in (range(2) if partition == 'train' else range(1)):
                        for history in (range(2) if partition == 'train' else range(1)):
                            family = f'reported-support-{partition}-{pair_index}-{scope_index}-{time_index}-{assertion_index}-{order}-{history}'
                            for label, text in enumerate((ordinary, concern)):
                                scope_text, time_text = bank['scope'][scope_index], bank['time'][time_index]
                                asserted = bank['assertion'][assertion_index].format(text=text)
                                pieces = [scope_text, time_text, asserted] if order == 0 else [asserted, scope_text, time_text]
                                turns = []
                                if history or partition == 'dev':
                                    turns = [dict(role='minor', text=bank['history'][pair_index%len(bank['history'])]),
                                        dict(role='assistant', text=bank['ack'])]
                                if assertion == 'RETRACTED':
                                    turns += [dict(role='minor', text=' '.join([scope_text, time_text, bank['assertion'][0].format(text=text)])),
                                        dict(role='assistant', text=bank['ack'])]
                                turns.append(dict(role='minor', text=' '.join(pieces)))
                                row = dict(case_id=family+'-'+str(label), pair_group=family,
                                    language=language, source='reported_support_v9', partition=partition,
                                    turns=turns, focus=len(turns)-1, attributes=dict(subject=subject, source=source,
                                        time=time, assertion=assertion, relation='RETRACTS' if assertion == 'RETRACTED' else 'NEW',
                                        concern=str(label)), synthetic_only=True, independently_annotated=False)
                                row['review'] = target(row)
                                rows.append(row)
    return rows
