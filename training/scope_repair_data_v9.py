"""Authored denial and scope clarifications for synthetic regression repair."""
import itertools
from training.joint_semantic_data_v9 import CONTENT, TIME


ASSERTIONS = {
    'ro': {
        'AFFIRMED': ['Susțin cele spuse.', 'Da, confirm ce am spus.', 'Consider adevărat ceea ce am relatat.',
                     'Nu neg cele relatate.', 'Rămân la cele spuse.', 'Este adevărat ce am descris.'],
        'NEGATED': ['Neg veridicitatea celor relatate.', 'De fapt, neg cele spuse.',
                    'Nu confirm ce am spus; lucrurile nu sunt așa.', 'Dezmint cele relatate.',
                    'Spusele acestea nu sunt adevărate.', 'Neg că descrierea ar fi adevărată.'],
        'UNCERTAIN': ['Nu pot confirma sau infirma cele relatate.', 'Încă nu știu dacă este adevărat.',
                      'Sunt nesigur în privința celor relatate.', 'Ar putea fi adevărat, dar nu știu.',
                      'Nu am certitudinea că așa stau lucrurile.', 'Nu știu ce să cred despre cele descrise.'],
        'RETRACTED': ['Retrag ceea ce am afirmat.', 'Îmi retrag spusele de mai devreme.',
                      'Nu mai susțin relatarea mea anterioară.', 'Renunț la ceea ce am afirmat anterior.',
                      'Vreau să îmi retrag declarația.', 'Retrag afirmația făcută înainte.'],
    },
    'uk': {
        'AFFIRMED': ['Підтверджую сказане.', 'Так, усе сказане є правдою.', 'Я підтверджую свій опис.',
                     'Не заперечую сказаного.', 'Дотримуюся своїх слів.', 'Описане мною є правдою.'],
        'NEGATED': ['Заперечую правдивість сказаного.', 'Насправді я заперечую сказане.',
                    'Не підтверджую сказаного: усе було не так.', 'Спростовую цей опис.',
                    'Ці слова не відповідають дійсності.', 'Заперечую, що цей опис правдивий.'],
        'UNCERTAIN': ['Не можу ні підтвердити, ні спростувати сказане.', 'Ще не знаю, чи це правда.',
                      'Не впевнений щодо описаного.', 'Це може бути правдою, але я не знаю.',
                      'Не маю певності, що все саме так.', 'Не знаю, що думати про описане.'],
        'RETRACTED': ['Відкликаю своє твердження.', 'Відкликаю свої попередні слова.',
                      'Більше не підтримую свого попереднього опису.', 'Відмовляюся від попереднього твердження.',
                      'Хочу відкликати свою заяву.', 'Відкликаю те, що стверджував раніше.'],
    },
}

SCOPES = {
    'ro': {
        'SELF': ['Descriu propria mea viață.', 'Este experiența mea reală.', 'Vorbesc aici despre mine.',
                 'Aceasta este situația mea, în realitate.', 'Experiența descrisă este a mea.', 'Relatarea mă privește pe mine.'],
        'OTHER': ['Redau experiența reală a unui prieten.', 'Este povestea reală a altcuiva.',
                  'Vorbesc despre un coleg real.', 'Relatarea se referă la o altă persoană, nu la mine.',
                  'Reproduc mesajul unui prieten despre viața lui.', 'Acestea sunt trăirile unei persoane cunoscute.'],
        'FICTIONAL': ['Acestea sunt cuvintele unui personaj dintr-un roman.',
                      'Este un pasaj dintr-o carte de ficțiune.', 'Am inventat scena pentru o povestire.',
                      'Mă refer la eroul unei nuvele, nu la o persoană reală.',
                      'Textul aparține unui personaj inventat de mine.', 'Este o replică dintr-o piesă de teatru fictivă.'],
    },
    'uk': {
        'SELF': ['Описую власне життя.', 'Це мій реальний досвід.', 'Тут я говорю про себе.',
                 'Це моя ситуація в реальному житті.', 'Це мій особистий досвід.', 'Ця розповідь стосується мене.'],
        'OTHER': ['Переказую реальний досвід друга.', 'Це справжня історія іншої людини.',
                  'Говорю про реального однокласника.', 'Опис стосується іншої людини, а не мене.',
                  'Наводжу повідомлення друга про його життя.', 'Це переживання знайомої мені людини.'],
        'FICTIONAL': ['Це слова персонажа з роману.', 'Це уривок із книжки про вигадані події.',
                      'Цю сцену придумано для оповідання.', 'Ідеться про героя повісті, а не про реальну людину.',
                      'Цей текст належить вигаданому мною персонажу.', 'Це репліка персонажа вигаданої п’єси.'],
    },
}


def generate(language):
    rows = []
    for index, assertion, variant, subject, when, layout in itertools.product(
            range(12), ASSERTIONS[language], range(6), SCOPES[language], TIME[language], range(2)):
        content = CONTENT[language][index] + '.'
        scope = SCOPES[language][subject][variant]
        temporal = TIME[language][when][variant % 3]
        claim = ASSERTIONS[language][assertion][variant]
        # In the clarification layout the final turn resolves a preceding
        # statement; the concern must be recovered from that earlier turn.
        prior = content if layout else ' '.join((scope, temporal, content))
        focal = ' '.join((scope, temporal, claim)) if layout else ' '.join((content, claim, scope, temporal))
        attributes = dict(subject=subject, time=when, assertion=assertion,
            source={'SELF': 'DIRECT', 'OTHER': 'REPORTED', 'FICTIONAL': 'FICTION'}[subject],
            relation='RETRACTS' if assertion == 'RETRACTED' else 'NEW', concern='1' if index < 8 else '0')
        rows.append(dict(case_id='v9-scope66-' + language + '-' + str(len(rows)), language=language,
            source='scope_repair_paraphrases_v9', synthetic=True, focus=2, attributes=attributes,
            turns=[dict(role='minor', text=prior),
                   dict(role='assistant', text='Te ascult.' if language == 'ro' else 'Я слухаю.'),
                   dict(role='minor', text=focal)]))
    return rows
