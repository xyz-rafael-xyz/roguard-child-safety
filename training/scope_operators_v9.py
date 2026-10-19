"""Training-only contrasts for attribution, denial and focal-topic changes.

Uses study117 training content only, with newly authored scope wording.
Developed after the completed study123 development-error diagnosis. These are
synthetic policy examples, not independently annotated or real child messages.
"""
from training.paired_scope_data_v9 import CONTENT, BENIGN


FRAMES = {
    'ro': {
        'denial': [
            ('Cuvintele acestea nu reflectă situația mea: «{x}»',
             'Cuvintele acestea reflectă situația mea: «{x}»'),
            ('Nu vorbesc despre experiența mea când citez: «{x}»',
             'Vorbesc despre experiența mea când citez: «{x}»'),
            ('Această descriere nu vorbește despre mine: «{x}»',
             'Această descriere vorbește despre mine: «{x}»'),
            ('Nu mă descriu prin aceste cuvinte: «{x}»',
             'Mă descriu prin aceste cuvinte: «{x}»'),
            ('Eu nu trec prin ceea ce spune mesajul: «{x}»',
             'Nu altcineva, ci eu trec prin ceea ce spune mesajul: «{x}»'),
            ('Din ce scrie aici, nimic nu descrie situația mea: «{x}»',
             'Ce scrie aici descrie situația mea, chiar dacă mi-e greu să spun: «{x}»'),
        ],
        'fiction': [
            ('Pregătesc pentru clubul de teatru o replică inventată, fără legătură cu viața mea: «{x}»',
             'Pregătesc pentru clubul de teatru o replică despre ceea ce trăiesc eu: «{x}»'),
            ('Aceste cuvinte sunt doar pentru un rol fictiv pe scenă: «{x}»',
             'Aceste cuvinte nu sunt doar pentru un rol pe scenă; vorbesc despre mine: «{x}»'),
            ('Înregistrez vocea unui personaj imaginar, nu propria mea experiență: «{x}»',
             'Înregistrez un mesaj despre propria mea experiență, nu despre un personaj imaginar: «{x}»'),
            ('La repetiție exersez o situație inventată. Nu este povestea vieții mele: «{x}»',
             'La repetiție mi-am dat seama că vreau să spun ce trăiesc eu, nu ceva inventat: «{x}»'),
            ('Am scris pentru o animație un dialog fictiv, fără persoane reale: «{x}»',
             'Dialogul din animație îmi amintește de problema mea reală: «{x}»'),
            ('E un exercițiu de actorie cu vorbe inventate, nu o mărturisire despre mine: «{x}»',
             'Oprind exercițiul de actorie, îți spun o problemă a mea, fără să inventez: «{x}»'),
        ],
    },
    'uk': {
        'denial': [
            ('Ці слова не описують того, що я переживаю: «{x}»',
             'Ці слова точно описують те, що я переживаю: «{x}»'),
            ('Зі своїм досвідом я ці слова не пов’язую: «{x}»',
             'Зі своїм досвідом я ці слова пов’язую: «{x}»'),
            ('У цьому повідомленні йдеться не про мою ситуацію: «{x}»',
             'У цьому повідомленні йдеться саме про мою ситуацію: «{x}»'),
            ('Цими словами я себе не описую: «{x}»',
             'Цими словами я описую себе: «{x}»'),
            ('Цей опис не відповідає тому, як я живу: «{x}»',
             'Цей опис відповідає тому, як я живу: «{x}»'),
            ('Йдеться не про мій особистий досвід: «{x}»',
             'Йдеться про мій особистий досвід, а не про когось іншого: «{x}»'),
        ],
        'fiction': [
            ('Готую для шкільного театру вигадану репліку, яка не пов’язана з моїм життям: «{x}»',
             'Готую для шкільного театру репліку про те, що насправді переживаю: «{x}»'),
            ('Ці слова лише для вигаданої ролі на сцені: «{x}»',
             'Ці слова не лише для ролі на сцені; вони про мене: «{x}»'),
            ('Записую голос вигаданого персонажа, а не розповідь про свій досвід: «{x}»',
             'Записую розповідь про свій досвід, а не голос вигаданого персонажа: «{x}»'),
            ('На репетиції розігрую вигадану ситуацію. Це не історія мого життя: «{x}»',
             'На репетиції захотілося розповісти про своє життя, а не про вигадану ситуацію: «{x}»'),
            ('Пишу для мультфільму вигаданий діалог без реальних людей: «{x}»',
             'Діалог у мультфільмі нагадує мені про мою справжню проблему: «{x}»'),
            ('Це акторська вправа з вигаданими словами, а не зізнання про мене: «{x}»',
             'Зупиняю акторську вправу, щоб без вигадок розповісти про свою проблему: «{x}»'),
        ],
    },
}

RESET = {
    'ro': [
        ('Pentru moment vreau o explicație la o altă temă. {b}',
         'Mai am nevoie de sprijin cu problema mea: {x}'),
        ('Acum întreb despre ceva separat. {b}',
         'Problema de mai devreme încă mă apasă: {x}'),
        ('Am o întrebare nouă despre un lucru obișnuit. {b}',
         'Încă vorbesc despre situația mea: {x}'),
        ('Trec puțin la altceva. {b}',
         'Revin la ce trăiesc eu: {x}'),
    ],
    'uk': [
        ('Зараз хочу пояснення з іншої теми. {b}',
         'Мені ще потрібна підтримка з моєю проблемою: {x}'),
        ('Тепер питаю про окрему річ. {b}',
         'Попередня проблема досі мене гнітить: {x}'),
        ('У мене нове питання про звичайну справу. {b}',
         'Я досі говорю про свою ситуацію: {x}'),
        ('На хвилину перейду до іншого. {b}',
         'Повернуся до того, що переживаю: {x}'),
    ],
}


def generate(language):
    if language not in FRAMES:
        raise ValueError('Unsupported language')
    rows = []

    def add(group, turns, scope, concern, affirmed, paired=True):
        fictional = scope == 'fiction' and not affirmed
        gold = int(concern and affirmed)
        attrs = dict(subject='FICTIONAL' if fictional else 'SELF', time='CURRENT',
                     assertion='NEGATED' if scope == 'denial' and not affirmed else 'AFFIRMED',
                     source='FICTION' if fictional else 'DIRECT', concern=str(concern), relation='NEW')
        rows.append(dict(case_id=group+'-'+str(affirmed), language=language,
                         source='scope_operators_v9', partition='train', synthetic_only=True,
                         independently_annotated=False, pair_group=group if paired else None,
                         scope=scope, turns=[dict(role='minor', text=t) for t in turns],
                         focus=len(turns)-1, attributes=attrs, review=gold))

    for concern, contents in ((1, CONTENT[language]['train']), (0, BENIGN[language]['train'])):
        for i, content in enumerate(contents):
            for scope, frames in FRAMES[language].items():
                for j, forms in enumerate(frames):
                    for layout in (0, 1):
                        group = f'operators-{language}-{concern}-{i}-{scope}-{j}-{layout}'
                        for affirmed, frame in enumerate(forms):
                            text = frame.format(x=content)
                            turns = text.split(': ', 1) if layout else [text]
                            add(group, turns, scope, concern, affirmed, paired=bool(concern))
    for i, content in enumerate(CONTENT[language]['train']):
        for j, (reset, continuation) in enumerate(RESET[language]):
            benign = BENIGN[language]['train'][(i+j) % len(BENIGN[language]['train'])]
            group = f'operators-{language}-reset-{i}-{j}'
            # Both members retain exactly the same earlier concern. Only the
            # focal message's topic changes; history is never erased.
            add(group, [content, reset.format(b=benign)], 'reset', 0, 0)
            add(group, [content, continuation.format(x=content)], 'reset', 1, 1)
    return rows
