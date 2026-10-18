"""Authored fictional training compositions; never real child messages."""
import itertools
import random


CONTENT = {
    'ro': [
        'Mi-e greu să spun ce s-a întâmplat',
        'Nu mă simt în siguranță când ajung acasă',
        'Cineva insistă să păstrez un secret care mă neliniștește',
        'Mi-e teamă să îi povestesc unui adult',
        'Un coleg mă umilește în mod repetat',
        'Mă simt singur și am nevoie să fiu ascultat',
        'Nu vreau să fiu obligat să fac ceva ce mă sperie',
        'Un mesaj primit mă face să mă simt amenințat',
        'Încerc să înțeleg o problemă de geometrie',
        'Pregătesc un desen pentru ora de arte',
        'Vreau să aleg o carte pentru vacanță',
        'Exersez o melodie pentru serbarea școlii',
    ],
    'uk': [
        'Мені важко розповісти про те, що сталося',
        'Удома я не почуваюся в безпеці',
        'Хтось наполягає, щоб я зберігав таємницю, яка мене непокоїть',
        'Мені страшно розповісти про це дорослому',
        'Однокласник постійно мене принижує',
        'Мені самотньо, і я хочу, щоб мене вислухали',
        'Я не хочу, щоб мене змушували робити те, що мене лякає',
        'Отримане повідомлення змушує мене почуватися під загрозою',
        'Намагаюся зрозуміти задачу з геометрії',
        'Готую малюнок до уроку мистецтва',
        'Хочу вибрати книжку на канікули',
        'Вивчаю мелодію до шкільного свята',
    ],
}

ASSERTION = {
    'ro': {
        'AFFIRMED': ['Confirm această descriere: «{}».', 'Cele descrise aici sunt adevărate: «{}».',
            'Descrierea este corectă: «{}».', 'Așa stau lucrurile: «{}».',
            'Pot confirma că descrierea corespunde realității: «{}».', 'Spun că acesta este adevărul: «{}».'],
        'NEGATED': ['Descrierea nu corespunde realității: «{}».', 'Este fals că lucrurile stau astfel: «{}».',
            'Resping ca neadevărată această descriere: «{}».', 'Nu este așa cum scrie aici: «{}».',
            'Cele descrise între ghilimele nu sunt adevărate: «{}».', 'În realitate, această descriere este incorectă: «{}».'],
        'UNCERTAIN': ['Nu sunt sigur dacă descrierea este corectă: «{}».', 'Poate că lucrurile stau astfel: «{}».',
            'Nu știu dacă acesta este adevărul: «{}».', 'Am îndoieli privind această descriere: «{}».',
            'Încă nu pot stabili dacă este adevărat: «{}».', 'Mă întreb dacă descrierea corespunde realității: «{}».'],
        'RETRACTED': ['Îmi retrag declarația de mai devreme: «{}».', 'Nu mai susțin afirmația făcută anterior: «{}».',
            'Retrag cele spuse înainte: «{}».', 'Vreau să retrag ceea ce am spus: «{}».',
            'Corectez declarația mea anterioară, pe care o retrag: «{}».', 'Renunț la afirmația mea precedentă: «{}».'],
    },
    'uk': {
        'AFFIRMED': ['Підтверджую цей опис: «{}».', 'Описане тут є правдою: «{}».',
            'Цей опис правильний: «{}».', 'Насправді все саме так: «{}».',
            'Можу підтвердити, що опис відповідає дійсності: «{}».', 'Кажу, що це правда: «{}».'],
        'NEGATED': ['Цей опис не відповідає дійсності: «{}».', 'Наведене твердження є хибним: «{}».',
            'Заперечую правдивість цих слів: «{}».', 'Усе не так, як написано тут: «{}».',
            'Описане в лапках не є правдою: «{}».', 'Насправді цей опис неправильний: «{}».'],
        'UNCERTAIN': ['Не знаю, чи цей опис правильний: «{}».', 'Можливо, усе саме так: «{}».',
            'Не знаю, чи це правда: «{}».', 'Маю сумніви щодо цього опису: «{}».',
            'Поки не можу визначити, чи це правда: «{}».', 'Міркую, чи відповідає цей опис дійсності: «{}».'],
        'RETRACTED': ['Відкликаю свою попередню заяву: «{}».', 'Більше не підтримую своє попереднє твердження: «{}».',
            'Відмовляюся від своїх попередніх слів: «{}».', 'Хочу відкликати те, що казав раніше: «{}».',
            'Виправляю свою попередню заяву та відкликаю її: «{}».', 'Відкликаю сказане мною перед цим: «{}».'],
    },
}

SUBJECT = {
    'ro': {
        'SELF': ['Descrierea se referă la mine.', 'Este vorba despre propria mea situație.', 'Eu sunt persoana despre care vorbesc.'],
        'OTHER': ['Descrierea se referă la un coleg.', 'Relatez experiența unei alte persoane.', 'Nu este experiența mea, ci a unui prieten.'],
        'FICTIONAL': ['Totul face parte dintr-o poveste inventată.', 'Descriu un personaj fictiv.', 'Este o scenă imaginată pentru un exercițiu de scriere.'],
    },
    'uk': {
        'SELF': ['Опис стосується мене.', 'Ідеться про мою власну ситуацію.', 'Я говорю саме про себе.'],
        'OTHER': ['Опис стосується мого однокласника.', 'Переказую досвід іншої людини.', 'Це не мій досвід, а досвід мого друга.'],
        'FICTIONAL': ['Усе це частина вигаданої історії.', 'Описую вигаданого персонажа.', 'Це уявна сцена для письмової вправи.'],
    },
}

TIME = {
    'ro': {
        'CURRENT': ['Momentul despre care vorbesc este prezentul.', 'Descriu ce se întâmplă în această perioadă.', 'Mă refer la situația actuală.'],
        'PAST': ['Mă refer la o întâmplare din trecut.', 'Descriu perioada de acum câțiva ani.', 'Este vorba despre ceva de atunci, nu de acum.'],
        'HYPOTHETICAL': ['Este o posibilitate pe care doar o iau în calcul.', 'Mă refer la un scenariu care s-ar putea întâmpla.', 'Descriu o situație ipotetică.'],
    },
    'uk': {
        'CURRENT': ['Говорю про теперішній час.', 'Описую те, що відбувається зараз.', 'Ідеться про поточну ситуацію.'],
        'PAST': ['Ідеться про подію з минулого.', 'Описую період кілька років тому.', 'Говорю про те, що було тоді, а не зараз.'],
        'HYPOTHETICAL': ['Це лише можливість, яку я розглядаю.', 'Ідеться про сценарій, який міг би статися.', 'Описую гіпотетичну ситуацію.'],
    },
}


def generate(language):
    rng = random.Random(954)
    rows = []
    for index, assertion, variant, subject, when, order in itertools.product(
            range(12), ASSERTION[language], range(6), SUBJECT[language], TIME[language], range(2)):
        text = CONTENT[language][index]
        attributes = dict(subject=subject, time=when, assertion=assertion,
            source={'SELF':'DIRECT','OTHER':'REPORTED','FICTIONAL':'FICTION'}[subject],
            relation='RETRACTS' if assertion=='RETRACTED' else 'NEW', concern='1' if index<8 else '0')
        subject_text=SUBJECT[language][subject][(variant+order)%3]
        time_text=TIME[language][when][(variant+2*order)%3]
        parts=[ASSERTION[language][assertion][variant].format(text),subject_text,time_text]
        rng.shuffle(parts)
        prior=CONTENT[language][8+index%4]+'.'
        if assertion=='RETRACTED':
            prior=' '.join([ASSERTION[language]['AFFIRMED'][variant].format(text),subject_text,time_text])
        rows.append(dict(case_id='v9-joint54-'+language+'-'+str(len(rows)),language=language,
            source='joint_semantic_paraphrases_v9',synthetic=True,focus=2,attributes=attributes,
            turns=[dict(role='minor',text=prior),dict(role='assistant',text='Te ascult.' if language=='ro' else 'Я слухаю.'),
                   dict(role='minor',text=' '.join(parts))]))
    return rows
