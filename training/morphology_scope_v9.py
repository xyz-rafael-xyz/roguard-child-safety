"""Training-only scope contrasts with balanced feminine/masculine wording.

The study123 development diagnosis found no explicit feminine fictional-character
terms in its training pool. This is a coverage intervention, not a claim that
grammatical gender alone caused the observed errors. Content comes only from
study117 training; protected development/test framing is not copied.
"""
import re
from training.paired_scope_data_v9 import CONTENT, BENIGN

FRAMES = {
    'ro': {
        'feminine': [
            ('Acea eroină e creată de mine; îi scriu replica:', 'O colegă mi-a povestit despre situația ei:'),
            ('Protagonista există doar în scenariul meu, unde spune:', 'O prietenă mi-a descris viața ei în cuvintele acestea:'),
            ('Naratoarea e o figură inventată; textul ei este:', 'O fată pe care o cunosc mi-a spus despre ea:'),
            ('Am creat o eroină pentru un joc de rol; replica ei este:', 'O colegă de echipă mi-a cerut sprijin prin mesajul:'),
            ('Această voce feminină aparține unei povestiri fictive:', 'O vecină de vârsta mea mi-a mărturisit:'),
            ('Personajul feminin din ficțiunea mea rostește:', 'O fată din grupul meu de prieteni mi-a scris despre ce trăiește:'),
        ],
        'masculine': [
            ('Acel erou e creat de mine; îi scriu replica:', 'Un coleg mi-a povestit despre situația lui:'),
            ('Protagonistul există doar în scenariul meu, unde spune:', 'Un prieten mi-a descris viața lui în cuvintele acestea:'),
            ('Naratorul e o figură inventată; textul lui este:', 'Un băiat pe care îl cunosc mi-a spus despre el:'),
            ('Am creat un erou pentru un joc de rol; replica lui este:', 'Un coleg de echipă mi-a cerut sprijin prin mesajul:'),
            ('Această voce masculină aparține unei povestiri fictive:', 'Un vecin de vârsta mea mi-a mărturisit:'),
            ('Personajul masculin din ficțiunea mea rostește:', 'Un băiat din grupul meu de prieteni mi-a scris despre ce trăiește:'),
        ],
    },
    'uk': {
        'feminine': [
            ('Цю героїню я вигадав для гри; ось її репліка:', 'Однокласниця розповіла мені про свою ситуацію:'),
            ('Головна героїня існує лише в моєму сценарії та каже:', 'Подруга описала мені своє життя такими словами:'),
            ('Оповідачка вигадана, я написав для неї такі слова:', 'Дівчина, яку я знаю, розповіла про себе:'),
            ('Для рольової гри створюю героїню з такою реплікою:', 'Дівчина з нашої команди звернулася по підтримку з повідомленням:'),
            ('Жіночий голос належить персонажці вигаданої історії:', 'Моя знайома ровесниця довірила мені такі слова:'),
            ('Вигадана персонажка в моїй сцені промовляє:', 'Дівчина з нашої компанії написала мені про своє життя:'),
        ],
        'masculine': [
            ('Цього героя я вигадав для гри; ось його репліка:', 'Однокласник розповів мені про свою ситуацію:'),
            ('Головний герой існує лише в моєму сценарії та каже:', 'Друг описав мені своє життя такими словами:'),
            ('Оповідач вигаданий, я написав для нього такі слова:', 'Хлопець, якого я знаю, розповів про себе:'),
            ('Для рольової гри створюю героя з такою реплікою:', 'Хлопець з нашої команди звернувся по підтримку з повідомленням:'),
            ('Чоловічий голос належить персонажу вигаданої історії:', 'Мій знайомий ровесник довірив мені такі слова:'),
            ('Вигаданий персонаж у моїй сцені промовляє:', 'Хлопець з нашої компанії написав мені про своє життя:'),
        ],
    },
}

FEMININE = {
    'ro': {'invizibil':'invizibilă','singur':'singură','speriat':'speriată',
           'pedepsit':'pedepsită','apropiat':'apropiată','copleșit':'copleșită'},
    'uk': {'непомітним':'непомітною','залишився':'залишилася','Загубив':'Загубила'},
}


def realization(text, language, gender):
    if gender == 'masculine':
        return text
    substitutions = FEMININE[language]
    return re.sub(r'\b(?:'+ '|'.join(map(re.escape,substitutions)) +r')\b',
                  lambda match: substitutions[match.group()], text)


def generate(language):
    if language not in FRAMES:
        raise ValueError('Unsupported language')
    rows = []
    for concern, contents in ((1,CONTENT[language]['train']),(0,BENIGN[language]['train'])):
        for i, content in enumerate(contents):
            for gender, frames in FRAMES[language].items():
                quote = '«'+realization(content,language,gender)+'»'
                for j, forms in enumerate(frames):
                    for layout in (0,1):
                        group = f'morphology-{language}-{concern}-{i}-{gender}-{j}-{layout}'
                        for real, frame in enumerate(forms):
                            turns = ([dict(role='minor',text=frame),dict(role='minor',text=quote)] if layout
                                     else [dict(role='minor',text=frame+' '+quote)])
                            gold = concern*real
                            attrs = dict(subject='OTHER' if real else 'FICTIONAL',time='CURRENT',
                                         assertion='AFFIRMED',source='REPORTED' if real else 'FICTION',
                                         concern=str(concern),relation='NEW')
                            rows.append(dict(case_id=group+'-'+str(real),language=language,
                                             source='morphology_scope_v9',partition='train',synthetic_only=True,
                                             independently_annotated=False,grammatical_gender=gender,
                                             semantic_group=f'{language}-{concern}-{i}-{real}',
                                             pair_group=group if concern else None,
                                             turns=turns,focus=len(turns)-1,attributes=attrs,review=gold))
    return rows
