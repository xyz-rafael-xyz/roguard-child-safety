"""Decouple conversational frame from whether any concern is mentioned."""
from copy import deepcopy
from training.context_v7 import generate as original, LABELS, WRAPPERS, HELD
BENIGN={
'ro':{
 'train':['Îmi place să citesc despre animale.','Am învățat un cântec nou.','Vreau să exersez scrierea literelor.','Mă bucur de vacanță.','Am desenat un nor.','Îmi place să ascult muzică.','Am terminat o carte interesantă.','Vreau să învăț mai multe despre oceane.','Știu să număr până la o sută.','Am făcut o felicitare.','Mi-a ieșit bine exercițiul de gramatică.','Mă interesează cum cresc plantele.'],
 'dev':['Îmi place să învăț cuvinte noi.','Am ascultat o melodie veselă.','Vreau să aflu cum funcționează un termometru.','Am colorat o imagine.'],
 'cal':['Mă bucur că am înțeles lecția.','Am citit despre păduri.','Mi-ar plăcea să învăț o poezie.','Am completat un rebus.']},
'uk':{
 'train':['Мені подобається читати про тварин.','Я вивчив нову пісню.','Хочу потренуватися писати літери.','Радію канікулам.','Я намалював хмару.','Мені подобається слухати музику.','Я дочитав цікаву книжку.','Хочу дізнатися більше про океани.','Умію рахувати до ста.','Я зробив листівку.','Добре виконав граматичну вправу.','Мене цікавить, як ростуть рослини.'],
 'dev':['Мені подобається вивчати нові слова.','Я послухав веселу мелодію.','Хочу дізнатися, як працює термометр.','Я розфарбував малюнок.'],
 'cal':['Радію, що зрозумів урок.','Я читав про ліси.','Хотілося б вивчити вірш.','Я розгадав кросворд.']}}
def generate(split):
    rows=original(split)
    for lang in ('ro','uk'):
        for n,text in enumerate(BENIGN[lang][split]):
            for k,frame in enumerate(LABELS):
                contexts=WRAPPERS[lang][k] if split=='train' else [HELD[lang][split][k]]
                if split=='train':
                    from training.context_variants_v7 import VARIANTS
                    contexts=contexts+VARIANTS[lang][k]
                for j,context in enumerate(contexts):
                    for f,texts in enumerate(([context,text],[text+' '+context],[text,context])):
                        rows.append(dict(case_id=f'{split}-benign-{lang}-{n}-{k}-{j}-{f}',language=lang,turns=[dict(role='minor',text=t) for t in texts],frame=frame,signal=0,family=f'{split}-benign-{lang}-{n}',pair_group=f'{split}-benign-{lang}-{n}-{j}-{f}',fictional=True,surface='original',source='authored_factorial'))
            if split=='train':rows.append(dict(case_id=f'train-alone-{lang}-{n}',language=lang,turns=[dict(role='minor',text=text)],frame='ORDINARY',signal=0,family=f'train-benign-{lang}-{n}',pair_group=f'train-alone-{lang}-{n}',fictional=True,surface='original',source='authored_factorial'))
    return rows
