"""Ukrainian scope repair from inspected regression error classes, not fresh validation."""
from training.review_contrasts_v8 import TEXT
from training.disclosure_hesitancy_v8 import POSITIVE as HESITANCY

POSITIVE=[
'Хочеться знайти співрозмовника для розмови про мої переживання.',
'Прагну зустріти когось, хто поставиться до моїх почуттів із розумінням.',
'Мені потрібен хтось поруч, кому можна довірити свої тривоги.',
'Хотілося б поділитися тим, що на душі, з уважною людиною.',
'Потребую розмови про те, чому мені так сумно.',
'Хочу, щоб хтось почув, наскільки мені самотньо.',
'Було б легше, якби я міг поділитися своїм болем.',
'Мрію знайти людину, від якої не треба приховувати свої переживання.',
'Шукаю підтримки, бо важко впоратися з переживаннями.',
'Не вистачає людини, яка вислухала б мої страхи.',
'Хотілося поговорити про мої труднощі з кимось небайдужим.',
'Потрібна порада, як розповісти близьким про мої тривоги.'
]+TEXT['uk']['positive']+HESITANCY['uk']
NEGATIVE=[
'Не розумію, чому магніт притягує залізо.',
'Хочу з’ясувати, як утворюється туман.',
'Не можу пояснити, чому лід тане від тепла.',
'Цікаво, звідки береться луна в горах.',
'Не знаю, як поділити дріб на ціле число.',
'Хочу розібратися, чому Місяць змінює форму.',
'Мені незрозуміло, як працює компас.',
'Не збагну, чому влітку день довший.',
'Шукаю пояснення до задачі про швидкість.',
'Хочу, щоб хтось пояснив правило множення.',
'Мені потрібна порада, як налаштувати будильник.',
'З’ясовую, як зробити рамку для малюнка.',
'Потрібна допомога з нотами для пісні.',
'Не знаю, яку фарбу змішати із синьою.',
'Радий, що нарешті склав пазл.',
'Тішуся, бо навчився кататися на велосипеді.',
'У мене вийшло розв’язати приклад без підказки.',
'Пишаюся тим, що закінчив свій малюнок.',
'Хочу розповісти про свою колекцію камінців.',
'Цікаво послухати пояснення до досліду з водою.',
'Шукаю людину, яка навчить мене грати в шашки.',
'Хотілося б поговорити про правила настільної гри.',
'Прагну дізнатися більше про рослини на підвіконні.',
'Мені потрібен співрозмовник для вправи з англійської.'
]+TEXT['uk']['negative']
SCOPES={
'SELF':['Маю на увазі себе і власне самопочуття.','Це опис мого особистого досвіду.'],
'OTHER':['Це слова знайомої людини, які я переказую.','Так свій стан описує мій приятель.'],
'FICTIONAL':['Це цитата з вигаданого оповідання; я такого не переживаю.','Тут говорить персонаж книжки, а не я.'],
'PAST':['Так було зі мною у давніший період.','Описую свій колишній стан, а не теперішній.'],
'NEGATED':['Заперечую, що наступний опис правдивий: ','Стверджую, що наведені далі слова не відповідають реальності: ']}


def generate():
    rows=[]
    for signal,phrases in ((1,POSITIVE),(0,NEGATIVE)):
        for family,content in enumerate(phrases):
            def add(texts,**changes):
                attrs={'subject':'SELF','source':'DIRECT','time':'CURRENT','assertion':'AFFIRMED','relation':'NEW','concern':str(signal)};attrs.update(changes)
                for surface in ('original','lower'):
                    rows.append({'case_id':f'uk-scope-train-{len(rows)}','language':'uk','turns':[{'role':'minor','text':t.lower() if surface=='lower' else t} for t in texts],'focus':len(texts)-1,'attributes':attrs,'family':f'uk-scope-{signal}-{family}','source':'authored_uk_scope_contrasts_v8','surface':surface,'fictional':True})
            add([content])
            for scope,markers in SCOPES.items():
                attrs={'subject':'OTHER','source':'REPORTED'} if scope=='OTHER' else {'subject':'FICTIONAL','source':'FICTION'} if scope=='FICTIONAL' else {'time':'PAST'} if scope=='PAST' else {'assertion':'NEGATED'} if scope=='NEGATED' else {}
                for marker in markers:
                    add([marker+' '+content],**attrs);add([marker,content],**attrs)
                    if scope!='NEGATED':add([content,marker],**attrs)
                    else:
                        for reported in SCOPES['OTHER']:
                            add([reported,marker+' '+content],subject='OTHER',source='REPORTED',assertion='NEGATED')
                            add([reported+' '+marker+' '+content],subject='OTHER',source='REPORTED',assertion='NEGATED')
    return rows
