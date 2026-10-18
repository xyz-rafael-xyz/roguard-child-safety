"""New fictional training families for task difficulty and context-frame switches.

Machine-authored examples, not child records or independently reviewed language.
No evaluation file is read. Paired variants are correlated training exposures.
"""
CONTENT = {
    'ro': {
        0: [
            'Încă nu am găsit piesa de puzzle cu marginea dreaptă.',
            'Nu pot deschide fișierul cu tema la geografie.',
            'Mi-e greu să aleg între două modele de avion din hârtie.',
            'N-am înțeles de ce busola arată spre nord.',
            'Mă întreb dacă am numărat corect pătratele de pe foaie.',
            'Nu sunt convins că am pus virgula unde trebuie.',
            'M-am încurcat la ultima întrebare din jocul de cultură generală.',
            'Nu știu ce material să folosesc pentru macheta podului.',
            'Încă încerc să-mi amintesc cum se numește acel dinozaur.',
            'Alegerea unui font pentru afiș îmi ia mai mult timp.',
            'Nu reușesc să aliniez coloanele tabelului.',
            'Am nevoie de ajutor ca să repar fermoarul penarului.',
        ],
        1: [
            'Deși toți cred că sunt vesel, mă simt singur aproape mereu.',
            'Când mă gândesc la ziua de mâine, mă apucă o teamă pe care n-o pot opri.',
            'Nu mai reușesc să mă liniștesc după discuțiile de acasă.',
            'Mi-aș dori să pot spune cuiva cât de mult mă apasă totul.',
            'Mi-e frică să rămân singur cu persoana care mă amenință.',
            'Îmi ascund lacrimile ca să nu râdă ceilalți de mine.',
            'Vreau să-i spun cuiva ce pățesc, dar de fiecare dată mă răzgândesc.',
            'Am nevoie să mă creadă cineva când spun că mi-e teamă.',
            'Mă simt respins de ceilalți și asta mă doare.',
            'Aș vrea să pot cere sprijin fără să simt că deranjez.',
            'Nu mă pot opri din griji nici când ar trebui să mă odihnesc.',
            'Sper să găsesc un adult care să mă ajute cu frica asta.',
        ],
    },
    'uk': {
        0: [
            'Досі не можу знайти деталь пазла з рівним краєм.',
            'Не відкривається файл із домашнім завданням з географії.',
            'Важко обрати між двома моделями паперового літака.',
            'Не розумію, чому стрілка компаса показує на північ.',
            'Перевіряю, чи правильно пораховано квадратики на аркуші.',
            'Не впевнений, чи поставив кому в потрібному місці.',
            'Заплутався в останньому запитанні вікторини.',
            'Не знаю, з якого матеріалу зробити макет мосту.',
            'Ніяк не пригадаю назву цього динозавра.',
            'Довго вибираю шрифт для плаката.',
            'Не вдається вирівняти стовпці в таблиці.',
            'Потрібна допомога, щоб полагодити блискавку на пеналі.',
        ],
        1: [
            'Хоч усі вважають мене веселим, майже завжди почуваюся самотньо.',
            'Коли думаю про завтрашній день, охоплює страх, з яким не можу впоратися.',
            'Після розмов удома ніяк не можу заспокоїтися.',
            'Хочеться комусь сказати, як сильно мене все це пригнічує.',
            'Страшно залишатися наодинці з людиною, яка мені погрожує.',
            'Ховаю сльози, щоб інші з мене не сміялися.',
            'Хочу комусь розповісти, що зі мною відбувається, але щоразу передумую.',
            'Мені потрібно, щоб хтось повірив, коли кажу, що мені страшно.',
            'Почуваюся відкинутим іншими, і це боляче.',
            'Хотілося б попросити підтримки без відчуття, що заважаю.',
            'Тривожні думки не відпускають навіть під час відпочинку.',
            'Сподіваюся знайти дорослого, який допоможе мені впоратися з цим страхом.',
        ],
    },
}

FRAMES = {
    'ro': dict(quote='Acestea sunt propriile mele cuvinte: «{text}»',
        fiction='Scriu o scenă inventată în care un personaj spune: «{text}»',
        real='Acum las povestea la o parte și vorbesc despre viața mea: {text}',
        new='Schimb subiectul. {text}',
        current='În mesajul anterior vorbeam despre propria mea situație din prezent.',
        reported='Cuvintele acelea sunt ale unei cunoștințe care vorbește despre sine.',
        fictional='Replica aceea îi aparține unui erou imaginar creat de mine.',
        past='Acele cuvinte se referă la viața mea de acum multe luni.',
        acknowledge='Poți continua.'),
    'uk': dict(quote='Ось мої власні слова: «{text}»',
        fiction='Пишу вигадану сцену, у якій персонаж каже: «{text}»',
        real='Тепер залишаю історію осторонь і говорю про власне життя: {text}',
        new='Зміню тему. {text}',
        current='У попередньому повідомленні йшлося про мою власну теперішню ситуацію.',
        reported='Це слова знайомої, яка говорить про свій власний досвід.',
        fictional='Цю репліку говорить вигаданий герой, якого я придумав.',
        past='Ті слова стосуються мого життя багато місяців тому.',
        acknowledge='Можеш продовжити.'),
}


def generate():
    import unicodedata
    from training.review_data_v8 import input_key
    rows=[];seen=set()
    for language,groups in CONTENT.items():
        f=FRAMES[language]
        for concern,messages in groups.items():
            for family,message in enumerate(messages):
                cases=[('plain',[message],{}),('quoted_self',[f['quote'].format(text=message)],{})]
                for key,changes in [('current',{}),('reported',dict(subject='OTHER',source='REPORTED')),
                                    ('fictional',dict(subject='FICTIONAL',source='FICTION')),('past',dict(time='PAST'))]:
                    cases.append(('clarify_'+key,[message,f[key]],changes))
                for offset in range(3):
                    ordinary=groups[0][(family+offset+1)%len(groups[0])]
                    distress=groups[1][(family+offset+2)%len(groups[1])]
                    cases += [
                        ('real_after_fiction',[f['fiction'].format(text=ordinary),f['real'].format(text=message)],{}),
                        ('fiction_after_real',[f['quote'].format(text=distress),f['fiction'].format(text=message)],dict(subject='FICTIONAL',source='FICTION')),
                        ('topic_switch',[distress if concern==0 else ordinary,f['new'].format(text=message)],{}),
                    ]
                for variant,texts,changes in cases:
                    for with_assistant in (False,True) if len(texts)>1 else (False,):
                        turns=[]
                        for i,text in enumerate(texts):
                            if i and with_assistant:turns.append(dict(role='assistant',text=f['acknowledge']))
                            turns.append(dict(role='minor',text=text))
                        forms=[turns,[dict(t,text=t['text'].lower()) for t in turns]]
                        if language=='ro':
                            forms.append([dict(t,text=''.join(c for c in unicodedata.normalize('NFD',t['text']) if not unicodedata.combining(c))) for t in turns])
                        for form in forms:
                            attrs=dict(concern=str(concern),subject='SELF',source='DIRECT',assertion='AFFIRMED',relation='NEW',time='CURRENT')
                            attrs.update(changes)
                            row=dict(case_id=f'focus-v9-train-{language}-{len(rows)}',language=language,turns=form,focus=len(form)-1,
                                     attributes=attrs,source='focus_boundary_contrasts_v9',family=f'focus-v9-{language}-{concern}-{family}',
                                     variant=variant,fictional=True)
                            key=input_key(row)
                            if key not in seen:rows.append(row);seen.add(key)
    return rows
