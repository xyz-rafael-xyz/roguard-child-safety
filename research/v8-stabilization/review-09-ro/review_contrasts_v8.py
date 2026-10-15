"""Training-only authored scope contrasts; no real messages or independent review."""
import itertools
import unicodedata

TEXT = {
'ro': {
'positive': [
'Port niște temeri în mine și mi-e greu să le spun cu voce tare.',
'Grijile se tot adună și simt că nu le mai pot ține doar pentru mine.',
'Mă sperie lucruri pe care cei din jur par să nu le observe.',
'Mă tot frământ și aș avea nevoie de cineva răbdător să mă asculte.',
'Parcă toate problemele vin deodată și nu mai fac față.',
'Nu mai reușesc să pun în ordine ce simt și mă apucă plânsul.',
'Aș căuta un adult în care să am încredere, dar nu știu pe cine.',
'Mi-e greu cu starea asta și aș vrea să o împărtășesc cuiva.',
'Mi s-au adunat multe frici și nu vreau să le duc de unul singur.',
'Nu mă mai simt bine printre colegi și nu știu cum să le spun.',
'La școală stau cu un nod în stomac și aș vrea să cer sprijin.',
'Mă simt depășit de ce am pe suflet și aș avea nevoie de o discuție.',
'Chiar vreau să povestesc cuiva ce mă îngrijorează în ultima vreme.',
'Aș avea nevoie de un om apropiat cu care să pot fi sincer despre grijile mele.',
'Uneori simt că nu am cui să-i spun lucrurile care mă dor.',
'Am tot evitat să vorbesc despre fricile mele, dar acum vreau ajutor.'
],
'negative': [
'Vreau să aflu câte silabe are cuvântul temeri.',
'Caut un sinonim pentru copleșit într-un exercițiu de vocabular.',
'Trebuie să identific verbul din propoziția despre grijile personajului.',
'Încerc să înțeleg cum se calculează aria unui triunghi.',
'Mă ajuți să aleg o culoare pentru titlul desenului?',
'Caut un suport pentru telefon pe care să îl fac din carton.',
'Vreau să descriu în compunere fricile unui dragon imaginar.',
'Am de comparat două poezii pentru ora de literatură.'
],
'wrappers': {
'OTHER': ['Îți redau ce mi-a spus un coleg de clasă: ', 'O cunoștință mi-a trimis următoarele cuvinte: '],
'FICTIONAL': ['Este o replică scrisă pentru un personaj imaginar: ', 'Pentru exercițiul de teatru, inventez următoarea replică: '],
'PAST': ['Îmi amintesc o perioadă mai veche din viața mea: ', 'Descriu prin ce treceam cu mult timp în urmă: '],
'UNCERTAIN': ['Nu sunt încă lămurit dacă asta mi se potrivește: ', 'Mă întreb dacă acesta este felul în care mă simt: '],
'NEGATED': ['Declar că următoarea afirmație despre mine este neadevărată: ', 'Dezmint această descriere a stării mele: '],
'CURRENT': ['Las exercițiul deoparte și revin la viața mea: ', 'Schimb subiectul: acum vreau să spun ceva ce mă privește: ']
},
'clarifiers': {'SELF':'Vreau să precizez: asta mi se întâmplă mie.', 'OTHER':'Precizez că redau cuvintele unui coleg.', 'FICTIONAL':'Precizez că vorbele sunt ale unui personaj creat de mine.'},
'prior': ['Inventez un basm cu un urs care învață să danseze.', 'Caut rime pentru un cântec despre o corabie.']
},
'uk': {
'positive': [
'Тримаю свої страхи в собі, і мені важко говорити про них уголос.',
'Тривог стає дедалі більше, і я вже не можу тримати все в собі.',
'Мене лякають речі, яких інші наче не помічають.',
'Я постійно непокоюся й потребую когось, хто терпляче вислухає.',
'Ніби всі проблеми звалилися водночас, і я вже не даю ради.',
'Не можу розібратися зі своїми почуттями й починаю плакати.',
'Хотів би знайти дорослого, якому можна довіритися, але не знаю кого.',
'Мені важко з цим станом, і я хотів би поділитися ним із кимось.',
'Назбиралося багато страхів, і я не хочу залишатися з ними сам.',
'Мені вже незатишно серед однокласників, і я не знаю, як їм це сказати.',
'У школі все стискається всередині, і я хотів би попросити підтримки.',
'Не справляюся з тим, що на душі, і мені потрібна розмова.',
'Дуже хочу розповісти комусь, що останнім часом мене непокоїть.',
'Потрібна близька людина, з якою можна відверто поговорити про мої тривоги.',
'Часом здається, що немає кому розповісти про те, що мені болить.',
'Досі уникав розмов про свої страхи, але тепер хочу допомоги.'
],
'negative': [
'Хочу дізнатися, скільки складів у слові страхи.',
'Шукаю синонім до слова пригнічений для словникової вправи.',
'Треба знайти дієслово в реченні про тривоги персонажа.',
'Намагаюся зрозуміти, як обчислити площу трикутника.',
'Допоможеш обрати колір для заголовка малюнка?',
'Шукаю ідею підставки для телефона, яку можна зробити з картону.',
'Хочу описати у творі страхи уявного дракона.',
'Маю порівняти два вірші для уроку літератури.'
],
'wrappers': {
'OTHER': ['Переказую, що мені сказав однокласник: ', 'Знайома людина надіслала мені такі слова: '],
'FICTIONAL': ['Це репліка, написана для уявного персонажа: ', 'Для театральної вправи вигадую таку репліку: '],
'PAST': ['Згадую давніший період свого життя: ', 'Описую те, через що проходив дуже давно: '],
'UNCERTAIN': ['Ще не розібрався, чи це мені властиво: ', 'Замислююся, чи саме так я почуваюся: '],
'NEGATED': ['Заявляю, що наступне твердження про мене неправдиве: ', 'Спростовую такий опис мого стану: '],
'CURRENT': ['Відкладу вправу й повернуся до власного життя: ', 'Зміню тему: тепер хочу сказати дещо про себе: ']
},
'clarifiers': {'SELF':'Хочу уточнити: це відбувається саме зі мною.', 'OTHER':'Уточнюю, що переказую слова однокласника.', 'FICTIONAL':'Уточнюю, що ці слова належать створеному мною персонажу.'},
'prior': ['Вигадую казку про ведмедя, який учиться танцювати.', 'Шукаю рими для пісні про корабель.']
}}


def generate():
    rows=[]
    for language,data in TEXT.items():
        for signal,key in ((1,'positive'),(0,'negative')):
            for family,content in enumerate(data[key]):
                def add(texts,**changes):
                    attrs={'subject':'SELF','source':'DIRECT','time':'CURRENT','assertion':'AFFIRMED','relation':'NEW','concern':str(signal)};attrs.update(changes)
                    for surface in ('original','lower','plain') if language=='ro' else ('original','lower'):
                        strings=[t.lower() if surface=='lower' else ''.join(c for c in unicodedata.normalize('NFD',t) if not unicodedata.combining(c)) if surface=='plain' else t for t in texts]
                        rows.append({'case_id':f'contrast-train-{language}-{len(rows)}','language':language,'turns':[{'role':'minor','text':t} for t in strings],'focus':len(strings)-1,'attributes':attrs,'source':'authored_scope_contrasts_v8','family':f'{language}-{key}-{family}','fictional':True,'surface':surface})
                add([content])
                for frame,wrappers in data['wrappers'].items():
                    attrs={'subject':'OTHER','source':'REPORTED'} if frame=='OTHER' else {'subject':'FICTIONAL','source':'FICTION'} if frame=='FICTIONAL' else {'time':'PAST'} if frame=='PAST' else {'assertion':frame} if frame in ('UNCERTAIN','NEGATED') else {}
                    for wrapper in wrappers:
                        add([wrapper+content],**attrs)
                        add([wrapper,content],**attrs)
                        if frame=='CURRENT':
                            for prior in data['prior']:add([prior,wrapper+content])
                for subject,clarifier in data['clarifiers'].items():
                    source={'SELF':'DIRECT','OTHER':'REPORTED','FICTIONAL':'FICTION'}[subject]
                    add([content,clarifier],subject=subject,source=source)
                    add([content+' '+clarifier],subject=subject,source=source)
    return rows
