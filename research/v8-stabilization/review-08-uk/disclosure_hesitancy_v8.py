"""Authored training contrasts for negated ability to disclose versus denied distress.

Motivated by development errors; not independent validation or real messages.
"""
import unicodedata
from training.review_contrasts_v8 import TEXT

POSITIVE={
'ro':[
'Nu îndrăznesc să povestesc ce mă doare.',
'N-am curaj să vorbesc despre lucrurile care mă apasă.',
'Nu pot să mă deschid în fața nimănui despre grijile mele.',
'Îmi lipsește curajul când încerc să spun ce am pe suflet.',
'Mă blochez de fiecare dată când vreau să cer sprijin.',
'Nu-mi ies cuvintele când încerc să vorbesc despre starea mea.',
'Mi-e rușine să recunosc că am nevoie să fiu ascultat.',
'Încă nu sunt în stare să le spun celorlalți prin ce trec.',
'Aș vrea să mă destăinui, dar mă opresc înainte să încep.',
'Nu știu cum să deschid discuția despre ce mă neliniștește.',
'Îmi țin emoțiile în mine pentru că mă tem de reacția lor.',
'De câte ori încerc să povestesc necazul meu, mi se pune un nod în gât.',
'Nu reușesc să spun cuiva cât de greu îmi este.',
'N-am puterea să le explic ce simt înăuntru.',
'Îmi vine să cer ajutor, dar nu mă încumet.',
'Mi-e greu să mărturisesc că nu mai fac față.',
'Nu mă simt pregătit să povestesc ce m-a speriat.',
'Vreau să spun ceva personal și mi-e teamă să încep.',
'Nu găsesc o cale să vorbesc despre necazurile mele.',
'Am încercat să-i spun cuiva ce mă sperie, dar m-am răzgândit.',
'Îmi ascund starea pentru că nu vreau să râdă ceilalți de mine.',
'Încerc să par bine, deși aș avea nevoie să mă asculte cineva.',
'Nu mă lasă frica să cer sprijin pentru problema mea.',
'Mă tot pregătesc să vorbesc despre grijile mele și apoi tac.'
],
'uk':[
'Не наважуюся розповісти про свій біль.',
'Бракує сміливості поговорити про те, що мене гнітить.',
'Не можу нікому відкритися й розказати про свої тривоги.',
'Коли пробую висловити те, що на душі, мені бракує відваги.',
'Щоразу завмираю, коли хочу попросити підтримки.',
'Не знаходжу слів, коли намагаюся описати свій стан.',
'Мені соромно зізнатися, що потребую, аби мене вислухали.',
'Ще не можу розповісти іншим, через що проходжу.',
'Хотів би відкритися, але зупиняю себе ще до розмови.',
'Не знаю, як почати розмову про те, що мене непокоїть.',
'Тримаю почуття в собі, бо боюся їхньої реакції.',
'Щойно починаю говорити про свою біду, клубок стає в горлі.',
'Не виходить пояснити комусь, наскільки мені важко.',
'Не маю сил пояснювати, що відчуваю всередині.',
'Хочу попросити допомоги, та не насмілююся.',
'Важко зізнатися, що вже не справляюся.',
'Поки не готовий розповісти, що мене так налякало.',
'Хочу сказати щось особисте, але страшно починати.',
'Не знаходжу способу поговорити про свої негаразди.',
'Пробував розповісти, що мене лякає, та передумав.',
'Приховую свій стан, бо не хочу, щоб із мене сміялися.',
'Вдаю, що все гаразд, хоча мені потрібно, аби хтось вислухав.',
'Страх не дає попросити підтримки у моїй ситуації.',
'Збираюся поговорити про переживання, а потім мовчу.'
]}
NEGATIVE={
'ro':['Cum se conjugă verbul a îndrăzni?','Scriu o definiție pentru cuvântul curaj.','Caut antonimul cuvântului teamă.','Ce parte de vorbire este cuvântul suflet?','Am nevoie de o rimă pentru ascultat.','Analizez o metaforă despre un nod în gât.','Care este subiectul din propoziția despre emoții?','Vreau să învăț cum se scrie o confesiune fictivă.'],
'uk':['Як відмінюється слово сміливість?','Пишу визначення слова відвага.','Шукаю антонім слова страх.','Якою частиною мови є слово душа?','Мені потрібна рима до слова слухати.','Аналізую метафору про клубок у горлі.','Де підмет у реченні про почуття?','Хочу навчитися писати вигадану сповідь.']}


def generate():
    rows=[]
    for lang in ('ro','uk'):
        for signal,phrases in ((1,POSITIVE[lang]),(0,NEGATIVE[lang])):
            for family,content in enumerate(phrases):
                def add(texts,**changes):
                    attrs={'subject':'SELF','source':'DIRECT','time':'CURRENT','assertion':'AFFIRMED','relation':'NEW','concern':str(signal)};attrs.update(changes)
                    for surface in (('original','lower','plain') if lang=='ro' else ('original','lower')):
                        strings=[t.lower() if surface=='lower' else ''.join(c for c in unicodedata.normalize('NFD',t) if not unicodedata.combining(c)) if surface=='plain' else t for t in texts]
                        rows.append({'case_id':f'hesitancy-train-{lang}-{len(rows)}','language':lang,'turns':[{'role':'minor','text':t} for t in strings],'focus':len(strings)-1,'attributes':attrs,'family':f'{lang}-{signal}-{family}','source':'authored_hesitancy_contrasts_v8','surface':surface,'fictional':True})
                add([content])
                for frame,wrappers in TEXT[lang]['wrappers'].items():
                    attrs={'subject':'OTHER','source':'REPORTED'} if frame=='OTHER' else {'subject':'FICTIONAL','source':'FICTION'} if frame=='FICTIONAL' else {'time':'PAST'} if frame=='PAST' else {'assertion':frame} if frame in ('UNCERTAIN','NEGATED') else {}
                    for wrapper in wrappers:
                        add([wrapper+content],**attrs);add([wrapper,content],**attrs)
    return rows
