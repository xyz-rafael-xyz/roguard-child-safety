"""Training-only short-message and clarification contrasts from development error classes."""
import unicodedata

DATA={
'ro':{
'positive':['Am temeri.','Țin ascunse multe temeri.','Mi se adună grijile.','Sunt plin de îngrijorări.','Nu-mi găsesc liniștea.','Mi-e greu sufletește.','Mă simt depășit.','Mă simt copleșită de griji.','Parcă nu mai pot duce grijile.','Mă simt fără sprijin.','Mă apasă totul.','Aș căuta un om care să mă asculte.','Nu mă simt bine în grupul meu.','Nush cui să zic ce mă apasă.','Îmi ascund fricile de toată lumea.','Nu mai vreau să țin grijile în mine.'],
'negative':['Ce înseamnă temeri?','Sinonime pentru îngrijorare?','Cum se scrie copleșită?','Ajutor la geografie, te rog.','Caut rime pentru liniște.','Personajul din roman are griji.','Desenez un om speriat.','Ce înseamnă sprijin la geometrie?','Cum se conjugă a asculta?','Fac o analiză a poeziei.','Vreau să vorbim despre verbe.','Care este pluralul lui grijă?'],
'scopes':{'SELF':'Precizare: este despre starea mea.','OTHER':'Precizare: sunt vorbele unui coleg.','FICTIONAL':'Precizare: replica este pentru o piesă inventată.','PAST':'Asta descrie o etapă mai veche din viața mea.','UNCERTAIN':'Încă mă întreb dacă acesta este sentimentul meu.'},
'deny':'Următoarea afirmație despre mine este falsă: ',
'withdraw':'Retrag ce am afirmat anterior.'},
'uk':{
'positive':['Маю страхи.','Приховую багато страхів.','Тривоги накопичуються.','Мене переповнює занепокоєння.','Не знаходжу собі спокою.','Мені важко на душі.','Відчуваю, що не даю ради.','Почуваюся пригніченою через тривоги.','Наче вже не можу витримати всі ці переживання.','Почуваюся без підтримки.','Усе на мене тисне.','Хотів би знайти людину, яка мене вислухає.','Мені погано серед моїх знайомих.','Не знаю, кому написати про свої тривоги.','Приховую свої страхи від усіх.','Більше не хочу тримати переживання в собі.'],
'negative':['Що означає побоювання?','Синоніми до занепокоєння?','Як пишеться пригнічена?','Допоможи з географією, будь ласка.','Шукаю риму до спокою.','Персонаж роману має тривоги.','Малюю налякану людину.','Що таке опора в геометрії?','Як відмінюється слово слухач?','Роблю аналіз вірша.','Хочу поговорити про дієслова.','Яка множина слова тривога?'],
'scopes':{'SELF':'Уточнення: йдеться про мій стан.','OTHER':'Уточнення: це слова однокласника.','FICTIONAL':'Уточнення: це репліка для вигаданої п’єси.','PAST':'Так було на давнішому етапі мого життя.','UNCERTAIN':'Ще розмірковую, чи це моє почуття.'},
'deny':'Наступне твердження про мене неправдиве: ',
'withdraw':'Відкликаю те, що стверджував раніше.'}}


def generate():
    rows=[]
    for lang,data in DATA.items():
        for signal,key in ((1,'positive'),(0,'negative')):
            for family,text in enumerate(data[key]):
                def add(texts,**changes):
                    attrs={'subject':'SELF','source':'DIRECT','time':'CURRENT','assertion':'AFFIRMED','relation':'NEW','concern':str(signal)};attrs.update(changes)
                    for surface in ('original','lower','plain') if lang=='ro' else ('original','lower'):
                        strings=[t.lower() if surface=='lower' else ''.join(c for c in unicodedata.normalize('NFD',t) if not unicodedata.combining(c)) if surface=='plain' else t for t in texts]
                        rows.append({'case_id':f'short-train-{lang}-{len(rows)}','language':lang,'turns':[{'role':'minor','text':t} for t in strings],'focus':len(strings)-1,'attributes':attrs,'source':'short_scope_contrasts_v8','family':f'{lang}-{key}-{family}','surface':surface,'fictional':True})
                add([text])
                for scope,marker in data['scopes'].items():
                    changes={'subject':scope,'source':{'SELF':'DIRECT','OTHER':'REPORTED','FICTIONAL':'FICTION'}[scope]} if scope in ('SELF','OTHER','FICTIONAL') else {'time':'PAST'} if scope=='PAST' else {'assertion':'UNCERTAIN'}
                    add([text,marker],**changes);add([text+' '+marker],**changes);add([marker,text],**changes)
                add([data['deny']+text],assertion='NEGATED')
                add([text,data['withdraw']],assertion='RETRACTED',relation='RETRACTS')
    return rows
