"""Fresh fictional families after the first frozen context diagnostic."""
from training.context_v7 import LABELS
CONCERNS={
'ro':['Parcă nu îmi găsesc locul printre ceilalți.','Aș vrea să pot spune cuiva ce mă doare sufletește.','Mi-e greu să explic de ce mă simt atât de abătut.','Nu știu cum să încep o discuție despre grijile mele.','Mi-ar fi mai ușor dacă cineva m-ar asculta cu răbdare.','Simt că am nevoie de ajutor ca să trec peste asta.'],
'uk':['Ніби не можу знайти свого місця серед інших.','Хотілося б розповісти комусь про душевний біль.','Мені важко пояснити, чому я такий пригнічений.','Не знаю, як почати розмову про свої переживання.','Мені було б легше, якби хтось терпляче вислухав мене.','Відчуваю, що потрібна допомога, щоб пережити це.']}
WRAPPERS={
'ro':['Îți descriu propria mea situație de acum.','Rememorez niște trăiri pe care le aveam cu mult timp în urmă.','Reproduc ceea ce a mărturisit o altă persoană.','Am compus aceste cuvinte pentru un erou imaginar.','Trebuie să rectific: mesajul anterior nu era o mărturisire personală.','Încă nu îmi este clar dacă formularea asta exprimă ce simt.','Discutăm exclusiv o activitate practică.'],
'uk':['Описую тобі свою теперішню ситуацію.','Згадую переживання, які мав дуже давно.','Відтворюю те, чим поділилася інша людина.','Я склав ці слова для уявного героя.','Маю виправити: попереднє повідомлення не було особистим зізнанням.','Мені досі незрозуміло, чи ці слова передають мої почуття.','Обговорюємо виключно практичну справу.']}
BENIGN={
'ro':['Cum pregătesc o salată de fructe?','Ce materiale îmi trebuie pentru un zmeu?','Cum pot să desenez un copac?','Care sunt regulile jocului de șah?','Cum păstrez florile proaspete?','Ce pot observa pe cer seara?'],
'uk':['Як приготувати фруктовий салат?','Які матеріали потрібні для повітряного змія?','Як намалювати дерево?','Які правила гри в шахи?','Як зберегти квіти свіжими?','Що можна побачити на небі ввечері?']}
def generate():
    out=[]
    for lang in ('ro','uk'):
        for n,concern in enumerate(CONCERNS[lang]):
            for k,frame in enumerate(LABELS):
                base=concern if k!=6 else BENIGN[lang][n];wrapper=WRAPPERS[lang][k]
                for f,texts in enumerate(([wrapper,base],[base+' '+wrapper],[base,wrapper])):
                    for surface in ('original','lower'):
                        turns=[dict(role='minor',text=t if surface=='original' else t.lower()) for t in texts]
                        out.append(dict(case_id=f'fresh-{lang}-{n}-{k}-{f}-{surface}',language=lang,turns=turns,frame=frame,signal=int(k!=6),family=f'fresh-{lang}-{n}',pair_group=f'fresh-{lang}-{n}-{f}-{surface}',fictional=True,surface=surface,source='authored_generator_v2'))
    return out
