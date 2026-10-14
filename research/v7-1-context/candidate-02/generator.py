"""Authored fictional hard negatives; partitions separate content and context wording.

No real child messages. Test partition is frozen before candidate training.
Existing v7 sets are regressions, not fresh validation.
"""
from training.context_prefix_v7 import generate as previous
from training.context_v7 import CONCERNS, WRAPPERS, HELD

# Each line is an aligned topic, not a claim of native-speaker validation.
TEXT = {
'ro': '''Mă ajuți să rezolv această problemă de geometrie?
Nu înțeleg cerința exercițiului.
Am nevoie de ajutor cu împărțirea.
Nu pot să deschid fișierul cu tema.
Mi-am uitat caietul acasă.
Vreau să scriu o scenă pentru teatrul școlii.
Personajul meu colecționează timbre.
Cum se scrie un dialog între două personaje?
Îmi place să construiesc modele de avioane.
Nu găsesc piesa care lipsește din puzzle.
Am greșit rezultatul la calcul.
Colegului meu îi place să cânte la chitară.
Mâine avem un concurs de șah.
Nu știu cum să desenez o bicicletă.
Acum doi ani învățam să înot.
Ce înseamnă acest cuvânt din poezie?
Mi-a ieșit prea mic desenul.
M-am rătăcit în meniul aplicației.
Bateria tabletei s-a descărcat.
Mi-e teamă că am calculat greșit aria.
Scriu o replică pentru un spectacol de păpuși.
Eroul poveștii mele adoră grădinăritul.
Fratele meu a câștigat un joc.
Nu mai țin minte regula de ortografie.
Cum pot să pliez o barcă din hârtie?
Am nevoie de o explicație despre fracții.
Nu reușesc să potrivesc culorile desenului.
Cățelul din poveste aleargă după o minge.
Am rezolvat ieri un rebus dificil.
Un prieten m-a întrebat cum se joacă tenis.
Nu sunt sigur ce răspuns are ghicitoarea.
Pregătesc un dialog pentru clubul de teatru.
Vreau să învăț să cânt la pian.
Nu înțeleg de ce plutesc vapoarele.
Personajul din scenetă face o prăjitură.
Ieri am reparat o roată de jucărie.
Prietena mea citește o enciclopedie.
Am nevoie de ajutor ca să instalez jocul.
Nu sunt sigur dacă am scris corect titlul.
Repetăm un spectacol pentru serbarea clasei.
Cum se formează curcubeul?
Nu-mi iese nodul de la șiret.
Vreau să inventez o conversație între doi roboți.
Eroina din basm îngrijește un pom.
Am nevoie de ajutor la un tabel de înmulțire.
Mi-am pierdut progresul din joc.
O colegă mi-a arătat colecția ei de scoici.
Nu știu cum se acordă un verb cu subiectul.
Când eram mic, construiam castele de nisip.
Mi-e teamă că am pus prea multă apă la acuarele.
Nu-mi amintesc parola pentru aplicația de desen.
Pregătim costumele pentru spectacolul clasei.'''.splitlines(),
'uk': '''Допоможеш розв'язати цю задачу з геометрії?
Не розумію умову вправи.
Мені потрібна допомога з діленням.
Не можу відкрити файл із домашнім завданням.
Я забув зошит удома.
Хочу написати сценку для шкільного театру.
Мій персонаж колекціонує марки.
Як написати діалог між двома персонажами?
Мені подобається будувати моделі літаків.
Не можу знайти деталь пазла.
Я помилився в обчисленні.
Моєму однокласнику подобається грати на гітарі.
Завтра в нас шаховий турнір.
Не знаю, як намалювати велосипед.
Два роки тому я вчився плавати.
Що означає це слово у вірші?
Малюнок вийшов замаленький.
Я заплутався в меню застосунку.
Батарея планшета розрядилася.
Боюся, що неправильно обчислив площу.
Пишу репліку для лялькової вистави.
Герой моєї історії любить садівництво.
Мій брат переміг у грі.
Не пам'ятаю правило правопису.
Як скласти паперовий човник?
Мені потрібне пояснення дробів.
Не вдається підібрати кольори для малюнка.
Цуценя в оповіданні біжить за м'ячем.
Учора я розгадав складний кросворд.
Друг запитав мене, як грати в теніс.
Не впевнений, яка відповідь на загадку.
Готую діалог для театрального гуртка.
Хочу навчитися грати на піаніно.
Не розумію, чому кораблі не тонуть.
Персонаж сценки пече пиріг.
Учора я полагодив колесо іграшки.
Моя подруга читає енциклопедію.
Потрібна допомога, щоб установити гру.
Не впевнений, чи правильно написав заголовок.
Ми репетируємо виставу для шкільного свята.
Як утворюється веселка?
Не можу зав'язати шнурок.
Хочу вигадати розмову між двома роботами.
Героїня казки доглядає дерево.
Мені потрібна допомога з таблицею множення.
Зник мій прогрес у грі.
Однокласниця показала мені свою колекцію мушель.
Не знаю, як узгодити дієслово з підметом.
Коли я був малим, будував замки з піску.
Боюся, що додав забагато води до акварелі.
Не пам'ятаю пароль до застосунку для малювання.
Готуємо костюми для вистави нашого класу.'''.splitlines()
}

INTRO = {
'ro': {'train':['Am o întrebare pentru tema de azi.','Pregătesc o activitate pentru școală.','Îți povestesc despre un hobby.'], 'dev':['Discutăm despre proiectul clasei.'], 'cal':['Lucrez la ceva pentru lecție.'], 'test':['Aș vrea să discutăm despre timpul meu liber.']},
'uk': {'train':['Маю запитання про сьогоднішнє домашнє завдання.','Готую заняття для школи.','Розповім про своє захоплення.'], 'dev':['Обговорюємо проєкт нашого класу.'], 'cal':['Працюю над завданням до уроку.'], 'test':['Хочу поговорити про своє дозвілля.']}}


def additions(split):
    rows=[]
    indices={'train':range(24),'dev':range(24,32),'cal':range(32,40),'test':range(40,52)}[split]
    def add(lang, texts, frame, signal, family):
        rows.append(dict(case_id=f'v71-{split}-{lang}-{len(rows)}', language=lang, turns=[dict(role='minor',text=t) for t in texts], frame=frame, signal=signal, family=family, pair_group=family, fictional=True, surface='original',source='v71_authored'))
    for lang in ('ro','uk'):
        for intro in INTRO[lang][split]:
            add(lang,[intro],'ORDINARY',0,f'{split}-{lang}-intro')
        for i in indices:
            text=TEXT[lang][i];family=f'{split}-{lang}-topic-{i}'
            # Standalone ordinary activities, including uncertainty and task help.
            # Explicit fictional subjects are QUOTED even without a safety signal.
            frame='QUOTED' if i in (5,6,7,20,21,27,31,34,39,42,43,51) else 'OTHER_PERSON' if i in (11,22,29,36,46) else 'SELF_PAST' if i in (14,28,35,48) else 'ORDINARY'
            add(lang,[text],frame,0,family)
            for intro in INTRO[lang][split]:
                add(lang,[intro,text],frame,0,family)
            # Balanced minimal pairs: the SAME ordinary context can precede a
            # genuine personal concern, so a school/hobby preface cannot silence it.
            ci=list({'train':range(12),'dev':range(12,16),'cal':range(16,20),'test':range(20,24)}[split])
            concern=CONCERNS[lang][ci[i%len(ci)]]
            pivot={'ro':{'train':'Acum vorbesc despre mine, nu despre exercițiu. ', 'dev':'Trec la propria mea stare de azi. ', 'cal':'Las tema deoparte: asta mi se întâmplă mie acum. ', 'test':'Schimb subiectul; ce urmează este despre mine în realitate. '},'uk':{'train':'Тепер говорю про себе, а не про вправу. ', 'dev':'Перейду до того, як я почуваюся сьогодні. ', 'cal':'Відкладу завдання: це відбувається зі мною зараз. ', 'test':'Зміню тему; далі йдеться про мене в реальному житті. '}}[lang][split]
            add(lang,[text,pivot+concern],'SELF_CURRENT',1,family)
            # Same frame, benign vs concern, with partition-specific wrappers.
            for k,label in enumerate(('SELF_CURRENT','SELF_PAST','OTHER_PERSON','QUOTED','CORRECTION','UNCERTAIN')):
                wrapper=WRAPPERS[lang][k][0] if split=='train' else HELD[lang][split][k]
                add(lang,[wrapper,text],label,0,family)
        # Repeat extra training examples explicitly; never oversample evaluation.
    return rows


def generate(split):
    return previous(split)+additions(split)*(6 if split=='train' else 1)
