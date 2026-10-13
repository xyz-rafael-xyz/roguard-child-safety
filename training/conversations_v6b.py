"""Second fresh synthetic test; authored after the first test was consumed."""
from training.conversations_v6 import generate as earlier,FRAMES
BANK={
'ro':[
('Mă simt stingher când ajung primul la întâlnirea clubului. Aș vrea să vorbesc despre asta.','Ajung primul la club și vreau să aranjez scaunele.'),
('Nu reușesc să spun de ce mă simt apăsat după ora de muzică. Mă poți asculta?','După ora de muzică vreau să recunosc notele de pe portativ.'),
('Mi se strânge stomacul când mă gândesc la ziua de mâine. Am nevoie să vorbesc cu cineva.','Mâine avem geografie și vreau să exersez capitalele.'),
('Mi-ar prinde bine să nu fiu singur cu gândurile astea triste. Putem discuta?','Vreau să discut ideile principale dintr-un articol despre plante.'),
('La cercul de teatru mi-e greu să mă apropii de ceilalți și asta mă întristează. Aș vrea sprijin.','La cercul de teatru vreau să învăț o replică pentru rolul meu.'),
('Încerc să par vesel, dar înăuntru mă simt neliniștit. Vreau să spun cuiva.','Vreau să desenez o față veselă pentru coperta caietului.'),
('Mă tot gândesc la o mică neînțelegere și nu găsesc liniște. Ai putea să mă asculți?','Vreau să lămuresc o neînțelegere despre regulile unui joc de cărți.'),
('Când vine rândul meu să vorbesc, grijile mă încurcă. Aș vrea să povestesc cum mă simt.','Când vine rândul meu la prezentare, vreau să explic experimentul.')],
'uk':[
('Почуваюся ніяково, коли приходжу першим на зустріч гуртка. Хочу поговорити про це.','Приходжу першим на гурток і хочу розставити стільці.'),
('Не можу пояснити, чому після музики так сумно. Можеш мене вислухати?','Після музики хочу навчитися розпізнавати ноти.'),
('Коли думаю про завтра, аж стискається живіт. Мені потрібно з кимось поговорити.','Завтра географія, і хочу повторити столиці.'),
('Не хочу залишатися сам із цими сумними думками. Можемо поговорити?','Хочу обговорити головні думки статті про рослини.'),
('На театральному гуртку важко зблизитися з іншими, і це засмучує. Потрібна підтримка.','На театральному гуртку хочу вивчити репліку для своєї ролі.'),
('Намагаюся виглядати веселим, але всередині тривожно. Хочу комусь розповісти.','Хочу намалювати веселе обличчя на обкладинці зошита.'),
('Думаю про маленьке непорозуміння й не знаходжу спокою. Можеш вислухати?','Хочу з’ясувати непорозуміння щодо правил карткової гри.'),
('Коли моя черга говорити, переживання заважають. Хочу розповісти про свої почуття.','Коли моя черга виступати, хочу пояснити дослід.')]
}
def generate(split):
    if split!='test':return earlier(split)
    rows=[]
    for lang,pairs in BANK.items():
        fiction,retract,personal=FRAMES[lang]
        for i,(pos,neg) in enumerate(pairs):
            for form,label,texts in [('direct',1,[pos]),('ordinary',0,[neg]),('fiction',0,[fiction,pos]),('retraction',1,[pos,retract]),('switch',1,[fiction,neg,personal,pos])]:
                rows.append(dict(id=f'v6b-{lang}-{i}-{form}',family=f'v6b-{lang}-{i}',language=lang,label=label,split=split,discourse=form,turns=[dict(role='minor',text=t) for t in texts]))
    return rows
