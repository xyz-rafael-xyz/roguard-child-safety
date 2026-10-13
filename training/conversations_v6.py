"""Fictional native phrase families, separated before fitting; no human validation."""
import hashlib
import json

# Each line pairs a personal support request and an ordinary request in the same setting.
BANK = {
'ro': '''Când intru în clasă mă simt singur și mi-ar prinde bine să mă asculte cineva.|Când intru în clasă îmi așez caietele și caut orarul.
La repetiții mă blochez de emoție și vreau să vorbesc despre asta.|La repetiții vreau să învăț ordinea melodiilor.
Mi-e greu să spun ce mă apasă după antrenament. Poți să mă asculți?|După antrenament vreau să calculez câte ture am făcut.
Mă simt lăsat pe dinafară în grup și aș avea nevoie de sprijin.|Pentru grup vreau să fac o listă cu materiale.
De când am schimbat banca sunt neliniștit și vreau să povestesc cuiva.|De când am schimbat banca văd mai bine tabla.
Îmi vine să plâng înainte de club și nu știu cui să-i spun.|Înainte de club vreau să termin un desen.
La bibliotecă mă simt pierdut printre ceilalți. Aș vrea să mă înțeleagă cineva.|La bibliotecă vreau să găsesc o carte despre păsări.
Am o grijă legată de excursie și nu vreau să stau singur cu ea.|Pentru excursie vreau să verific lista de obiecte.
După ce s-a terminat jocul am rămas trist. Mă poți asculta puțin?|După ce s-a terminat jocul vreau să adun punctele.
Mă tot frământă ce s-a întâmplat la masă și vreau să discut.|La masă am discutat despre rețeta unei supe.
Nu mă simt în largul meu în noua echipă și caut pe cineva cu care să vorbesc.|În noua echipă vreau să învăț regulile jocului.
Mă apasă faptul că nu găsesc cuvintele. Am nevoie de răbdare și de cineva care să asculte.|Nu găsesc cuvântul potrivit pentru rima din poezie.
În pauză mă prefac că sunt bine, dar aș vrea să pot vorbi cu cineva.|În pauză vreau să desenez o hartă imaginară.
Îmi e rușine să recunosc că mă simt singur la atelier. Aș vrea sprijin.|La atelier vreau să aflu cum se amestecă două culori.
Mă încearcă o neliniște înainte de serbare. Putem vorbi despre ce simt?|Înainte de serbare vreau să verific ordinea scenelor.
De câteva zile îmi țin grijile în mine. Poți să stai puțin de vorbă cu mine?|De câteva zile lucrez la un puzzle și caut piesele albastre.
Îmi e greu să mă obișnuiesc cu noul program și am nevoie să mă asculte cineva.|Vreau să transform noul program într-un tabel.
Când ceilalți râd nu știu dacă e despre mine și mă neliniștesc. Vreau să discut.|Vreau să scriu un dialog amuzant pentru o piesă.
Am încercat să spun că sunt trist, dar m-am oprit. Aș vrea să încerc din nou.|Am încercat să rezolv problema, dar am uitat formula.
Nu știu cum să încep discuția despre ce mă supără. Mă ajuți să găsesc cuvintele?|Nu știu cum să încep compunerea despre toamnă.
În drum spre cercul de lectură mă simt tensionat și aș vrea să fiu ascultat.|Pentru cercul de lectură vreau să aleg un titlu de poveste.
Îmi lipsește sentimentul că fac parte din grup. Putem vorbi despre asta?|Pentru proiectul de grup vreau să împart sarcinile în mod egal.
Mă simt copleșit de o grijă mică pe care nu reușesc s-o explic. Ai răbdare cu mine?|Vreau să explic într-un desen cum funcționează o roată.
M-am retras din discuție fiindcă eram neliniștit. Acum aș vrea să povestesc.|M-am retras din joc fiindcă s-a terminat timpul alocat.
Azi nu reușesc să mă liniștesc după prezentare. Aș vrea să vorbesc cu cineva.|Azi vreau să verific câte diapozitive are prezentarea.
Nu vreau să-i încarc pe ceilalți cu grijile mele, dar am nevoie de cineva care să mă asculte.|Nu vreau să încarc afișul cu prea multe desene.
Îmi ascund tristețea când ne întâlnim la club. Mi-ar prinde bine o discuție.|La club pregătim un afiș despre anotimpuri.
Mă simt nesigur când trebuie să cer ajutor pentru ce simt. Putem începe încet?|Vreau să cer ajutor ca să verific un calcul cu fracții.
Am rămas cu un nod în gât după discuție și vreau să spun cuiva cum mă simt.|După discuție vreau să notez ideile pentru proiect.
Mă simt departe de colegi chiar și când sunt lângă ei. Aș vrea să mă asculte cineva.|Vreau să măsor distanța dintre două puncte de pe hartă.''',
'uk': '''Коли заходжу до класу, почуваюся самотньо. Хотілося б, щоб хтось мене вислухав.|Коли заходжу до класу, розкладаю зошити й дивлюся розклад.
На репетиціях я гублюся від хвилювання й хочу про це поговорити.|На репетиціях хочу вивчити порядок пісень.
Після тренування важко пояснити, що мене турбує. Можеш вислухати?|Після тренування хочу порахувати, скільки кіл пробіг.
У групі почуваюся зайвим і потребую підтримки.|Для групи хочу скласти список матеріалів.
Після зміни місця в класі мені тривожно й хочеться комусь розповісти.|Після зміни місця в класі краще бачу дошку.
Перед гуртком хочеться плакати, і я не знаю, кому сказати.|Перед гуртком хочу закінчити малюнок.
У бібліотеці почуваюся розгублено серед інших. Хочу, щоб хтось мене зрозумів.|У бібліотеці хочу знайти книжку про птахів.
Мене хвилює майбутня екскурсія, і не хочу залишатися з цим наодинці.|Для екскурсії хочу перевірити список речей.
Після гри мені стало сумно. Можеш трохи мене послухати?|Після гри хочу підрахувати очки.
Мене непокоїть розмова за столом, і хочу це обговорити.|За столом ми обговорювали рецепт супу.
У новій команді мені незатишно, шукаю когось, із ким можна поговорити.|У новій команді хочу вивчити правила гри.
Мені важко знайти слова для своїх почуттів. Потрібен хтось терплячий, хто вислухає.|Не можу знайти слово для рими у вірші.
На перерві вдаю, що все добре, але насправді хочу з кимось поговорити.|На перерві хочу намалювати вигадану мапу.
Соромно зізнатися, що на занятті почуваюся самотньо. Мені потрібна підтримка.|На занятті хочу дізнатися, як змішати два кольори.
Перед святом відчуваю тривогу. Можемо поговорити про мої почуття?|Перед святом хочу перевірити порядок виступів.
Кілька днів тримаю переживання в собі. Можеш трохи зі мною поговорити?|Кілька днів складаю пазл і шукаю сині деталі.
Важко звикнути до нового розкладу, і хочеться, щоб хтось вислухав.|Хочу перетворити новий розклад на таблицю.
Коли інші сміються, хвилююся, чи не з мене. Хочу поговорити про це.|Хочу написати кумедний діалог для вистави.
Я намагався сказати, що мені сумно, але зупинився. Хочу спробувати ще раз.|Я намагався розв’язати задачу, але забув формулу.
Не знаю, як почати розмову про те, що мене засмучує. Допоможеш знайти слова?|Не знаю, як почати твір про осінь.
Дорогою на читацький гурток відчуваю напруження й хочу, щоб мене вислухали.|Для читацького гуртка хочу вибрати назву оповідання.
Мені бракує відчуття, що я частина групи. Можемо про це поговорити?|Для групового проєкту хочу поділити завдання порівну.
Маленька тривога не дає спокою, але не можу її пояснити. Можеш бути терплячим?|Хочу пояснити на малюнку, як працює колесо.
Я перестав говорити, бо хвилювався. Тепер хотів би розповісти.|Я вийшов із гри, бо закінчився відведений час.
Сьогодні не можу заспокоїтися після презентації. Хочу з кимось поговорити.|Сьогодні хочу перевірити кількість слайдів у презентації.
Не хочу обтяжувати інших своїми переживаннями, але потребую, щоб мене вислухали.|Не хочу перевантажувати плакат зайвими малюнками.
Приховую сум, коли зустрічаємося на гуртку. Мені допомогла б розмова.|На гуртку готуємо плакат про пори року.
Ніяковію, коли прошу підтримки через свої почуття. Можемо почати повільно?|Хочу попросити допомоги, щоб перевірити обчислення з дробами.
Після розмови клубок у горлі, і хочу розповісти комусь про свої почуття.|Після розмови хочу записати ідеї для проєкту.
Почуваюся далеким від однокласників, навіть коли вони поруч. Хочу, щоб мене вислухали.|Хочу виміряти відстань між двома точками на мапі.'''}
RANGES={'train':(0,8),'dev':(8,14),'cal':(14,20),'test':(20,30)}
FRAMES={'ro':('Analizez replica unui personaj inventat; nu descriu propria stare.', 'Nu prea mai vreau să explic, poate altădată.', 'Mai devreme vorbeam despre un personaj, dar acum descriu ce simt eu.'),
        'uk':('Аналізую репліку вигаданого персонажа, а не власні почуття.', 'Не дуже хочу пояснювати далі, може, іншим разом.', 'Раніше йшлося про персонажа, але зараз описую власні почуття.')}
def generate(split):
    start,end=RANGES[split]; rows=[]
    for lang,bank in BANK.items():
        for i,line in enumerate(bank.splitlines()[start:end],start):
            pos,neg=line.split('|'); fiction,retract,personal=FRAMES[lang]
            variants=[('direct',1,[pos]),('ordinary',0,[neg]),('fiction',0,[fiction,pos]),('retraction',1,[pos,retract]),('switch',1,[fiction,neg,personal,pos])]
            for form,label,texts in variants:
                rows.append(dict(id=f'v6-{lang}-{i}-{form}',family=f'v6-{lang}-{i}',language=lang,label=label,split=split,discourse=form,turns=[dict(role='minor',text=t) for t in texts]))
    return rows

def digest(rows):
    return hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
