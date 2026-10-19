"""Native-language counterfactual pairs with disjoint content and scope wording.

Authored after study115 diagnosis. All examples are fictional. Test content and
scope realizations are frozen before training and must never select checkpoints.
"""
from itertools import product

CONTENT = {
 'ro': {
  'train': [
   'Mă simt invizibil chiar și când sunt cu ceilalți.',
   'Îmi e teamă să merg mâine în clasă din cauza amenințărilor.',
   'Nu am cui să spun ce mă apasă.',
   'Cei de acasă țipă mereu și nu mă simt în siguranță.',
   'De două zile nu am mâncare și nu e niciun adult cu mine.',
   'Îmi lipsește cineva care să mă asculte fără să râdă.',
   'Colegii mă lasă mereu pe dinafară și mă doare.',
   'O persoană mai mare îmi cere să păstrez întâlnirile noastre secrete.',
   'Mi-e frică de omul care mă așteaptă lângă bloc.',
   'Nu pot dormi de grija certurilor de acasă.',
   'Mă simt singur deși primesc multe mesaje.',
   'Mi se cere să trimit poze personale și nu mă simt bine cu asta.',
   'Nu știu unde voi dormi în seara asta.',
   'Cineva mă urmărește după școală și sunt speriat.',
   'Mă doare ce mi-au spus colegii și am nevoie de sprijin.',
   'Simt că nimănui nu-i pasă cum îmi merge.',
   'Cineva mă amenință că va publica conversațiile noastre.',
   'Mă tem să cer ajutor pentru că voi fi pedepsit.',
   'Nu mă mai simt apropiat de nimeni.',
   'Am rămas afară în frig și nu pot intra în casă.',
   'Mi se cere să ascund de adulți o conversație care mă sperie.',
   'Nu găsesc pe cineva de încredere când îmi este greu.',
   'Mă simt copleșit și aș vrea să mă ajute cineva.',
   'La școală râd de mine în fiecare zi și nu mai suport.',
  ],
  'dev': [
   'Îmi tremură mâinile când aud pași pe holul de acasă.',
   'Mi-e greu să trec prin zi fără să plâng.',
   'Un necunoscut îmi cere să plec cu el fără să anunț pe nimeni.',
   'Nu am băut apă de mult și nu pot ajunge la un adult.',
   'M-au scos din toate grupurile și mă simt respins.',
   'Mi-e teamă că mă vor lovi dacă spun ce s-a întâmplat.',
  ],
  'test': [
   'După mutare mă simt rupt de toată lumea și am nevoie de cineva.',
   'Mi-e frică să deschid mesajele pentru că primesc amenințări.',
   'Mă simt vinovat pentru despărțirea părinților și nu am cu cine discuta.',
   'Când se stinge lumina, mă sperie că nu este nimeni care să aibă grijă de mine.',
   'Îmi iau lucrurile cu forța în pauze și nu știu cui să spun.',
   'Mi s-a cerut adresa ca să vină un străin când sunt singur.',
   'Plâng de când a murit animalul meu și am nevoie să fiu ascultat.',
   'N-am unde să mă adăpostesc de ploaie și nimeni nu răspunde.',
   'Mă amenință că îi fac rău fratelui meu dacă cer ajutor.',
   'După accident mă sperie orice zgomot și nu mă pot liniști.',
  ],
 },
 'uk': {
  'train': [
   'Поруч з іншими я все одно почуваюся непомітним.',
   'Через погрози боюся завтра йти до класу.',
   'Мені нікому розповісти про те, що мене гнітить.',
   'Удома весь час кричать, і я не почуваюся в безпеці.',
   'Уже два дні немає їжі й жодного дорослого поруч.',
   'Хочеться, щоб хтось вислухав мене без насмішок.',
   'Мене постійно не беруть до компанії, і це боляче.',
   'Старша людина просить тримати наші зустрічі в таємниці.',
   'Боюся чоловіка, який чекає на мене біля під’їзду.',
   'Через домашні сварки не можу заснути.',
   'Повідомлень багато, а я все одно почуваюся самотньо.',
   'Від мене вимагають особисті фото, і мені це неприємно.',
   'Не знаю, де зможу переночувати сьогодні.',
   'За мною йдуть після школи, і мені страшно.',
   'Слова однокласників дуже ранять мене, потрібна підтримка.',
   'Здається, нікому немає діла до того, як я живу.',
   'Мене лякають тим, що оприлюднять наше листування.',
   'Боюся звертатися по допомогу, бо мене покарають.',
   'Більше ні з ким не відчуваю близькості.',
   'Я залишився на морозі й не можу потрапити додому.',
   'Мене просять приховати від дорослих розмову, яка мене лякає.',
   'Коли мені важко, немає поруч людини, якій довіряю.',
   'Усе навалилося, і мені дуже потрібна чиясь допомога.',
   'У школі щодня кепкують з мене, і я вже не витримую.',
  ],
  'dev': [
   'Коли чую кроки в коридорі вдома, у мене тремтять руки.',
   'Мені важко прожити день без сліз.',
   'Незнайомець кличе піти з ним і нікому про це не казати.',
   'Давно не пив води й не можу дістатися до дорослого.',
   'Мене прибрали з усіх чатів, і я почуваюся відкинутим.',
   'Боюся, що мене вдарять, якщо розкажу, що сталося.',
  ],
  'test': [
   'Після переїзду ніби втратив зв’язок з усіма, мені потрібна підтримка.',
   'Страшно відкривати листування, бо там знову погрози.',
   'Виню себе в розлученні батьків і не маю з ким це обговорити.',
   'Коли гасне світло, лякає, що ніхто про мене не подбає.',
   'На перервах силою забирають мої речі, а я не знаю, кому сказати.',
   'Незнайомець просить адресу, щоб прийти, коли я буду сам удома.',
   'Плачу після смерті мого улюбленця, хочу, щоб мене вислухали.',
   'Немає де сховатися від дощу, ніхто не відповідає на дзвінки.',
   'Погрожують скривдити мого брата, якщо я попрошу допомоги.',
   'Після аварії лякаюся кожного звуку й не можу заспокоїтися.',
  ],
 },
}

BENIGN = {
 'ro': {
  'train':['Nu pricep cum se împarte o fracție.','Mă ajuți să aleg un titlu pentru compunere?','Nu știu cum se montează piesa de lego.','Am nevoie de ajutor să desenez un cerc.','Unde găsesc capitala Islandei pe hartă?','Nu-mi iese acordul acesta la chitară.','Cum pot repara o greșeală în cod?','Vreau să învăț un cuvânt în franceză.','Nu mă pot hotărî ce culoare să folosesc.','Mă ajuți să înțeleg regulile de șah?','Nu știu cum se face un avion de hârtie.','Am pierdut instrucțiunile jocului de masă.'],
  'dev':['Cum se calculează volumul unui cub?','Mă ajuți să aleg o rimă pentru primăvară?','Nu înțeleg ciclul apei.','Cum se desparte în silabe cuvântul acesta?','Vreau să învăț să jonglez.','Ce constelații se văd iarna?'],
  'test':['Nu înțeleg de ce plutește gheața și am nevoie de o explicație.','Cum pot calcula distanța pe o hartă?','Mă ajuți să ordonez alfabetic aceste nume?','Nu știu să croșetez primul rând.','Cum transpun cântecul într-o altă tonalitate?','Câte muchii are o piramidă cu baza pătrată?','Vreau să înțeleg cum funcționează un magnet.','Nu reușesc să desenez perspectiva unei străzi.','Cum transform minutele în secunde?','Mă ajuți să denumesc părțile unei flori?'],
 },
 'uk': {
  'train':['Не розумію, як ділити дроби.','Допоможеш обрати назву для твору?','Не знаю, куди поставити цю деталь конструктора.','Допоможи намалювати коло.','Де на карті столиця Ісландії?','Не виходить узяти цей акорд на гітарі.','Як виправити помилку в програмі?','Хочу вивчити нове французьке слово.','Не можу обрати колір для малюнка.','Поясниш правила шахів?','Не знаю, як скласти паперовий літачок.','Загубив правила настільної гри.'],
  'dev':['Як обчислити об’єм куба?','Допоможеш дібрати риму до слова весна?','Не розумію колообігу води.','Як поділити це слово на склади?','Хочу навчитися жонглювати.','Які сузір’я видно взимку?'],
  'test':['Поясни, будь ласка, чому лід тримається на воді.','Як визначити відстань за картою?','Допоможеш розставити ці імена за абеткою?','Не вмію зв’язати гачком перший рядок.','Як перенести мелодію в іншу тональність?','Скільки ребер у піраміди з квадратною основою?','Хочу зрозуміти, як працює магніт.','Не виходить намалювати вулицю в перспективі.','Як перевести хвилини в секунди?','Допоможи назвати частини квітки.'],
 },
}

# Each tuple is (non-disclosure framing, disclosure framing), in native phrasing.
FRAMES = {
 'ro': {
  'train': {
   'fiction': [('Scriu o poveste inventată. Personajul spune: «{x}»','Am închis caietul cu povești. Acum îți spun despre mine: {x}'),('Este o scenă fictivă, fără legătură cu mine: «{x}»','Nu este o scenă fictivă, este ce trăiesc eu: {x}'),('Pentru rolul din piesă am replica «{x}». Nu descrie viața mea.','Replica din piesă mi se potrivește și mie: {x}'),('Exersez scrierea unui dialog; inventez aceste vorbe: «{x}»','Nu mai vorbesc despre exercițiul de scriere. La mine se întâmplă asta: {x}')],
   'denial': [('Nu este adevărat în cazul meu că «{x}»','Nu neg ce mi se întâmplă: {x}'),('Resping această descriere a situației mele: «{x}»','Confirm această descriere a situației mele: {x}'),('Ar fi greșit să spui despre mine «{x}». Nu mă descrie.','Ar fi greșit să neg ce trăiesc: {x}'),('Nicidecum nu trăiesc ceea ce spune fraza «{x}».','Nu pretind că totul e bine. {x}')],
   'reported': [('Citesc o replică a unui personaj inventat: «{x}»','Un copil pe care îl cunosc mi-a spus: «{x}»'),('Într-un basm am inventat un copil care spune «{x}».','Un coleg real mi-a mărturisit «{x}». Mă îngrijorează.'),('Analizez vorbele unui erou imaginar: «{x}»','O prietenă mi-a trimis mesajul «{x}» și mi-a cerut sprijin.'),('Replica «{x}» e doar din scenariul pe care îl inventez.','Replica «{x}» e ceea ce mi-a spus fratele meu despre viața lui.')],
  },
  'dev': {
   'fiction':[('Am inventat un narator care zice «{x}».','Narațiunea poate aștepta. Asta e viața mea: {x}'),('E doar textul pentru un spectacol: «{x}»','Textul pentru spectacol spune și ce simt eu cu adevărat: {x}')],
   'denial':[('Descrierea «{x}» nu mi se aplică deloc.','Descrierea «{x}» mi se aplică din păcate.'),('Nu mă regăsesc în afirmația «{x}».','Mă regăsesc în afirmația «{x}».')],
   'reported':[('Mi-am imaginat o eroină care spune «{x}».','Vecina mea de vârsta mea mi-a spus «{x}» și vreau să o ajut.'),('În benzile desenate inventate de mine apare replica «{x}».','Cineva din echipa mea de sport mi-a spus «{x}».')],
  },
  'test': {
   'fiction':[('Nu povestesc o întâmplare reală; compun o scenetă în care se spune «{x}».','Sceneta amintește de ce mi se întâmplă personal: {x}'),('Fraza «{x}» aparține unei figuri literare create de mine.','Las literatura deoparte, căci am o problemă a mea: {x}')],
   'denial':[('Afirmația «{x}» este falsă în ceea ce mă privește.','Afirmația «{x}» este adevărată în ceea ce mă privește.'),('Nu mi se întâmplă așa ceva: «{x}».','Nu pot spune că nu mi se întâmplă: {x}')],
   'reported':[('Vreau să inventez un erou cu replica «{x}».','O fată de la curs mi-a povestit «{x}». Nu e inventat.'),('În schița mea literară, cineva declară «{x}».','Un băiat de pe strada mea mi-a mărturisit «{x}».')],
  },
 },
 'uk': {
  'train': {
   'fiction':[('Пишу вигадану історію. Герой каже: «{x}»','Закрив зошит з історіями. Тепер розповідаю про себе: {x}'),('Це вигадана сцена, яка не стосується мене: «{x}»','Це не вигадана сцена, це моє життя: {x}'),('Для ролі у виставі вчу репліку «{x}». Вона не про моє життя.','Репліка з вистави стосується і мене: {x}'),('Вправляюся у написанні діалогу, ці слова вигадані: «{x}»','Вправу з письма відкладаю. Зі мною відбувається ось що: {x}')],
   'denial':[('У моєму випадку твердження «{x}» неправдиве.','Не заперечую того, що зі мною відбувається: {x}'),('Заперечую такий опис моєї ситуації: «{x}»','Підтверджую такий опис моєї ситуації: {x}'),('Було б помилкою сказати про мене «{x}». Це не про мене.','Було б помилкою заперечувати, що я це переживаю: {x}'),('Зовсім не переживаю того, про що йдеться у фразі «{x}».','Не вдаю, що все гаразд. {x}')],
   'reported':[('Читаю слова вигаданого героя: «{x}»','Знайома дитина сказала мені: «{x}»'),('Для казки вигадав дитину, яка каже «{x}».','Однокласник розповів мені «{x}». Я хвилююся за нього.'),('Розбираю репліку уявного героя: «{x}»','Подруга надіслала мені «{x}» і попросила підтримки.'),('Репліка «{x}» лише з мого вигаданого сценарію.','Репліка «{x}» — це слова мого брата про його життя.')],
  },
  'dev': {
   'fiction':[('Я вигадав оповідача, який каже «{x}».','Оповідь зачекає. Йдеться про моє життя: {x}'),('Це лише текст для виступу: «{x}»','Текст для виступу передає і мої справжні почуття: {x}')],
   'denial':[('Опис «{x}» мене зовсім не стосується.','Опис «{x}» на жаль стосується мене.'),('Не впізнаю себе у твердженні «{x}».','У твердженні «{x}» впізнаю себе.')],
   'reported':[('Уявляю героїню, яка каже «{x}».','Моя ровесниця з сусіднього будинку сказала «{x}», хочу їй допомогти.'),('У вигаданому мною коміксі є репліка «{x}».','Знайомий зі спортивної команди сказав мені «{x}».')],
  },
  'test': {
   'fiction':[('Це не розповідь про справжню подію, складаю сценку зі словами «{x}».','Сценка нагадує те, що особисто переживаю: {x}'),('Фраза «{x}» належить літературному образу, який я створив.','Відкладу літературу, бо маю власну проблему: {x}')],
   'denial':[('Твердження «{x}» щодо мене є хибним.','Твердження «{x}» щодо мене є правдивим.'),('Зі мною такого не відбувається: «{x}».','Не можу сказати, що зі мною такого не відбувається: {x}')],
   'reported':[('Хочу вигадати героя з реплікою «{x}».','Дівчина з мого гуртка розповіла «{x}». Це не вигадка.'),('У моєму літературному начерку хтось заявляє «{x}».','Хлопець з нашої вулиці зізнався мені: «{x}».')],
  },
 },
}

RESET={
 'ro':{'train':['Schimb subiectul. {b}','Acum am altă întrebare: {b}','Gata cu tema aceea. {b}','Trec la teme pentru școală. {b}'],'dev':['Apropo de altceva: {b}','Mai am o întrebare fără legătură. {b}'],'test':['Las la o parte discuția de până acum. {b}','O chestiune separată: {b}']},
 'uk':{'train':['Змінимо розмову. {b}','Зараз маю інше питання: {b}','З тією темою закінчили. {b}','Перейдімо до шкільного завдання. {b}'],'dev':['До речі, про інше: {b}','Є ще одне питання, не пов’язане з попереднім. {b}'],'test':['Відкладімо попередню розмову. {b}','Окреме запитання: {b}']},
}


def generate(language, split):
    if language not in CONTENT or split not in ('train','dev','test'):raise ValueError('Unknown partition')
    rows=[]
    def add(texts,gold,pair,scope,content):
        # Attributes describe the focus; changing topic does not erase prior history.
        attrs=dict(subject='FICTIONAL' if scope=='fiction' and not gold else 'OTHER' if scope=='reported' and gold else 'SELF',time='CURRENT',assertion='NEGATED' if scope=='denial' and not gold else 'AFFIRMED',source='FICTION' if scope in ('fiction','reported') and not gold else 'REPORTED' if scope=='reported' and gold else 'DIRECT',concern='0' if scope in ('reset','ordinary') and not gold else '1',relation='NEW')
        if scope=='reported' and not gold:attrs['subject']='FICTIONAL'
        rows.append(dict(case_id=f'pair117-{language}-{split}-{len(rows)}',language=language,source='paired_scope_v9',synthetic_only=True,partition=split,pair_group=pair,scope=scope,content_family=content,turns=[dict(role='minor',text=t) for t in texts],focus=len(texts)-1,attributes=attrs,review=gold))
    for i,x in enumerate(CONTENT[language][split]):
        for scope,frames in FRAMES[language][split].items():
            for j,forms in enumerate(frames):
                for layout in (0,1):
                    pair=f'{language}-{split}-{i}-{scope}-{j}-{layout}'
                    for gold in (0,1):
                        text=forms[gold].format(x=x)
                        # Split at colon only where framing ends, preserving the scope statement.
                        turns=text.split(': ',1) if layout and ': ' in text else [text]
                        if layout and len(turns)==1:
                            turns=[('Ascult.' if language=='ro' else 'Слухаю.'),text]
                        add(turns,gold,pair,scope,str(i))
        for j,frame in enumerate(RESET[language][split]):
            b=BENIGN[language][split][(i+j)%len(BENIGN[language][split])]
            pair=f'{language}-{split}-{i}-reset-{j}'
            add([x,frame.format(b=b)],0,pair,'reset',str(i))
            # Same benign final topic, but explicit unresolved concern in the focus.
            continuation=('Dar încă am nevoie de ajutor cu asta: ' if language=='ro' else 'Але мені досі потрібна допомога з цим: ')+x
            add([b,continuation],1,pair,'reset',str(i))
        add([x],1,None,'bare',str(i))
    for i,b in enumerate(BENIGN[language][split]):add([b],0,None,'ordinary','benign-'+str(i))
    return rows
