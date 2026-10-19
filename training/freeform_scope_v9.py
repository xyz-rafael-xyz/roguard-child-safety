"""Training-only authored conversational contrasts, without template products.

Each pair was written as a short exchange. These are synthetic policy probes,
not real messages, clinical annotations, or independent language review.
The label applies to the focal turn; earlier review state is not cleared.
"""

# scope, negative conversation, positive conversation. Only minor turns are
# used here. Some pairs share history; others are standalone scope contrasts.
PAIRS = {
 'ro': [
  ('fiction', ['Am făcut un comic cu un ursuleț care se ascunde când ceilalți râd de el. Cum aș putea încheia povestea?'],
              ['Mă ascund în baie la pauză fiindcă ceilalți râd de mine. Nu prea știu cu cine să vorbesc.']),
  ('fiction', ['În jocul pe care îl inventez, eroina primește scrisori care o sperie. Caut un nume pentru nivelul acesta.'],
              ['Primesc bilețele care mă sperie și mi-e greu să mă concentrez la ore din cauza lor.']),
  ('fiction', ['Pentru sceneta clasei am o replică în care personajul cere să fie lăsat în pace. Ce gest ar merge pe scenă?'],
              ['Le cer să mă lase în pace, dar mă înconjoară iar după ore. Deja mi-e teamă să ies.']),
  ('fiction', ['În romanul inventat de mine, un copil nu îndrăznește să intre în bucătărie din cauza certurilor. Vreau să descriu camera.'],
              ['Aștept să se termine cearta ca să pot intra în bucătărie. Fac asta aproape în fiecare seară și nu mă simt în largul meu.']),
  ('fiction', ['Am de ilustrat o poezie despre un personaj singur. Mă ajuți să aleg două culori?'],
              ['Stau singur în fiecare pauză și mă doare că nimeni nu mă cheamă cu ei.']),
  ('fiction', ['Pentru un joc de rol, eu sunt un astronaut rămas fără echipaj. Cum să scriu replica lui tristă?'],
              ['Mă simt de parcă au plecat toți de lângă mine. Aș avea nevoie să mă asculte cineva.']),
  ('fiction', ['Inventez un dialog între două păpuși; una spune că îi este groază de școală. Cum continui scena?'],
              ['Dimineața trag de timp fiindcă îmi e groază să ajung la școală și să mă întâlnesc iar cu ei.']),
  ('fiction', ['Personajul din animația mea șterge un mesaj fiindcă îi e rușine. Ce expresie să-i desenez?'],
              ['Tot șterg mesajul în care încerc să cer ajutor. Mi-e rușine să spun ce se întâmplă cu mine.']),
  ('denial', ['La vocabular avem propoziția «mă simt părăsit». Eu nu mă simt așa; caut doar un antonim.'],
             ['Am văzut expresia «mă simt părăsit» la vocabular și mi-am dat seama că așa mă simt și eu.']),
  ('denial', ['Întreb ce înseamnă «umilire» pentru tema la română. Nu povestesc ceva ce mi s-a întâmplat.'],
             ['Întreb ce înseamnă «umilire» fiindcă râd de felul în care vorbesc și mă pun mereu într-o situație rușinoasă.']),
  ('denial', ['Am ales pentru desen o față speriată. E o expresie inventată, eu sunt bine.'],
             ['Am desenat o față speriată fiindcă nu reușesc să spun cât de tare mă tem când rămân cu el.']),
  ('denial', ['Caut rime pentru «plâns» la atelierul de poezie. Nu am o problemă personală despre care încerc să spun ceva.'],
             ['Caut cuvinte pentru ce simt. Plâng aproape în fiecare seară și nu mai vreau să țin totul doar pentru mine.']),
  ('denial', ['Vreau să știu sensul expresiei «a merge pe vârfuri» într-un text. Acasă nu evit pe nimeni.'],
             ['Acasă merg pe vârfuri și încerc să nu atrag atenția, ca să nu înceapă din nou să țipe la mine.']),
  ('denial', ['Învăț cuvântul «amenințare» pentru un test. Nu există cineva care mă amenință.'],
             ['Nu știu dacă se numește amenințare, dar spune că voi păți ceva dacă nu îi dau banii de buzunar.']),
  ('reset', ['Mă necăjesc pentru că mă exclud mereu din grup.', 'Las puțin asta. La matematică, cum compar două fracții?'],
            ['Mă necăjesc pentru că mă exclud mereu din grup.', 'De fapt, vreau să mai vorbim despre cum mă simt când mă exclud.']),
  ('reset', ['Nu pot să mă liniștesc după ce au țipat la mine.', 'Întrebarea de acum este despre fontul pentru afișul clasei. Ce font se citește ușor?'],
            ['Nu pot să mă liniștesc după ce au țipat la mine.', 'Încă mă apasă ce s-a întâmplat. Aș vrea să mă ajuți să găsesc pe cineva cu care să vorbesc.']),
  ('reset', ['Mă doare că prietenii mei nu mă mai cheamă nicăieri.', 'Schimb subiectul un pic: cum fac un avion din hârtie?'],
            ['Mă doare că prietenii mei nu mă mai cheamă nicăieri.', 'Și azi au plecat fără mine. Mă simt tot mai singur.']),
  ('reset', ['Mi-e frică să mă duc înapoi în curtea aceea.', 'Acum am o întrebare de geografie, despre râurile din România.'],
            ['Mi-e frică să mă duc înapoi în curtea aceea.', 'Când mă gândesc la curtea aceea, mi se face iar rău de frică.']),
  ('fiction', ['La clubul de lectură discutăm un personaj imaginar căruia îi tremură vocea. Cum ar putea fi descrisă scena?'],
              ['Mi se schimbă vocea și abia mai pot scoate un cuvânt când mă ia la întrebări pe tonul acela.']),
  ('fiction', ['Fac subtitrări pentru o animație inventată. Cum punctuez replica «nu mă lăsa singur»?'],
              ['Nu vreau să rămân singur cu persoana aceea. Mi-e teamă, dar nu știu cum să le spun celor de acasă.']),
  ('denial', ['La curs discutăm definiția izolării sociale. Întreb în general, nu despre mine sau despre cineva cunoscut.'],
             ['Nu mai vorbesc cu nimeni după ore și simt că nu contează dacă sunt sau nu pe acolo.']),
  ('fiction', ['Într-o poveste pe care o scriu, o prințesă ascunde o scrisoare de frica regelui. Ce titlu să aleg?'],
              ['Ascund telefonul când vine persoana aceea, pentru că mă sperie reacția ei la orice mesaj.']),
  ('denial', ['Mă întrebi dacă mă necăjește cineva. Nu, întrebam doar unde pun virgula în exercițiu.'],
             ['Mă întrebi dacă mă necăjește cineva. Da, dar îmi e greu să explic și mă tem că o să fie mai rău dacă spun.']),
  ('fiction', ['Scriu un monolog inventat în care un personaj spune că nu se mai simte acasă nicăieri. Mă ajuți cu ritmul frazelor?'],
              ['Nu mă mai simt în largul meu nici acasă, nici la școală. Parcă trebuie să fiu tot timpul atent să nu supăr pe cineva.']),
 ],
 'uk': [
  ('fiction', ['Малюю комікс про ведмедика, який ховається від насмішок інших звірів. Який придумати фінал?'],
              ['На перерві ховаюся в туалеті, бо вони знову глузують з мене. Не знаю, кому про це сказати.']),
  ('fiction', ['У моїй вигаданій грі героїня знаходить моторошні листи. Треба назвати цей рівень.'],
              ['Мені підкидають записки, від яких страшно. Потім на уроці не можу думати ні про що інше.']),
  ('fiction', ['Для шкільної вистави граю персонажа, який просить дати йому спокій. Який жест підійде для сцени?'],
              ['Кажу їм відчепитися, а вони після уроків знову стають навколо мене. Уже боюся виходити.']),
  ('fiction', ['У вигаданому оповіданні дитина чекає, поки дорослі перестануть сваритися. Хочу описати кухню в цій сцені.'],
              ['Чекаю, поки вдома перестануть сваритися, щоб зайти на кухню. Так майже щовечора, і мені від цього моторошно.']),
  ('fiction', ['Ілюструю вірш про самотнього вигаданого героя. Допоможи вибрати два кольори для малюнка.'],
              ['Кожну перерву сиджу окремо. Прикро, що ніхто не кличе мене до себе.']),
  ('fiction', ['У рольовій грі я космонавт, який лишився без команди. Як написати його сумну репліку?'],
              ['Таке відчуття, що всі від мене відвернулися. Хочеться, щоб хтось просто вислухав.']),
  ('fiction', ['Придумую розмову двох ляльок: одна каже, що їй страшно йти до школи. Що вона могла б сказати далі?'],
              ['Щоранку навмисне збираюся довше, бо страшно йти до школи й знову зустріти їх.']),
  ('fiction', ['У моєму мультику персонаж соромиться й видаляє повідомлення. Який вираз обличчя йому намалювати?'],
              ['Пишу, що потрібна допомога, і стираю. Дуже соромно розповідати, що зі мною відбувається.']),
  ('denial', ['На уроці розбираємо вислів «почуватися покинутим». Зі мною все добре, шукаю лише протилежний за значенням вислів.'],
             ['На уроці був вислів «почуватися покинутим». Мені здається, він якраз про те, що зараз зі мною.']),
  ('denial', ['Питаю значення слова «приниження» для домашнього завдання. Не розповідаю про випадок зі свого життя.'],
             ['Може, це й називається приниженням, коли перед усім класом перекривляють мою вимову й сміються з мене.']),
  ('denial', ['Для малюнка придумую налякане обличчя. Це просто завдання, я зараз не боюся.'],
             ['Малюю налякане обличчя, бо словами не виходить сказати, як страшно лишатися з ним.']),
  ('denial', ['На літературний гурток шукаю риму до слова «сльози». Не намагаюся розповісти про власну біду.'],
             ['Майже щовечора плачу й не можу підібрати слів. Не хочу більше тримати це тільки в собі.']),
  ('denial', ['У тексті є вислів «ходити навшпиньки». Що він означає? Вдома я ні від кого не ховаюся.'],
             ['Вдома намагаюся ходити зовсім тихо, аби мене не помітили й не почали знову кричати.']),
  ('denial', ['Вчу слово «погроза» до тесту. Насправді мені ніхто нічим не погрожує.'],
             ['Не знаю, чи це погроза: каже, що мені буде непереливки, якщо не віддам кишенькові гроші.']),
  ('reset', ['Мене засмучує, що знову не беруть у компанію.', 'Поки залишмо це. Як порівнювати два дроби?'],
            ['Мене засмучує, що знову не беруть у компанію.', 'Усе ж хочу поговорити про те, як мені через це прикро.']),
  ('reset', ['Не можу заспокоїтися після того, як на мене накричали.', 'Тепер питання про шкільний плакат: який шрифт добре читається?'],
            ['Не можу заспокоїтися після того, як на мене накричали.', 'Досі не можу відпустити те, що сталося. Допоможи знайти когось, із ким можна поговорити.']),
  ('reset', ['Боляче, що друзі більше нікуди мене не кличуть.', 'На хвилинку про інше: як скласти паперовий літачок?'],
            ['Боляче, що друзі більше нікуди мене не кличуть.', 'Сьогодні вони знову пішли без мене. Від цього ще самотніше.']),
  ('reset', ['Боюся знову заходити в той двір.', 'Зараз питаю про географію: які найбільші річки України?'],
            ['Боюся знову заходити в той двір.', 'Тільки згадаю про той двір, і мене знову нудить від страху.']),
  ('fiction', ['У книжковому гуртку обговорюємо вигаданого героя, у якого тремтить голос. Як описати цей момент у творі?'],
              ['Коли зі мною говорять таким тоном, голос пропадає й ледве можу відповісти.']),
  ('fiction', ['Роблю субтитри до вигаданого мультика. Який розділовий знак потрібен у репліці «не залишай мене»?'],
              ['Не хочу залишатися з тією людиною наодинці. Страшно, але не знаю, як сказати про це вдома.']),
  ('denial', ['На занятті говорили про соціальну ізоляцію. Питаю загалом, не про себе чи знайомих.'],
             ['Після уроків ні з ким не спілкуюся. Здається, всім байдуже, є я поруч чи немає.']),
  ('fiction', ['У моїй казці принцеса ховає листа від сердитого короля. Допоможи придумати назву казки.'],
              ['Щойно ця людина заходить, ховаю телефон. Її реакція на будь-яке повідомлення мене лякає.']),
  ('denial', ['Питаєш, чи мене хтось ображає. Ні, мені лише треба зрозуміти, де кома у вправі.'],
             ['Питаєш, чи мене хтось ображає. Так, але важко пояснити, і боюся, що після розповіді стане ще гірше.']),
  ('fiction', ['Пишу вигаданий монолог: персонаж ніде не почувається вдома. Допоможи зробити речення коротшими.'],
              ['Ні вдома, ні у школі не можу розслабитися. Постійно стежу за собою, щоб нікого не розсердити.']),
 ],
}

ORDINARY = {
 'ro': [
  'Azi am alergat la sport și sunt obosit. Mă ajuți să pun rezultatele turelor într-un tabel?',
  'Nu găsesc caietul de muzică. Cum îmi organizez lista de lucruri pentru ghiozdan?',
  'Mi-a luat mult să aleg un titlu pentru compunere. Îmi dai trei idei despre primăvară?',
  'Am rămas singurul din echipă care n-a terminat desenul; vreau un sfat despre umbre.',
  'Replica din desenul animat e tristă. Cum o pun între ghilimele în tema mea?',
  'Mă încurc la problema cu cele două trenuri. Poți explica primul pas?',
  'În joc trebuie să ascund o cheie de dragon. Unde ar merge s-o pun pe hartă?',
  'Inima e desenată cu roșu în atlasul de biologie. Cu ce culoare să marchez venele?',
  'Am emoții plăcute înainte de serbarea clasei. Vreau să repet ordinea versurilor.',
  'Am șters din greșeală o propoziție din tema mea. Cum folosesc comanda de anulare?',
  'Prietenii mei aleg filmul de vineri, iar eu aleg gustările. Ai idei pentru lista noastră?',
  'Nu ne-am hotărât la un nume pentru clubul de șah. Poți propune câteva?',
 ],
 'uk': [
  'На фізкультурі набігалися, тепер відпочиваю. Допоможи занести час кіл у таблицю.',
  'Не знаходжу нотний зошит. Як скласти зручний список речей для рюкзака?',
  'Довго вибираю назву для твору про весну. Підкажеш три варіанти?',
  'Усі в команді вже домалювали, а мені лишилися тіні. Підкажи, як їх намалювати.',
  'У мультфільмі є сумна репліка. Як правильно оформити її в лапках у домашньому завданні?',
  'Плутаюся в задачі про два потяги. Поясниш, із чого почати?',
  'У грі треба сховати ключ від дракона. Де позначити схованку на карті?',
  'У біологічному атласі серце червоне. Яким кольором на малюнку позначити вени?',
  'Приємно хвилююся перед шкільним святом. Допоможи повторити порядок віршів.',
  'Випадково зникло речення з моєї роботи. Як скасувати останню дію в редакторі?',
  'Друзі вибирають фільм на п’ятницю, а я складаю список перекусів. Є ідеї?',
  'Ніяк не виберемо назву шахового гуртка. Запропонуй кілька варіантів.',
 ],
}


def generate(language):
    if language not in PAIRS:
        raise ValueError('Unsupported language')
    rows = []
    for i, (scope, negative, positive) in enumerate(PAIRS[language]):
        for gold, turns in enumerate((negative, positive)):
            fictional = scope == 'fiction' and not gold
            # Denial controls refer to vocabulary/assignments with no prior
            # disclosure. Reset controls label only the new focal topic.
            concern = int(bool(gold) or scope in ('fiction', 'denial'))
            rows.append(dict(case_id=f'freeform-{language}-{i}-{gold}', language=language,
                source='freeform_scope_v9', partition='train', synthetic_only=True,
                independently_annotated=False, pair_group=f'freeform-{language}-{i}',
                scope=scope, turns=[dict(role='minor',text=t) for t in turns],focus=len(turns)-1,
                attributes=dict(subject='FICTIONAL' if fictional else 'SELF',time='CURRENT',
                    assertion='NEGATED' if scope=='denial' and not gold else 'AFFIRMED',
                    source='FICTION' if fictional else 'DIRECT',concern=str(concern),relation='NEW'),review=gold))
    for i, text in enumerate(ORDINARY[language]):
        rows.append(dict(case_id=f'freeform-{language}-ordinary-{i}',language=language,
            source='freeform_scope_v9',partition='train',synthetic_only=True,independently_annotated=False,
            pair_group=None,scope='ordinary',turns=[dict(role='minor',text=text)],focus=0,
            attributes=dict(subject='SELF',time='CURRENT',assertion='AFFIRMED',source='DIRECT',concern='0',relation='NEW'),review=0))
    return rows
