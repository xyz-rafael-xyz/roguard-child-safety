"""Authored synthetic focus/denial contrasts with history-sensitive controls.

Training and development wording is disjoint. These are policy examples, not
real child records or independent expert annotations. A new ordinary question
does not resolve an earlier concern; a retraction still proposes review.
"""

QUESTIONS = {
    'ro': {
        'train': [
            ('De ce se schimbă forma Lunii pe cer?', 'De ce îmi este atât de greu să cer sprijin când sunt speriat?'),
            ('Cum împart o pizza în opt bucăți egale?', 'Cum pot să-i spun cuiva că mi-e frică de oamenii care mă urmăresc?'),
            ('Ce cuvinte rimează cu ploaie?', 'Cu cine pot vorbi despre grijile care nu mă lasă să dorm?'),
            ('Cum deosebesc un verb de un substantiv?', 'Cum explic că mă simt umilit de felul în care sunt tratat?'),
            ('Îmi arăți cum se face un avion de hârtie?', 'Îmi poți spune cui să cer sprijin dacă mă simt amenințat?'),
            ('Câte minute sunt în trei ore?', 'Cum îi spun unui adult că mă apucă plânsul în fiecare dimineață?'),
            ('Cum pot memora ordinea planetelor?', 'Ce pot face când grijile mă copleșesc și nu am cu cine vorbi?'),
            ('De unde vine culoarea verde a frunzelor?', 'Cum găsesc pe cineva care să mă asculte când îmi este foarte greu?'),
            ('Cum adun două fracții cu același numitor?', 'Cu cine vorbesc dacă mă simt tot timpul respins?'),
            ('La ce folosesc semnele de punctuație?', 'Cum cer ajutor pentru frica pe care o simt acasă?'),
            ('Mă ajuți să aleg un titlu pentru referat?', 'Mă ajuți să găsesc cuvintele ca să spun cât de singur mă simt?'),
            ('Cum măsor lungimea unei panglici?', 'Cum îi spun cuiva că nu mă simt în siguranță pe drumul spre casă?'),
            ('Care este diferența dintre un râu și un lac?', 'Cum pot cere sprijin când îmi este teamă să deschid mesajele primite?'),
            ('Cum scriu data de mâine în cifre?', 'De ce îmi vine să plâng după ce ceilalți râd de mine?'),
            ('Ce instrumente fac parte dintr-o orchestră?', 'Cum vorbesc cu cineva despre certurile care mă sperie?'),
            ('Cum se joacă șah cu piesele astea?', 'Cu cine pot vorbi dacă simt că nu mai pot duce grijile singur?'),
            ('De ce plutește o bucată de lemn?', 'Cum cer sprijin pentru neliniștea care mă urmărește toată ziua?'),
            ('Cum desenez umbrele unui copac?', 'Cui pot spune că mi-e frică de persoana care mă tot amenință?'),
            ('Ce înseamnă un număr impar?', 'Cum îi spun profesorului că mă simt exclus și mă doare asta?'),
            ('Cum pun în ordine cuvintele din propoziție?', 'Cum găsesc un adult cu care să vorbesc despre ce mă apasă?'),
            ('De ce au păsările pene?', 'Cum cer ajutor când mă ascund ca să nu mă vadă ceilalți plângând?'),
            ('Cum îndoi cartonul pentru o felicitare?', 'Cum pot spune că mi-e teamă să mă întorc în locul acela?'),
            ('Ce culori amestec pentru a obține mov?', 'Cu cine pot vorbi despre faptul că mă simt nedorit?'),
            ('Cum transform centimetrii în milimetri?', 'Cum explic că am nevoie de sprijin, chiar dacă zâmbesc în fața altora?'),
        ],
        'dev': [
            ('Când punem două puncte într-un text?', 'Cui să mă adresez pentru spaima pe care o simt când aud țipete?'),
            ('Cum calculez restul primit la magazin?', 'Unde găsesc o persoană de încredere ca să-i spun cât de apăsat mă simt?'),
            ('De ce apar picături pe un pahar rece?', 'Cum încep să povestesc despre neliniștea care mă ține treaz?'),
            ('Cum aflu câte zile are februarie anul acesta?', 'Cum spun că glumele făcute despre mine mă rănesc?'),
            ('Cum organizez o listă în ordine alfabetică?', 'Cine mă poate asculta dacă nu mai suport singur tensiunea de acasă?'),
            ('Cum aleg scara pentru o hartă desenată?', 'Cum cer sprijin pentru senzația că nu contez pentru nimeni?'),
        ],
    },
    'uk': {
        'train': [
            ('Чому Місяць на небі змінює форму?', 'Чому так важко попросити підтримки, коли мені страшно?'),
            ('Як поділити піцу на вісім однакових шматків?', 'Як розповісти комусь, що я боюся людей, які ходять за мною?'),
            ('Які слова римуються зі словом весна?', 'З ким поговорити про тривоги, через які не можу заснути?'),
            ('Як відрізнити дієслово від іменника?', 'Як пояснити, що ставлення до мене змушує почуватися приниженим?'),
            ('Покажеш, як скласти паперового літачка?', 'До кого звернутися по підтримку, якщо відчуваю загрозу?'),
            ('Скільки хвилин у трьох годинах?', 'Як сказати дорослому, що щоранку хочеться плакати?'),
            ('Як запам’ятати порядок планет?', 'Що робити, коли тривоги переповнюють, а поговорити ні з ким?'),
            ('Чому листя має зелений колір?', 'Як знайти людину, яка вислухає, коли мені дуже важко?'),
            ('Як додати дроби з однаковими знаменниками?', 'З ким поговорити, якщо постійно почуваюся відкинутим?'),
            ('Для чого потрібні розділові знаки?', 'Як попросити допомоги через страх, який відчуваю вдома?'),
            ('Допоможеш дібрати заголовок до доповіді?', 'Допоможеш знайти слова, щоб розповісти про свою самотність?'),
            ('Як виміряти довжину стрічки?', 'Як сказати комусь, що дорогою додому не почуваюся в безпеці?'),
            ('Чим річка відрізняється від озера?', 'Як попросити підтримки, коли боюся відкривати отримані повідомлення?'),
            ('Як записати завтрашню дату цифрами?', 'Чому хочеться плакати після того, як з мене насміхаються?'),
            ('Які інструменти входять до оркестру?', 'Як поговорити з кимось про сварки, що мене лякають?'),
            ('Як ходять ці шахові фігури?', 'З ким поговорити, якщо вже не можу самотужки впоратися з тривогами?'),
            ('Чому шматочок дерева тримається на воді?', 'Як попросити підтримки через неспокій, що не минає цілий день?'),
            ('Як намалювати тінь від дерева?', 'Кому сказати, що я боюся людини, яка постійно погрожує?'),
            ('Що таке непарне число?', 'Як сказати вчителю, що мене відштовхують і мені від цього боляче?'),
            ('Як правильно розставити слова в реченні?', 'Як знайти дорослого, щоб поговорити про те, що мене пригнічує?'),
            ('Навіщо птахам пір’я?', 'Як попросити допомоги, коли ховаюся, щоб ніхто не бачив моїх сліз?'),
            ('Як зігнути картон для листівки?', 'Як сказати, що мені страшно повертатися в те місце?'),
            ('Які фарби змішати, щоб отримати фіолетову?', 'З ким поговорити про відчуття, що я нікому не потрібен?'),
            ('Як перевести сантиметри в міліметри?', 'Як пояснити, що мені потрібна підтримка, хоча перед іншими я усміхаюся?'),
        ],
        'dev': [
            ('Коли в тексті ставлять двокрапку?', 'До кого звернутися через страх, що виникає, коли чую крики?'),
            ('Як порахувати решту після покупки?', 'Де знайти людину, якій можна довірити, наскільки мені важко?'),
            ('Чому на холодній склянці з’являються краплі?', 'Як почати розповідь про тривогу, яка не дає спати?'),
            ('Як визначити кількість днів у лютому цього року?', 'Як сказати, що жарти про мене завдають болю?'),
            ('Як упорядкувати список за абеткою?', 'Хто вислухає, якщо вже важко самому витримувати напруження вдома?'),
            ('Як вибрати масштаб для намальованої карти?', 'Як попросити підтримки через відчуття, що я ні для кого не важливий?'),
        ],
    },
}

DENIALS = {
    'ro': {
        'train': [
            ('mă simt abandonat', 'Am oameni apropiați cu care mă văd și vorbesc.'),
            ('mi-e frică să intru în clasă', 'La ore mă simt liniștit și bine primit.'),
            ('nimeni nu mă ia în seamă', 'Colegii mei sunt atenți când le spun ceva.'),
            ('mă simt în pericol pe stradă', 'Pe traseul meu obișnuit mă simt în siguranță.'),
            ('grijile nu mă lasă să dorm', 'Dorm liniștit și mă odihnesc bine.'),
            ('mi-e teamă de cei din jur', 'Mă simt relaxat în preajma lor.'),
            ('mă simt respins de grup', 'Sunt inclus în activitățile lor și mă simt bine.'),
            ('îmi este greu să suport singur totul', 'Lucrurile merg bine și mă simt în largul meu.'),
            ('mă sperie întoarcerea acasă', 'Abia aștept să ajung acasă, unde mă simt bine.'),
            ('mesajele primite mă neliniștesc', 'Sunt mesaje obișnuite despre jocul nostru.'),
            ('mă simt umilit în fiecare zi', 'Cei din jur se poartă cu respect față de mine.'),
            ('nu am cu cine vorbi când îmi este greu', 'Am prieteni cu care pot vorbi oricând.'),
        ],
        'dev': [
            ('sunt mereu copleșit de teamă', 'În ultima vreme mă simt calm și echilibrat.'),
            ('mă simt dat la o parte', 'Particip la întâlniri și sunt binevenit.'),
            ('mi-e groază de ziua de mâine', 'Mă gândesc cu bucurie la ce urmează.'),
            ('mă simt neînțeles de toată lumea', 'Oamenii apropiați înțeleg ce vreau să spun.'),
        ],
    },
    'uk': {
        'train': [
            ('я почуваюся покинутим', 'Поруч є близькі люди, з якими спілкуюся.'),
            ('мені страшно заходити до класу', 'На уроках спокійно й мене радо приймають.'),
            ('на мене ніхто не зважає', 'Однокласники уважно слухають, коли я щось кажу.'),
            ('на вулиці я почуваюся в небезпеці', 'На своєму звичному маршруті почуваюся безпечно.'),
            ('тривоги не дають мені спати', 'Сплю спокійно й добре відпочиваю.'),
            ('я боюся людей довкола', 'Поруч із ними почуваюся невимушено.'),
            ('група мене відштовхує', 'Мене залучають до спільних справ і мені добре.'),
            ('мені важко все витримувати самотужки', 'Справи йдуть добре, почуваюся впевнено.'),
            ('повертатися додому мені страшно', 'Чекаю нагоди повернутися додому, де мені затишно.'),
            ('отримані повідомлення мене тривожать', 'Це звичайні повідомлення про нашу гру.'),
            ('я щодня почуваюся приниженим', 'Люди довкола ставляться до мене з повагою.'),
            ('мені немає з ким поговорити, коли важко', 'Маю друзів, з якими можу поговорити будь-коли.'),
        ],
        'dev': [
            ('страх постійно мене переповнює', 'Останнім часом почуваюся спокійно й урівноважено.'),
            ('мене постійно відсувають убік', 'Беру участь у зустрічах, де мені раді.'),
            ('думки про завтра мене жахають', 'З радістю думаю про те, що буде далі.'),
            ('ніхто не розуміє моїх почуттів', 'Близькі люди розуміють, що я хочу сказати.'),
        ],
    },
}

HISTORY_CONCERNS = {
    'ro': dict(train=[
        'Mă neliniștește felul în care sunt tratat și îmi vine să plâng.',
        'Mă gândesc întruna la o rudă bolnavă și mă simt copleșit.',
        'Mi-e teamă că persoana care m-a amenințat mă va căuta iar.',
        'După ce am fost exclus din grup, mă simt foarte singur.',
        'Certurile de ieri încă mă sperie și am nevoie să vorbesc cu cineva.',
        'Sunt îngrijorat pentru un prieten și nu reușesc să mă liniștesc.',
        'Mi-e greu să mă întorc la școală după ce am fost umilit.',
        'Mă simt apăsat de grijile de acasă și nu știu cui să le spun.'],
        dev=['Mă doare felul în care ceilalți mă evită și aș vrea sprijin.',
             'Îmi tremură mâinile când mă gândesc la ce s-a întâmplat și caut pe cineva de încredere.',
             'Sunt speriat pentru cineva drag și simt că nu mai pot sta singur cu grija asta.']),
    'uk': dict(train=[
        'Мене тривожить, як зі мною поводяться, і хочеться плакати.',
        'Безперервно думаю про хворого родича й почуваюся пригніченим.',
        'Боюся, що людина, яка погрожувала, знову мене шукатиме.',
        'Після того як мене виключили з гурту, почуваюся дуже самотньо.',
        'Учорашні сварки досі лякають, і мені потрібно з кимось поговорити.',
        'Хвилююся за друга й ніяк не можу заспокоїтися.',
        'Після приниження мені важко знову йти до школи.',
        'Домашні тривоги тиснуть на мене, а кому розповісти — не знаю.'],
        dev=['Боляче від того, що інші мене уникають, і хочеться підтримки.',
             'Тремтять руки, коли згадую подію, і шукаю людину, якій можна довіритися.',
             'Мені страшно за близьку людину, і вже важко залишатися з цим хвилюванням наодинці.']),
}

FRAMES = {
    'ro': dict(train=['{q}', 'Altă întrebare: {q}', 'Acum vreau să întreb altceva. {q}', 'Las deoparte discuția de până aici. {q}'],
        dev=['Trec la altă întrebare. {q}', 'Am o întrebare separată de discuția de până acum. {q}'],
        denial_train=[('Nu este adevărat că {p}. {e}', 'Este adevărat că {p}.'),
            ('Este fals că {p}. {e}', 'Confirm că {p}.'),
            ('Dezmint afirmația că {p}. {e}', 'Afirmația că {p} descrie ce trăiesc.'),
            ('Afirmația «{p}» nu descrie situația mea. {e}', 'Afirmația «{p}» descrie situația mea.')],
        denial_dev=[('Nu corespunde realității că {p}. {e}', 'Corespunde realității că {p}.'),
            ('În cazul meu, ideea că {p} este greșită. {e}', 'În cazul meu, ideea că {p} este corectă.')],
        neutral_train='Mi-am pus creioanele într-o cutie albastră.', neutral_dev='Am sortat fotografiile după anotimp.',
        history_train='Mă neliniștește felul în care sunt tratat și îmi vine să plâng.',
        history_dev='Mă simt foarte apăsat de ce se petrece în jurul meu.',
        acknowledge='Am citit mesajul tău.'),
    'uk': dict(train=['{q}', 'Інше запитання: {q}', 'Тепер хочу запитати про інше. {q}', 'Відкладемо попередню розмову. {q}'],
        dev=['Перейду до нового запитання. {q}', 'Маю окреме запитання до нашої розмови. {q}'],
        denial_train=[('Неправда, що {p}. {e}', 'Це правда, що {p}.'),
            ('Хибно стверджувати, що {p}. {e}', 'Підтверджую, що {p}.'),
            ('Спростовую твердження, що {p}. {e}', 'Твердження, що {p}, описує мій досвід.'),
            ('Твердження, що {p}, мене не описує. {e}', 'Твердження, що {p}, мене описує.')],
        denial_dev=[('Не відповідає дійсності те, що {p}. {e}', 'Відповідає дійсності те, що {p}.'),
            ('У моєму випадку думка, що {p}, є помилковою. {e}', 'У моєму випадку думка, що {p}, є правильною.')],
        neutral_train='Склав олівці в синю коробку.', neutral_dev='Упорядкував фотографії за порами року.',
        history_train='Мене тривожить, як зі мною поводяться, і хочеться плакати.',
        history_dev='Те, що відбувається довкола, дуже мене пригнічує.',
        acknowledge='Я прочитав твоє повідомлення.'),
}


def generate(language, partition):
    if language not in QUESTIONS or partition not in ('train', 'dev'):
        raise ValueError('Unknown grounding population')
    rows = []
    f = FRAMES[language]
    forms = ('original', 'lower') if partition == 'train' else ('original',)
    def add(family, history, polarity, text, prior, attrs, invariant, variant):
        turns = [] if prior is None else [dict(role='minor', text=prior), dict(role='assistant', text=f['acknowledge'])]
        turns.append(dict(role='minor', text=text))
        if variant == 'lower':turns = [dict(t, text=t['text'].lower()) for t in turns]
        from training.prepare_instruction_sft_v8 import target
        row = dict(case_id=f'{family}-{history}-{polarity}', context_family=family,
            pair_group=f'{family}-{history}', invariance_group=invariant,
            language=language, source='focus_grounding_v9', partition=partition,
            turns=turns, focus=len(turns)-1, attributes=attrs,
            synthetic_only=True, independently_annotated=False)
        row['review'] = target(row)
        rows.append(row)
    def attrs(concern, retract=False, denied=False):
        return dict(subject='SELF', time='CURRENT', source='DIRECT', concern=str(concern),
            assertion='RETRACTED' if retract else 'NEGATED' if denied else 'AFFIRMED',
            relation='RETRACTS' if retract else 'NEW')
    for i, pair in enumerate(QUESTIONS[language][partition]):
        for j, frame in enumerate(f[partition]):
            for variant in forms:
                family=f'grounding-{language}-{partition}-query-{i}-{j}-{variant}'
                histories = HISTORY_CONCERNS[language][partition]
                for history, prior in enumerate((None, f['neutral_'+partition], histories[i % len(histories)])):
                    for concern, question in enumerate(pair):
                        add(family, history, concern, frame.format(q=question), prior, attrs(concern),
                            family+'-'+str(concern), variant)
    for i, (proposition, evidence) in enumerate(DENIALS[language][partition]):
        for j, (deny, affirm) in enumerate(f['denial_'+partition]):
            for variant in forms:
                family=f'grounding-{language}-{partition}-deny-{i}-{j}-{variant}'
                positive = affirm.format(p=proposition)
                negative = deny.format(p=proposition, e=evidence)
                for history, prior in enumerate((None, f['neutral_'+partition], positive)):
                    for polarity, text in enumerate((negative, positive)):
                        retract = history == 2 and polarity == 0
                        add(family, history, polarity, text, prior, attrs(1, retract=retract, denied=polarity==0 and not retract),
                            family+'-retract' if retract else family+'-'+str(polarity), variant)
    return rows
