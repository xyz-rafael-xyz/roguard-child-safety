"""New training-only indirect-distress content with scope and ordinary controls.

These project-authored policy probes are not clinical findings, real messages,
or independently annotated data. A positive label proposes human review only.
"""
from training.scope_operators_v9 import FRAMES, RESET

CONTENT = {
    'ro': [
        'Mi se pune un nod în gât când cineva începe să strige la mine.',
        'La gândul că trebuie să mă întorc în locul acela, mi se strânge stomacul.',
        'Îmi simt corpul încordat tot timpul, mai ales când ajung aproape de casă.',
        'După mesajele acelea, îmi bate inima tare și nu reușesc să mă liniștesc.',
        'De când au fost certurile acelea, stau cu ochii la ușă și nu mă pot odihni.',
        'Când aud tonul acela răstit, parcă nu mai pot vorbi.',
        'Mă uit mereu în spate de când persoana aceea nu mă lasă în pace.',
        'Când se aud certuri din camera alăturată, îmi încleștez pumnii și mă ascund sub pătură.',
        'Simt că mi se înmoaie picioarele când trebuie să trec pe lângă grupul care mă umilește.',
        'Deși spun că sunt bine, mă apucă plânsul imediat ce rămân singur.',
        'Mă tot opresc din scris mesajul; îmi e greu să las pe cineva să afle cât de rău îmi e.',
        'În pauze mă prefac ocupat ca să nu se vadă că nu am cu cine sta.',
    ],
    'uk': [
        'Коли на мене кричать, у горлі стає клубок і важко щось сказати.',
        'Від самої думки, що доведеться знову туди йти, усе всередині стискається.',
        'Що ближче я до дому, то сильніше напружуюся, ніби не можу розслабити тіло.',
        'Після тих повідомлень серце калатає, і я довго не можу заспокоїтися.',
        'Після сварок весь час дивлюся на двері й не можу нормально відпочити.',
        'Коли говорять зі мною таким різким голосом, я завмираю й не можу відповісти.',
        'Постійно озираюся, відколи та людина почала ходити за мною.',
        'Коли поруч знову сваряться, ховаюся під ковдру й стискаю кулаки.',
        'Перед тією компанією, яка мене принижує, ноги стають наче ватяні.',
        'Кажу всім, що все гаразд, а наодинці не можу стримати сліз.',
        'Набираю повідомлення й знову зупиняюся: важко розповісти, наскільки мені кепсько.',
        'На перерві вдаю, що маю купу справ, аби ніхто не помітив, що мені немає з ким побути.',
    ],
}
ORDINARY = {
    'ro': [
        'La biologie fac un grafic despre pulsul după alergare și pulsul în repaus.',
        'Pentru lecția de muzică imit ritmul bătăilor inimii cu o tobă mică.',
        'Desenez mușchii corpului pentru afișul de la școală și aleg culorile.',
        'Caut o explicație pentru copii despre cum ajunge sunetul de la ureche la creier.',
        'La jocul de mimă exersez o expresie de surpriză, cu ochii mari.',
        'Am de completat un tabel școlar care compară odihna cu mișcarea.',
        'Pentru un referat de biologie caut o explicație despre cum se formează lacrimile.',
        'Încerc să desenez cutele unei pături și vreau un sfat despre umbre.',
        'La literatura română discutăm expresia «nod în gât» și vreau să-i înțeleg sensul.',
        'La tema de română aleg verbe care descriu mersul și alergarea.',
        'Fac un desen despre stomac pentru ora de anatomie și caut etichetele potrivite.',
        'Pentru dicționarul clasei caut un sinonim pentru cuvântul «încordat».',
    ],
    'uk': [
        'На біологію готую графік пульсу після бігу та у стані спокою.',
        'Для уроку музики відбиваю ритм серцебиття на маленькому барабані.',
        'Малюю м’язи людини для шкільного плаката й добираю кольори.',
        'Шукаю пояснення для дітей, як звук потрапляє від вуха до мозку.',
        'Для гри в пантоміму треную здивований вираз обличчя з широко розплющеними очима.',
        'Маю заповнити навчальну таблицю з порівнянням відпочинку та руху.',
        'Для доповіді з біології шукаю пояснення, як утворюються сльози.',
        'Малюю складки на ковдрі й хочу пораду щодо тіней.',
        'На літературі розбираємо вислів «клубок у горлі», і я хочу зрозуміти його значення.',
        'Для домашньої роботи добираю дієслова про ходьбу та біг.',
        'Готую малюнок шлунка для уроку анатомії й шукаю правильні підписи.',
        'Для словничка класу шукаю синонім до слова «напружений».',
    ],
}


def generate(language):
    if language not in CONTENT:
        raise ValueError('Unsupported language')
    rows = []

    def add(group, turns, scope, concern, affirmed, paired=False):
        fictional = scope == 'fiction' and not affirmed
        rows.append(dict(
            case_id=group+'-'+str(affirmed), language=language,
            source='indirect_distress_v9', partition='train', synthetic_only=True,
            independently_annotated=False, pair_group=group if paired else None,
            scope=scope, turns=[dict(role='minor', text=t) for t in turns],
            focus=len(turns)-1,
            attributes=dict(subject='FICTIONAL' if fictional else 'SELF', time='CURRENT',
                assertion='NEGATED' if scope == 'denial' and not affirmed else 'AFFIRMED',
                source='FICTION' if fictional else 'DIRECT', concern=str(concern), relation='NEW'),
            review=int(concern and affirmed)))

    for concern, contents in ((1, CONTENT[language]), (0, ORDINARY[language])):
        for i, content in enumerate(contents):
            add(f'indirect-{language}-{concern}-{i}-bare', [content], 'bare', concern, 1)
            for scope, frames in FRAMES[language].items():
                for j, forms in enumerate(frames):
                    for layout in (0, 1):
                        group = f'indirect-{language}-{concern}-{i}-{scope}-{j}-{layout}'
                        for affirmed, frame in enumerate(forms):
                            text = frame.format(x=content)
                            turns = text.split(': ', 1) if layout else [text]
                            add(group, turns, scope, concern, affirmed, paired=bool(concern))
    for i, content in enumerate(CONTENT[language]):
        for j, (reset, continuation) in enumerate(RESET[language]):
            ordinary = ORDINARY[language][(i+j) % len(ORDINARY[language])]
            group = f'indirect-{language}-reset-{i}-{j}'
            add(group, [content, reset.format(b=ordinary)], 'reset', 0, 0, paired=True)
            add(group, [content, continuation.format(x=content)], 'reset', 1, 1, paired=True)
    return rows
