"""Authored synthetic training situations with matched scope changes.

These are fictional, machine-authored language examples, not child records or
independently reviewed translations. Reserved evaluation inputs are not read.
"""
import unicodedata


CONTENT = {
    'ro': {
        1: [
            'În ultima vreme îmi vine să plâng chiar și când stau cu prietenii.',
            'Mă trezesc cu o grijă pe care nu reușesc s-o explic nimănui.',
            'De câteva zile nu mai am chef de lucrurile care îmi plăceau.',
            'Mi-e greu să cer ajutor fiindcă mă tem că voi fi luat în râs.',
            'La școală mă prefac că sunt bine, dar acasă plâng pe ascuns.',
            'Aș avea multe de spus, numai că mi se blochează cuvintele.',
            'Când încerc să povestesc ce mă doare, schimb repede subiectul.',
            'Mi-aș dori să mă asculte cineva fără să mă certe.',
            'Uneori spun că glumesc doar ca să nu se vadă cât mă doare.',
            'Nu știu cu cine să vorbesc despre lucrurile care mă sperie.',
            'În pauze stau singur pentru că ceilalți mă alungă din grup.',
            'Cineva îmi trimite mesaje răutăcioase și mi-e frică să deschid telefonul.',
            'Mi se strânge stomacul când aud că iar trebuie să merg acolo.',
            'Seara mă gândesc mereu la ce s-a întâmplat și nu pot adormi.',
            'Nu vreau să intru în detalii încă, dar am nevoie de sprijin.',
            'Am încercat să spun ce se întâmplă și mi s-a zis să tac.',
            'Mi-e teamă că dacă povestesc, cineva se va supăra rău pe mine.',
            'În fața celorlalți râd, dar înăuntru sunt foarte trist.',
            'Azi abia am reușit să mă ridic din pat de atâta tristețe.',
            'Nu mă simt în siguranță când persoana aceea se apropie.',
            'Mă simt vinovat pentru ceva ce altcineva mi-a făcut.',
            'Aș vrea să vorbesc cu un adult care să mă ia în serios.',
            'Tot șterg mesajul înainte să-l trimit fiindcă mi-e rușine.',
            'Nu mai pot ține toate grijile astea doar pentru mine.',
        ],
        0: [
            'Am desenat un pod și vreau să colorez apa de sub el.',
            'Îmi pregătesc ghiozdanul pentru excursia de mâine.',
            'Vreau să învăț cum se face un nod pentru brățara mea.',
            'Am găsit o carte despre planete la bibliotecă.',
            'Îmi place să joc șah cu bunicul după prânz.',
            'Încerc să aleg un titlu pentru povestea mea cu roboți.',
            'Mi-am udat planta și acum o pun la lumină.',
            'Aș vrea să fac o felicitare cu stele din hârtie.',
            'La matematică exersăm împărțirea și vreau un exemplu.',
            'Am făcut o fotografie cu norii de deasupra parcului.',
            'Vreau să aflu de ce unele frunze se îngălbenesc.',
            'Mâine aduc la clasă colecția mea de pietricele.',
        ],
    },
    'uk': {
        1: [
            'Останнім часом хочеться плакати навіть поруч із друзями.',
            'Прокидаюся з тривогою, яку ніяк не можу пояснити іншим.',
            'Уже кілька днів не тішить те, що раніше подобалося.',
            'Важко попросити допомоги, бо боюся, що з мене сміятимуться.',
            'У школі вдаю, що все гаразд, а вдома потай плачу.',
            'Хочу багато чого розповісти, але слова наче застрягають.',
            'Щойно починаю говорити про свій біль, одразу змінюю тему.',
            'Хочеться, щоб хтось вислухав мене без докорів.',
            'Іноді кажу, що жартую, аби ніхто не помітив, як мені боляче.',
            'Не знаю, з ким поговорити про те, що мене лякає.',
            'На перервах сиджу окремо, бо інші проганяють мене з компанії.',
            'Хтось надсилає образливі повідомлення, і страшно відкривати телефон.',
            'У мене стискається живіт, коли чую, що знову треба туди йти.',
            'Щовечора згадую те, що сталося, і ніяк не можу заснути.',
            'Поки не хочу розповідати подробиці, але потребую підтримки.',
            'Коли намагаюся розповісти, що відбувається, мені наказують мовчати.',
            'Боюся, що після моєї розповіді хтось дуже на мене розсердиться.',
            'Перед іншими сміюся, а насправді мені дуже сумно.',
            'Сьогодні через сильний смуток було важко навіть встати з ліжка.',
            'Не почуваюся в безпеці, коли та людина наближається.',
            'Відчуваю провину за те, що зі мною зробила інша людина.',
            'Хочу поговорити з дорослим, який поставиться до мене серйозно.',
            'Щоразу стираю повідомлення перед надсиланням, бо соромно.',
            'Уже не можу тримати всі ці переживання лише в собі.',
        ],
        0: [
            'Малюю міст і хочу розфарбувати воду під ним.',
            'Готую рюкзак до завтрашньої екскурсії.',
            'Хочу навчитися зав’язувати вузол для свого браслета.',
            'У бібліотеці є цікава книжка про планети.',
            'Люблю грати з дідусем у шахи після обіду.',
            'Добираю назву для своєї історії про роботів.',
            'Поливаю свою рослину й ставлю її ближче до світла.',
            'Хочу зробити листівку з паперовими зірками.',
            'На математиці вчимо ділення, і мені потрібен приклад.',
            'Фотографую хмари над парком.',
            'Хочу дізнатися, чому деяке листя жовтіє.',
            'Завтра принесу до класу свою колекцію камінців.',
        ],
    },
}


def generate():
    from training.operator_contrasts_v9 import WORDS
    rows = []
    for lang, groups in CONTENT.items():
        for signal, messages in groups.items():
            for family, message in enumerate(messages):
                variants = [('plain', 0, message)]
                for op in ('deny','affirm','self','other','fiction'):
                    variants.extend((op,i,f.format(text=message)) for i,f in enumerate(WORDS[lang][op]))
                for op,variant,text in variants:
                    attrs = dict(subject='OTHER' if op=='other' else 'FICTIONAL' if op=='fiction' else 'SELF',
                        source='REPORTED' if op=='other' else 'FICTION' if op=='fiction' else 'DIRECT',
                        assertion='NEGATED' if op=='deny' else 'AFFIRMED',relation='NEW',time='CURRENT',concern=str(signal))
                    forms = [text]
                    if lang=='ro':
                        forms.append(''.join(c for c in unicodedata.normalize('NFD',text) if not unicodedata.combining(c)))
                    for form in dict.fromkeys(forms):
                        turns = [{'role':'minor','text':form}]
                        if op!='plain':
                            turns.insert(0,{'role':'minor','text':WORDS[lang]['prior'][(family+variant)%6]})
                        rows.append(dict(case_id=f'natural-v9-train-{lang}-{len(rows)}',language=lang,
                            turns=turns,focus=len(turns)-1,attributes=attrs,source='natural_contrasts_v9',
                            family=f'natural-v9-{lang}-{signal}-{family}',fictional=True))
    return rows
