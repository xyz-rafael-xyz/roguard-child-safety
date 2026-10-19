"""Synthetic scope transitions with matched histories and held-out wording.

These are project-authored policy examples, not real child messages or clinical
annotations. Explicit current attribution changes the label; irrelevant earlier
topics should not. Development bodies and framing are disjoint from training.
"""

HISTORY = {
    'ro': ['Inventăm o poveste despre un explorator care repară o barcă.',
           'Îmi aranjez cărțile pe raft după culoarea coperților.',
           'Mai devreme am vorbit despre o zi în care m-am simțit speriat.'],
    'uk': ['Вигадуємо історію про мандрівника, який ремонтує човен.',
           'Розставляю книжки на полиці за кольором обкладинок.',
           'Раніше ми говорили про день, коли мені було страшно.'],
}

FRAMES = {
    'ro': {
        'train': [
            ('Schimb subiectul și îți spun ceva personal: {x}', 'Schimb subiectul: scriu o poveste nouă, iar personajul spune «{x}».'),
            ('În mesajul acesta vorbesc despre mine, în viața mea de zi cu zi: {x}', 'În mesajul acesta discut numai o replică a unui personaj inventat: «{x}».'),
            ('Acum îți povestesc ce trăiesc eu, separat de discuția de până aici. {x}', 'Acum începem un exercițiu nou de teatru. Replica personajului este: «{x}».'),
            ('Pun deoparte subiectul anterior. Ce urmează este experiența mea: {x}', 'Pun deoparte subiectul anterior. Ce urmează este un citat din povestea inventată de mine: «{x}».'),
            ('Trec la ceva din viața mea și spun cu cuvintele mele: {x}', 'Trec la o scenă nouă de ficțiune; pentru rolul personajului am scris: «{x}».'),
            ('Nu mai continui exercițiul. Vreau să îți spun ce simt eu: {x}', 'Încep un exercițiu separat de scriere. Un personaj imaginar spune: «{x}».'),
        ],
        'dev': [
            ('Las tema în urmă; îți vorbesc sincer despre propria mea situație. {x}', 'Las tema în urmă și trec la o altă scenă imaginară, cu replica: «{x}».'),
            ('Aș vrea să discutăm acum despre mine ca persoană, fără vreun rol. {x}', 'Aș vrea să analizăm acum vocea unui narator fictiv care spune: «{x}».'),
        ],
    },
    'uk': {
        'train': [
            ('Зміню тему й розповім дещо особисте: {x}', 'Зміню тему: пишу нову історію, а її персонаж каже: «{x}».'),
            ('У цьому повідомленні я говорю про себе та своє повсякденне життя: {x}', 'У цьому повідомленні обговорюю лише репліку вигаданого персонажа: «{x}».'),
            ('Зараз розповідаю про власний досвід, окремо від попередньої розмови. {x}', 'Зараз починаємо нову театральну вправу. Репліка персонажа така: «{x}».'),
            ('Відкладемо попередню тему. Далі йдеться про мій досвід: {x}', 'Відкладемо попередню тему. Далі буде цитата з моєї вигаданої історії: «{x}».'),
            ('Перейду до того, що є в моєму житті, і скажу своїми словами: {x}', 'Перейду до нової вигаданої сцени; для ролі персонажа я написав таку репліку: «{x}».'),
            ('Я вже не продовжую вправу. Хочу розповісти, що відчуваю: {x}', 'Починаю окрему письмову вправу. Уявний персонаж каже: «{x}».'),
        ],
        'dev': [
            ('Залишу домашнє завдання позаду й чесно розповім про власну ситуацію. {x}', 'Залишу домашнє завдання позаду й перейду до іншої вигаданої сцени з реплікою: «{x}».'),
            ('Тепер хочу поговорити про себе як про людину, без жодної ролі. {x}', 'Тепер хочу проаналізувати голос вигаданого оповідача, який каже: «{x}».'),
        ],
    },
}

CONTENT = {
    'ro': {
        'train': {
            1: ['Îmi vine să plâng înainte să intru pe poarta școlii.',
                'Mă simt dat la o parte și nu știu cui să mă adresez.',
                'Mi-e frică să merg singur pe drumul acela după ce s-a întâmplat ieri.',
                'Aș avea nevoie de cineva care să mă asculte fără să râdă de mine.',
                'Nu mai găsesc liniște de când primesc mesajele acelea urâte.',
                'Mă ascund în camera mea când încep țipetele.',
                'Am rămas fără ceva de mâncare și nu e nimeni aici să mă ajute.',
                'Sunt copleșit și mi-e greu să explic de unde vine starea asta.',
                'O persoană mă tot urmărește și nu mă simt în siguranță.',
                'Mă simt foarte singur, chiar când sunt alți oameni lângă mine.',
                'Îmi tremură mâinile când trebuie să răspund grupului care mă jignește.',
                'Aș vrea ajutor ca să spun unui adult că trec prin ceva greu.'],
            0: ['Învăț să împăturesc o barcă din hârtie.', 'Vreau să aleg un titlu pentru desenul meu.',
                'Număr câte pagini are capitolul de geografie.', 'Caut o rimă pentru un cântec despre anotimpuri.',
                'Exersez tabla înmulțirii cu niște cartonașe.', 'Desenez traseul unui autobuz pe o hartă.',
                'Pregătesc o listă cu ingredientele unui chec.', 'Aleg culorile unui semn de carte.',
                'Citesc despre aripile fluturilor pentru școală.', 'Încerc să construiesc un pod din bețe.',
                'Caut un joc de cuvinte pentru colegii mei.', 'Pun etichete pe borcanele cu creioane.'],
        },
        'dev': {
            1: ['Mă apasă mult ce se întâmplă acasă și nu știu cu cine să vorbesc.',
                'Mi-e teamă să deschid telefonul din cauza mesajelor primite.',
                'Parcă nimeni nu mă observă, iar asta mă doare.',
                'Am nevoie de ajutor ca să ajung într-un loc în care să nu-mi fie frică.',
                'Mă simt atât de pierdut încât nici nu știu cum să încep explicația.',
                'Mi-e greu să suport singur grijile astea.'],
            0: ['Compar lungimea a două panglici pentru un proiect.',
                'Învăț numele instrumentelor dintr-o orchestră.',
                'Așez piesele unui puzzle după forma marginilor.',
                'Caut informații despre migrația păsărilor.',
                'Vreau să fac un calendar cu fotografii ale plantelor.',
                'Îmi pregătesc materialele pentru o lecție de desen.'],
        },
    },
    'uk': {
        'train': {
            1: ['Хочеться плакати ще до того, як зайду на шкільне подвір’я.',
                'Відчуваю, що мене відштовхують, і не знаю, до кого звернутися.',
                'Після вчорашнього боюся йти тією дорогою наодинці.',
                'Мені потрібен хтось, хто вислухає й не сміятиметься з мене.',
                'Не можу заспокоїтися, відколи отримую ті образливі повідомлення.',
                'Ховаюся у своїй кімнаті, коли починаються крики.',
                'У мене немає чого їсти, а поруч нікого, хто міг би допомогти.',
                'Мене все це перевантажує, і важко пояснити, звідки такий стан.',
                'Одна людина постійно ходить за мною, і я не почуваюся в безпеці.',
                'Мені дуже самотньо, навіть коли поруч є інші люди.',
                'Тремтять руки, коли треба відповісти тим, хто мене ображає.',
                'Хочу, щоб мені допомогли розповісти дорослому про те, як мені важко.'],
            0: ['Вчуся складати паперовий човник.', 'Хочу дібрати назву до свого малюнка.',
                'Рахую сторінки в розділі з географії.', 'Шукаю риму для пісні про пори року.',
                'Повторюю таблицю множення за картками.', 'Малюю автобусний маршрут на карті.',
                'Готую список складників для кексу.', 'Добираю кольори для книжкової закладки.',
                'Читаю про крила метеликів для уроку.', 'Пробую збудувати міст із паличок.',
                'Шукаю гру зі словами для однокласників.', 'Наклеюю підписи на баночки з олівцями.'],
        },
        'dev': {
            1: ['Те, що відбувається вдома, дуже тисне на мене, а поговорити ні з ким.',
                'Боюся відкривати телефон через повідомлення, які надходять.',
                'Здається, мене ніхто не помічає, і від цього боляче.',
                'Потрібна допомога, щоб дістатися туди, де не буде страшно.',
                'Почуваюся настільки розгублено, що не знаю, з чого почати пояснення.',
                'Мені важко залишатися з усіма цими тривогами наодинці.'],
            0: ['Порівнюю довжину двох стрічок для проєкту.',
                'Вчу назви інструментів в оркестрі.',
                'Сортую деталі пазла за формою країв.',
                'Шукаю відомості про переліт птахів.',
                'Хочу зробити календар із фотографіями рослин.',
                'Готую матеріали для уроку малювання.'],
        },
    },
}


def generate(language, partition):
    if language not in CONTENT or partition not in ('train', 'dev'):
        raise ValueError('Unsupported language or partition')
    rows = []
    for concern, contents in CONTENT[language][partition].items():
        for i, content in enumerate(contents):
            for j, frames in enumerate(FRAMES[language][partition]):
                family = f'transition-{partition}-{language}-{concern}-{i}-{j}'
                for history, prefix in enumerate(HISTORY[language]):
                    for attribution, frame in enumerate(frames):
                        real = attribution == 0
                        rows.append(dict(case_id=f'{family}-{history}-{attribution}', language=language,
                            source='context_transition_v9', partition=partition, synthetic_only=True,
                            independently_annotated=False, context_family=family,
                            pair_group=f'{family}-{history}', invariance_group=f'{family}-{attribution}',
                            turns=[dict(role='minor', text=prefix), dict(role='minor', text=frame.format(x=content))],
                            focus=1, attributes=dict(subject='SELF' if real else 'FICTIONAL', time='CURRENT',
                                assertion='AFFIRMED', source='DIRECT' if real else 'FICTION',
                                relation='NEW', concern=str(concern)), review=int(concern and real)))
    return rows
