"""Synthetic contrasts: a personal framing phrase alone supplies no concern.

An unresolved pronoun can inherit a concern from context. The same pronoun
after an ordinary activity does not invent one. No real child messages used.
"""
PHRASES = {
    'ro': {
        'train': ['Despre mine vreau să vorbesc.', 'Încerc să lămuresc ce se întâmplă în cazul meu.',
            'Mă refer la experiența mea de până acum.', 'Asta are legătură cu viața mea.',
            'Vreau să povestesc ceva din perspectiva mea.', 'Încă mă gândesc cum să descriu asta.',
            'Povestesc despre o situație în care mă aflu.', 'Așa arată lucrurile pentru mine.',
            'Nu am terminat de explicat despre ce este vorba.', 'Încerc să-mi dau seama ce înseamnă asta pentru mine.',
            'Mă întorc la ceea ce povesteam despre mine.', 'Aș vrea să discut mai mult despre asta.'],
        'dev': ['Încerc să pun în ordine ce am de povestit despre mine.',
            'Vorbesc din punctul meu de vedere despre situație.',
            'Mai am câteva lucruri de adăugat despre asta.', 'Îmi caut cuvintele pentru a descrie experiența.'],
        'neutral_train': ['Tocmai am ales o carte despre dinozauri.', 'Învăț să tricotez un fular.',
            'Am aranjat piesele unui puzzle.', 'Am găsit o nouă melodie pentru pian.'],
        'neutral_dev': ['Am construit un pod din bețe pentru proiect.', 'Îmi aleg un caiet cu foi colorate.'],
        'concern_train': ['Mă simt amenințat de mesajele pe care le primesc.',
            'Mă apucă plânsul când mă gândesc la felul în care sunt respins.',
            'Mi-e frică să rămân singur cu persoana care mă intimidează.',
            'Am nevoie de sprijin pentru neliniștea care mă copleșește.'],
        'concern_dev': ['Sunt copleșit de teamă după întâmplarea de la școală.',
            'Nu mai pot ține doar pentru mine cât de mult mă doare excluderea.'],
        'ack': 'Poți continua mesajul.',
    },
    'uk': {
        'train': ['Хочу розповісти про себе.', 'Намагаюся з’ясувати, що відбувається в моєму випадку.',
            'Маю на увазі свій попередній досвід.', 'Це пов’язано з моїм життям.',
            'Хочу описати дещо зі свого погляду.', 'Ще думаю, як це описати.',
            'Розповідаю про ситуацію, у якій перебуваю.', 'Ось як це виглядає для мене.',
            'Ще не закінчив пояснювати, про що йдеться.', 'Намагаюся збагнути, що це означає для мене.',
            'Повертаюся до своєї розповіді про себе.', 'Хотілося б поговорити про це докладніше.'],
        'dev': ['Пробую впорядкувати те, що хочу про себе розповісти.',
            'Описую ситуацію зі своєї точки зору.',
            'Маю ще дещо додати до цієї розповіді.', 'Добираю слова, щоб описати власний досвід.'],
        'neutral_train': ['Щойно вибрав книжку про динозаврів.', 'Вчуся плести шарф.',
            'Розклав деталі пазла.', 'Знайшов нову мелодію для фортепіано.'],
        'neutral_dev': ['Збудував міст із паличок для проєкту.', 'Обираю зошит із кольоровими сторінками.'],
        'concern_train': ['Відчуваю загрозу через повідомлення, які отримую.',
            'Починаю плакати, коли думаю про те, як мене відштовхують.',
            'Боюся залишатися наодинці з людиною, яка мене залякує.',
            'Потребую підтримки через неспокій, що мене переповнює.'],
        'concern_dev': ['Мене переповнює страх після події у школі.',
            'Уже не можу тримати в собі, як боляче від того, що мене виключають із гурту.'],
        'ack': 'Можеш продовжити повідомлення.',
    },
}


def generate(language, partition):
    if language not in PHRASES or partition not in ('train', 'dev'):
        raise ValueError('Unknown referent population')
    data = PHRASES[language]
    rows = []
    for i, phrase in enumerate(data[partition]):
        for j, concern in enumerate(data['concern_'+partition]):
            neutral = data['neutral_'+partition][j]
            for form in (('original', 'lower') if partition == 'train' else ('original',)):
                family = f'referent-{language}-{partition}-{i}-{j}-{form}'
                # Six records: vague or explicit current text, each with no,
                # neutral, or concern history. The vague concern pair remains
                # positive; it must not receive the invariant-negative loss.
                for history, prior in enumerate((None, neutral, concern)):
                    for explicit in (0, 1):
                        # Use a distinct suffix to avoid repeating standalone
                        # vague statements across the j variants.
                        text = phrase + ' ' + (concern if explicit else neutral)
                        if history == 2 and not explicit:
                            text = phrase
                        turns = [] if prior is None else [dict(role='minor', text=prior), dict(role='assistant', text=data['ack'])]
                        turns.append(dict(role='minor', text=text))
                        if form == 'lower':
                            turns = [dict(t, text=t['text'].lower()) for t in turns]
                        positive = explicit == 1 or history == 2
                        row = dict(case_id=f'{family}-{history}-{explicit}', context_family=family,
                            pair_group=f'{family}-{history}', language=language, source='referent_grounding_v9',
                            partition=partition, turns=turns, focus=len(turns)-1,
                            attributes=dict(subject='SELF', time='CURRENT', source='DIRECT',
                                assertion='AFFIRMED', relation='NEW', concern=str(int(positive))),
                            review=int(positive), synthetic_only=True, independently_annotated=False)
                        rows.append(row)
        # Standalone unspecified framing is not a concrete concern. Keep these
        # separate from matched families because its history changes its label.
        for form in (('original', 'lower') if partition == 'train' else ('original',)):
            rows.append(dict(case_id=f'referent-{language}-{partition}-bare-{i}-{form}',
                language=language, source='referent_grounding_v9', partition=partition,
                turns=[dict(role='minor', text=phrase.lower() if form == 'lower' else phrase)], focus=0,
                attributes=dict(subject='SELF', time='CURRENT', source='DIRECT', assertion='AFFIRMED', relation='NEW', concern='0'),
                review=0, synthetic_only=True, independently_annotated=False))
    return rows
