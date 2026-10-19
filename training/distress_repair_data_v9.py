"""Authored Ukrainian distress/benign contrasts; no copied evaluation messages."""
import itertools

CONTENT = {
    1: [
        'Мені здається, що я зайвий у будь-якій компанії',
        'Щовечора плачу, бо ні з ким поговорити',
        'Навіть серед друзів мені дуже самотньо',
        'Давно нічого не тішить, навіть улюблені заняття',
        'Я весь час напружена і не можу заспокоїтися',
        'Мене не кличуть нікуди, і від цього дуже боляче',
        'Мені соромно розповісти, як важко мені останнім часом',
        'Я перестав спілкуватися з усіма, хоча хочу підтримки',
        'У школі мене постійно принижують і я боюся туди йти',
        'Мені погрожують, якщо я розкажу про це дорослим',
        'Після сварок удома я тремчу і не можу заснути',
        'Мені страшно залишатися поруч із цією людиною',
        'Я почуваюся нікому не потрібною вже багато тижнів',
        'Наче між мною та всіма іншими стіна, і мені від цього зле',
        'Хочу попросити допомоги, але боюся, що з мене сміятимуться',
        'Я не справляюся з цими переживаннями самостійно',
    ],
    0: [
        'Люблю читати наодинці й добре почуваюся',
        'Сьогодні хочу побути без компанії, щоб відпочити',
        'Мої друзі грають у футбол, а я вибираю малювання',
        'Не йду на вечірку, бо готую подарунок сестрі',
        'Мені зручно виконувати домашнє завдання в тиші',
        'Трохи хвилююся перед виступом, але хочу спробувати',
        'Я новенька в гуртку й уже познайомилася з двома дітьми',
        'Попросив друзів не писати, поки я роблю уроки',
        'Сумую за табором, але радий знову бачити родину',
        'Хочу навчитися спокійно висловлювати незгоду',
        'Ми посперечалися про правила гри й потім домовилися',
        'Втомилася після тренування та збираюся відпочити',
        'Мені подобається мати кілька близьких друзів',
        'Не люблю гучних місць, тому обираю тихе кафе',
        'Мій настрій покращився після прогулянки з другом',
        'Я попросила допомогти з математикою й отримала пояснення',
    ],
}


def generate():
    rows=[]
    for concern,index,subject,when,assertion,layout in itertools.product(
            (0,1),range(16),('SELF','OTHER','FICTIONAL'),('CURRENT','PAST'),('AFFIRMED','NEGATED'),range(2)):
        text=CONTENT[concern][index]+'.'
        scope={'SELF':'Це про мене.', 'OTHER':'Це розповідь мого друга його словами.',
               'FICTIONAL':'Це слова вигаданого персонажа мого оповідання.'}[subject]
        time={'CURRENT':'Описую те, що відбувається зараз.', 'PAST':'Описую те, що було раніше.'}[when]
        claim={'AFFIRMED':'Підтверджую цей опис.', 'NEGATED':'Заперечую цей опис: насправді все було інакше.'}[assertion]
        turns=([dict(role='minor',text=' '.join((text,scope,time,claim)))] if layout==0 else
               [dict(role='minor',text=text),dict(role='assistant',text='Я слухаю.'),dict(role='minor',text=' '.join((scope,time,claim)))])
        rows.append(dict(case_id='distress83-uk-'+str(len(rows)),language='uk',source='distress_repair_v9',
            synthetic=True,turns=turns,focus=len(turns)-1,attributes=dict(subject=subject,time=when,assertion=assertion,
                source={'SELF':'DIRECT','OTHER':'REPORTED','FICTIONAL':'FICTION'}[subject],relation='NEW',concern=str(concern))))
    # Plain messages matter: avoid teaching a requirement for formal qualifiers.
    for concern,phrases in CONTENT.items():
        for phrase in phrases:
            for text in (phrase+'.', phrase.lower(), 'Не знаю, як це пояснити. '+phrase+'.'):
                rows.append(dict(case_id='distress83-uk-'+str(len(rows)),language='uk',source='distress_repair_v9',synthetic=True,
                    turns=[dict(role='minor',text=text)],focus=0,attributes=dict(subject='SELF',time='CURRENT',assertion='AFFIRMED',source='DIRECT',relation='NEW',concern=str(concern))))
    return rows
