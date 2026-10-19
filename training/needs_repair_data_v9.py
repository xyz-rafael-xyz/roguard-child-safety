"""Synthetic emotional-disconnection and unmet-needs contrasts with family holdouts."""
from training.distress_repair_data_v9 import generate as prior_generate

# Separate authored phrase families are held out for development stress testing.
TRAIN = {
    1: [
        'Відчуваю відчуження, ніби нікому немає діла до мене',
        'Усі наче далеко від мене, навіть коли сидять поруч',
        'Мені не вдається відчути близькість ні з ким, і це мучить',
        'Я замкнувся в собі та не можу повернутися до звичного спілкування',
        'Відчуваю себе чужою серед людей, з якими виросла',
        'Уже давно не відчуваю зв’язку з друзями й через це страждаю',
        'Мене наче відрізало від усіх, хоч я не хочу бути наодинці',
        'Мені здається, що я зовсім ізольований і ніхто не почує',
        'Удома немає продуктів, а дорослих не можу знайти',
        'Я не їла від учора, бо ніхто не залишив мені нічого',
        'Мені ніде переночувати сьогодні й немає до кого звернутися',
        'Мене залишили без нагляду на кілька днів, я не знаю, що робити',
        'Не можу дістати питну воду й поруч немає дорослих',
        'Я мерзну надворі, бо мені не дозволяють зайти додому',
        'Мені потрібні ліки, але ніхто не допомагає їх отримати',
        'Дорослі не повертаються, а я не можу сам приготувати собі їсти',
    ],
    0: [
        'Буває приємно провести вечір окремо від компанії',
        'Переїхала в інше місто й підтримую зв’язок із друзями телефоном',
        'Не почуваюся самотнім, просто люблю тихі заняття',
        'Ми з друзями маємо різні інтереси та добре ладнаємо',
        'Чекаю на вечерю, яку зараз готує тато',
        'Забула перекус, але можу поїсти в шкільній їдальні',
        'Повертаюся додому по теплу куртку',
        'Батьки домовилися, що сьогодні я ночую в бабусі',
        'Пляшка спорожніла, зараз наберу чистої води',
        'Мама допомогла мені записатися до лікаря',
        'Хочу приготувати простий сніданок разом із дорослими',
        'Питаю, які продукти потрібні для супу',
        'У квартирі прохолодно, тато вже ввімкнув опалення',
        'На годину лишаюся вдома, а тітка живе поруч і на зв’язку',
        'Вибираю книжку про дружбу для шкільного проєкту',
        'Планую зустріч із другом на вихідні',
    ],
}
DEV = {
    1: ['Наче втратив емоційний зв’язок з усіма, і мені важко',
        'Я серед людей, але почуваюся чужою та потребую підтримки',
        'Не маю чим повечеряти, а доглядати за мною нікому',
        'Уже добу чекаю дорослих і не знаю, де взяти їжу'],
    0: ['Сьогодні відпочиваю сама, так мені затишно',
        'Я не дуже товариський, але маю надійних друзів',
        'Зголодніла після уроків, удома на мене чекає обід',
        'Тато затримується на роботі, я вечеряю з дідусем'],
}


def generate(split='train'):
    phrases=TRAIN if split=='train' else DEV
    # The old generator provides fixed context/scope structures. Replace only
    # its independently authored content families, preserving semantic labels.
    rows=prior_generate();result=[]
    from training.distress_repair_data_v9 import CONTENT
    for row in rows:
        original=None
        for label,values in CONTENT.items():
            for index,text in enumerate(values):
                if any(text.casefold() in t['text'].casefold() for t in row['turns']):
                    original=(label,index,text);break
            if original:break
        if original is None:raise ValueError('Unknown contrast content')
        label,index,text=original
        if index>=len(phrases[label]):continue
        replacement=phrases[label][index]
        turns=[]
        for t in row['turns']:
            v=t['text'].replace(text,replacement).replace(text.lower(),replacement.lower())
            turns.append(dict(t,text=v))
        result.append(dict(row,case_id='needs85-'+split+'-'+str(len(result)),turns=turns,source='needs_repair_v9'))
    return result
