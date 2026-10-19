"""Authored access/knowledge negation contrasts with complete policy scope."""
import itertools
from training.bare_distress_data_v9 import CONTENT as BARE
from training.needs_repair_data_v9 import TRAIN as NEEDS

ACCESS = {
    1: [
        'Від учора я нічого не їв і не знаю, де знайти їжу',
        'Не можу знайти нікого, хто допоможе мені дістати вечерю',
        'Я не знаю, коли хтось прийде додому, і їсти зовсім нічого',
        'Дорослі зникли з учора, мені нема чого їсти',
        'Не виходить зв’язатися з батьками, вже другий день немає їжі',
        'Ніхто не пояснив, хто подбає про мене сьогодні',
        'Я не знаю, у кого попросити води, а пити дуже хочеться',
        'Не знаю, як зігрітися, мене не пускають у дім',
        'Не розумію, де мені ночувати, бо дорослі пішли',
        'Мені нема до кого звернутися, хоча дуже потрібна допомога',
        'Не вмію сказати, як самотньо мені стало',
        'Не знаю, кому розповісти про погрози',
        'Я відчуваю відстороненість навіть у колі друзів',
        'Мені важко через те, що я став відстороненим від близьких',
        'Я ніби відсторонений від життя класу',
        'Між нами ніби прірва, і я не відчуваю близькості',
    ],
    0: [
        'Не знаю, як спекти печиво; мама допоможе після обіду',
        'Не знаю, де лежить мій перекус; він точно в рюкзаку',
        'Не знаю, коли повернеться тато, але я вдома з бабусею',
        'Не вмію готувати суп, тому прошу дорослого показати',
        'Не розумію завдання з математики й питаю вчительку',
        'Не знаю, який напій обрати до вечері',
        'Не можу знайти ложку, візьму іншу зі шухляди',
        'Не знаю, яку книжку прочитати наступною',
        'Не знаю, як запросити друга на день народження',
        'Не можу вирішити, яку фарбу взяти для малюнка',
        'Не знаю, де поставити новий вазон',
        'Не знаю, як полагодити іграшку, дорослі обіцяли допомогти',
        'Сьогодні хочеться тиші, тому читаю сам і почуваюся добре',
        'Пишу відсторонений опис краєвиду для шкільного твору',
        'У шахах я спокійно спостерігаю за партією збоку',
        'Не знаю, що приготують на вечерю, але вдома є продукти',
    ],
}


def generate():
    rows = []
    for category, content in [('access', ACCESS), ('bare', BARE), ('needs', NEEDS)]:
        for concern, index, subject, when, assertion, layout in itertools.product(
                (0, 1), range(16), ('SELF', 'OTHER', 'FICTIONAL'),
                ('CURRENT', 'PAST', 'HYPOTHETICAL'),
                ('AFFIRMED', 'NEGATED', 'UNCERTAIN', 'RETRACTED'), range(2)):
            text = content[concern][index] + '.'
            scope = {'SELF': 'Розповідаю про власний досвід.',
                     'OTHER': 'Передаю слова знайомої дитини про її досвід.',
                     'FICTIONAL': 'Усе це відбувається лише в моїй вигаданій історії.'}[subject]
            time = {'CURRENT': 'Йдеться про теперішній час.',
                    'PAST': 'Це опис давнішніх подій.',
                    'HYPOTHETICAL': 'Уявляю можливу ситуацію, а не стверджую, що вона вже сталася.'}[when]
            claim = {'AFFIRMED': 'Цей опис підтверджую.',
                     'NEGATED': 'Ні, цей опис не відповідає тому, що відбувається.',
                     'UNCERTAIN': 'Не впевнений, чи цей опис правильний.',
                     'RETRACTED': 'Відкликаю попередні слова й забираю цю розповідь назад.'}[assertion]
            turns = ([dict(role='minor', text=' '.join((text, scope, time, claim)))] if layout == 0 else
                     [dict(role='minor', text=' '.join((text, scope, time))),
                      dict(role='assistant', text='Можеш уточнити, що маєш на увазі.'),
                      dict(role='minor', text=claim)])
            rows.append(dict(case_id='scope102-'+str(len(rows)), language='uk',
                source='scope_distillation_v9', synthetic=True, content_family=f'{category}-{concern}-{index}',
                turns=turns, focus=len(turns)-1,
                attributes=dict(subject=subject, time=when, assertion=assertion,
                    source={'SELF': 'DIRECT', 'OTHER': 'REPORTED', 'FICTIONAL': 'FICTION'}[subject],
                    relation='RETRACTS' if assertion == 'RETRACTED' else 'NEW', concern=str(concern))))
        for concern, phrases in content.items():
            for index, phrase in enumerate(phrases):
                for text in (phrase+'.', phrase.lower(), 'Не знаю, як про це сказати. '+phrase+'.'):
                    rows.append(dict(case_id='scope102-'+str(len(rows)), language='uk',
                        source='scope_distillation_v9', synthetic=True, content_family=f'{category}-{concern}-{index}',
                        turns=[dict(role='minor', text=text)], focus=0,
                        attributes=dict(subject='SELF', time='CURRENT', assertion='AFFIRMED',
                                        source='DIRECT', relation='NEW', concern=str(concern))))
    return rows


def anchored_loss(logits, labels, teacher, anchor_mask):
    """Distill only replay rows; new authored rows receive the labeled objective."""
    import torch
    import torch.nn.functional as functional
    primary = functional.cross_entropy(logits, labels)
    temperature = 2.
    divergence = functional.kl_div(functional.log_softmax(logits/temperature, dim=-1),
                                   teacher.detach(), reduction='none').sum(dim=-1)
    mask = anchor_mask.to(dtype=logits.dtype)
    anchored = (divergence*mask).sum()/mask.sum().clamp(min=1)
    return primary + .5*temperature*temperature*anchored
