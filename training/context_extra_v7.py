"""Training-only everyday negatives and transliteration surface variants."""
from copy import deepcopy
TRAIN_BENIGN={
'ro':['Cum fac un avion de hârtie?','Cum funcționează un magnet?','Care este capitala Franței?','De ce plouă?','Cum se formează curcubeul?','Ajută-mă să plantez o sămânță.','Cum se citește ceasul?','Explică regula de la fotbal.','Cum fac o omletă?','Vreau să desenez o casă.','Cum repar un creion mecanic?','Ce înseamnă acest cuvânt?'],
'uk':['Як зробити паперовий літачок?','Як працює магніт?','Яка столиця Франції?','Чому йде дощ?','Як утворюється веселка?','Допоможи посадити насінину.','Як визначати час за годинником?','Поясни правило у футболі.','Як приготувати омлет?','Хочу намалювати будинок.','Як полагодити механічний олівець?','Що означає це слово?']}
UK_MAP=dict(zip('абвгґдеєжзиіїйклмнопрстуфхцчшщьюя',['a','b','v','h','g','d','e','ye','zh','z','y','i','yi','y','k','l','m','n','o','p','r','s','t','u','f','kh','ts','ch','sh','shch','','yu','ya']))
def transliterate(text):return ''.join(UK_MAP.get(c,c) for c in text.lower())
def augment(rows):
    result=list(rows)
    for r in rows:
        if r['surface']!='original':continue
        if r['language']=='uk':
            v=deepcopy(r);v['case_id']+='-latin';v['surface']='transliterated'
            for t in v['turns']:t['text']=transliterate(t['text'])
            result.append(v)
    for lang in ('ro','uk'):
        for n,text in enumerate(TRAIN_BENIGN[lang]):
            result.append(dict(case_id=f'extra-{lang}-{n}',language=lang,turns=[dict(role='minor',text=text)],frame='ORDINARY',signal=0,family=f'extra-{lang}-{n}',pair_group=f'extra-{lang}-{n}',fictional=True,surface='original',source='authored_training_extra'))
    return result
