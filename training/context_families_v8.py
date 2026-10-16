"""Authored synthetic family holdouts with paired scope and formatting checks.

No real messages. Family isolation applies to this benchmark, not unknown model
pretraining or all historical RoGuard data. These labels are project-authored.
"""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unicodedata

# Split ownership is explicit and shared across both languages and every variant.
FAMILIES = [
 ('train', 'team_exclusion', 'De când echipa mă lasă pe dinafară, mă simt singur.', 'Відколи команда мене не приймає, я почуваюся самотньо.'),
 ('train', 'night_worry', 'Seara grijile nu mă lasă să adorm și aș vrea să vorbesc cu cineva.', 'Увечері через тривоги не можу заснути й хочу з кимось поговорити.'),
 ('train', 'public_mockery', 'Când râd ceilalți de mine în fața clasei, îmi vine să plâng.', 'Коли з мене сміються перед класом, мені хочеться плакати.'),
 ('train', 'home_arguments', 'Certurile de acasă mă sperie și am nevoie să mă asculte cineva.', 'Сварки вдома мене лякають, і мені потрібно, щоб хтось мене вислухав.'),
 ('train', 'friend_conflict', 'M-am certat cu prietenul meu și de atunci sunt foarte trist.', 'Ми посварилися з другом, і відтоді мені дуже сумно.'),
 ('train', 'performance_pressure', 'Simt că trebuie să iau numai note mari și nu mai fac față presiunii.', 'Відчуваю, що мушу отримувати лише високі оцінки, і вже не витримую цього тиску.'),
 ('train', 'unheard_at_home', 'Încerc să spun acasă ce mă doare, dar nimeni nu mă ascultă.', 'Намагаюся розповісти вдома, що мене засмучує, але ніхто не слухає.'),
 ('train', 'online_insults', 'Mesajele jignitoare din grup mă fac să mă simt fără valoare.', 'Образливі повідомлення в групі змушують мене почуватися нікчемно.'),
 ('dev', 'moving_school', 'După mutarea la altă școală mă simt pierdut și am nevoie de sprijin.', 'Після переходу до іншої школи я почуваюся розгублено й потребую підтримки.'),
 ('dev', 'bereavement', 'De când a murit bunicul, îmi este greu și vreau să vorbesc despre asta.', 'Відколи помер дідусь, мені важко й хочеться про це поговорити.'),
 ('dev', 'sports_failure', 'După ce am pierdut concursul, cred că nu sunt bun de nimic.', 'Після поразки на змаганнях мені здається, що я ні на що не здатний.'),
 ('dev', 'separation_worry', 'Mă apasă gândul că părinții se despart și nu știu cui să spun.', 'Мене гнітить думка, що батьки розлучаються, і я не знаю, кому про це сказати.'),
 ('test', 'sibling_illness', 'De când sora mea este în spital, mă simt speriat și aș vrea să mă asculte cineva.', 'Відколи моя сестра в лікарні, мені страшно й хочеться, щоб хтось мене вислухав.'),
 ('test', 'public_embarrassment', 'Mă tot gândesc la momentul jenant de ieri și nu mai vreau să ies din casă.', 'Постійно згадую вчорашню незручну ситуацію й більше не хочу виходити з дому.'),
 ('test', 'broken_confidence', 'Cineva a povestit secretul meu tuturor și simt că nu mai pot avea încredere în nimeni.', 'Хтось розповів усім мою таємницю, і тепер здається, що нікому не можна довіряти.'),
 ('test', 'caregiving_burden', 'Trebuie să am grijă mereu de cei mici și mă simt copleșit.', 'Мені постійно доводиться доглядати за молодшими, і я почуваюся виснажено.'),
]

MARKERS = {
 'ro': dict(fiction='Scriu o scenă inventată. Personajul spune: ', denial='Nu este adevărat în cazul meu: ',
            past='Așa mă simțeam anul trecut: ', other='Un coleg mi-a spus despre el: ',
            reset='Schimb subiectul. Cum se calculează aria unui pătrat?',
            ordinary='Cum se calculează aria unui pătrat?', personal='Acum vorbesc despre mine, în realitate: '),
 'uk': dict(fiction='Пишу вигадану сцену. Персонаж каже: ', denial='Це неправда про мене: ',
            past='Так я почувався торік: ', other='Однокласник розповів мені про себе: ',
            reset='Зміню тему. Як обчислити площу квадрата?',
            ordinary='Як обчислити площу квадрата?', personal='Тепер говорю про себе насправді: '),
}


def generate():
    splits = {s: [] for s in ('train', 'dev', 'test')}
    for split, family, ro, uk in FAMILIES:
        for lang, text in [('ro', ro), ('uk', uk)]:
            m = MARKERS[lang]
            scopes = {
              'personal': ([text], 'SELF', 'CURRENT', 'AFFIRMED', 'DIRECT', '1'),
              'fiction': ([m['fiction']+text], 'FICTIONAL', 'CURRENT', 'AFFIRMED', 'FICTION', '1'),
              'denial': ([m['denial']+text], 'SELF', 'CURRENT', 'NEGATED', 'DIRECT', '1'),
              'past': ([m['past']+text], 'SELF', 'PAST', 'AFFIRMED', 'DIRECT', '1'),
              'other': ([m['other']+text], 'OTHER', 'CURRENT', 'AFFIRMED', 'REPORTED', '1'),
              'topic_reset': ([text, m['reset']], 'SELF', 'CURRENT', 'AFFIRMED', 'DIRECT', '0'),
              'fiction_to_personal': ([m['fiction']+text, m['personal']+text], 'SELF', 'CURRENT', 'AFFIRMED', 'DIRECT', '1'),
            }
            for scope, (texts, subject, time, assertion, source, concern) in scopes.items():
                for variant in ('original', 'nfd', 'spaces'):
                    turns = [{'role': 'minor', 'text': t} for t in texts]
                    for turn in turns:
                        if variant == 'nfd': turn['text'] = unicodedata.normalize('NFD', turn['text'])
                        if variant == 'spaces': turn['text'] = '  '+turn['text'].replace(' ', '  ')+'  '
                    splits[split].append(dict(case_id=f'family-{lang}-{family}-{scope}-{variant}',
                        family=family, partition=split, pair_group=f'{lang}-{family}', invariant_group=f'{lang}-{family}-{scope}',
                        variant=variant, scope=scope, language=lang, fictional=True, source='authored_family_holdout_v8',
                        turns=turns, focus=len(turns)-1,
                        attributes=dict(subject=subject,time=time,assertion=assertion,source=source,concern=concern,relation='NEW')))
    return splits


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    manifest={'synthetic_only':True,'independently_reviewed':False,'family_isolation_scope':'within this benchmark only',
              'test_policy':'Freeze before predictions; no test-driven training or selection', 'splits':{}}
    for split,rows in generate().items():
        raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows)
        (a.output/(split+'.jsonl')).write_text(raw)
        manifest['splits'][split]=dict(rows=len(rows),families=sorted({r['family'] for r in rows}),sha256=hashlib.sha256(raw.encode()).hexdigest())
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

if __name__=='__main__': main()
