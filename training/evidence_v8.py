"""Fictional bilingual factorial scenarios; split before fitting any model.

Counts describe controlled combinations, not independently authored messages.
No authentic child messages are used. Wording has no independent fluency review.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import random
from pathlib import Path
from training.context_v7 import CONCERNS
from training.context_v71 import TEXT

SPLITS = {'train': (range(12), range(24), 20000),
          'dev': (range(12,16), range(24,32), 4000),
          'cal': (range(16,20), range(32,40), 4000),
          'test': (range(20,24), range(40,52), 8000)}
# Four separately authored context surfaces. All partitions share the ontology.
WORDS = {
 'ro': {
  'subject': [
   ['Vorbesc despre mine.', 'Redau ce spune un coleg.', 'Scriu replica unui personaj inventat.'],
   ['Asta mă privește pe mine.', 'Povestesc ce mi-a zis o prietenă.', 'Compun un dialog pentru un roman.'],
   ['Îți descriu propria mea experiență.', 'Citez o altă persoană.', 'Este vocea unui erou fictiv.'],
   ['Eu sunt persoana despre care este vorba.', 'Reproduc vorbele vecinului meu.', 'Fragmentul aparține unei povești imaginare.']],
  'time': [['Se întâmplă acum.', 'Era așa acum câțiva ani.', 'Mă gândesc la o situație posibilă.'],
           ['Asta e situația de azi.', 'Vorbesc despre o perioadă trecută.', 'Doar îmi imaginez că s-ar întâmpla.'],
           ['Așa stau lucrurile în prezent.', 'Descriu cum era odinioară.', 'Presupun că ar putea fi așa.'],
           ['Mă refer la zilele acestea.', 'E o amintire de demult.', 'Este numai o ipoteză despre viitor.']],
  'assertion': [['Asta este descrierea:', 'Nu este adevărată această descriere:', 'Nu știu dacă este adevărat:', 'Retrag afirmația de mai devreme:'],
                ['Așa se simte:', 'Această afirmație este falsă:', 'Poate că este așa:', 'Corectez ce am afirmat înainte:'],
                ['Aceasta este starea:', 'Neg această afirmație:', 'Încă nu sunt sigur de asta:', 'Anulez ce am spus anterior:'],
                ['Lucrurile stau astfel:', 'Resping descrierea aceasta:', 'Rămâne neclar dacă este așa:', 'Nu mai susțin declarația precedentă:']],
 },
 'uk': {
  'subject': [['Говорю про себе.', 'Передаю слова однокласника.', 'Пишу репліку вигаданого персонажа.'],
              ['Це стосується мене.', 'Розповідаю, що сказала подруга.', 'Складаю діалог для роману.'],
              ['Описую власний досвід.', 'Цитую іншу людину.', 'Це голос вигаданого героя.'],
              ['Йдеться саме про мене.', 'Відтворюю слова сусіда.', 'Уривок належить уявній історії.']],
  'time': [['Це відбувається зараз.', 'Так було кілька років тому.', 'Думаю про можливу ситуацію.'],
           ['Це сьогоднішня ситуація.', 'Говорю про минулий період.', 'Лише уявляю, що це могло б статися.'],
           ['Такі обставини тепер.', 'Описую, як було колись.', 'Припускаю, що так могло б бути.'],
           ['Маю на увазі ці дні.', 'Це давній спогад.', 'Це лише гіпотеза про майбутнє.']],
  'assertion': [['Ось опис:', 'Цей опис не відповідає дійсності:', 'Не знаю, чи це правда:', 'Відкликаю попереднє твердження:'],
                ['Ось як це відчувається:', 'Це твердження неправдиве:', 'Можливо, це так:', 'Виправляю те, що стверджував раніше:'],
                ['Ось цей стан:', 'Заперечую це твердження:', 'Ще не впевнений у цьому:', 'Скасовую сказане раніше:'],
                ['Обставини такі:', 'Відкидаю цей опис:', 'Залишається незрозумілим, чи це так:', 'Більше не підтримую попередню заяву:']],
 }}
SUBJECTS = ['SELF','OTHER','FICTIONAL']
TIMES = ['CURRENT','PAST','HYPOTHETICAL']
ASSERTIONS = ['AFFIRMED','NEGATED','UNCERTAIN','RETRACTED']

def unit(lang, split, content, subject, time, assertion, layout=0):
    k = list(SPLITS).index(split)
    phrases = [WORDS[lang]['subject'][k][subject], WORDS[lang]['time'][k][time],
               WORDS[lang]['assertion'][k][assertion] + ' «' + content + '»']
    return ' '.join(phrases[i] for i in ((0,1,2),(1,0,2),(2,0,1))[layout])

def generate(split):
    concerns, ordinary, count = SPLITS[split]
    result=[]
    for lang in ('ro','uk'):
        # A paired block changes one interpretation variable while content and
        # unrelated task remain identical. Pair blocks never straddle splits.
        blocks=[]
        for ci,bi,s,t,a,layout in itertools.product(concerns,ordinary,range(3),range(3),range(4),range(3)):
            for signal in (0,1):
                content=CONCERNS[lang][ci] if signal else TEXT[lang][bi]
                text=unit(lang,split,content,s,t,a,layout)
                attrs={'subject':SUBJECTS[s], 'time':TIMES[t], 'assertion':ASSERTIONS[a],
                       'source':'DIRECT' if s==0 else 'REPORTED' if s==1 else 'FICTION',
                       'relation':'RETRACTS' if a==3 else 'NEW', 'concern':str(signal)}
                # Previous unit makes retractions refer to an actual assertion.
                prior=unit(lang,split,content,s,t,0) if a==3 else TEXT[lang][bi]
                turns=[{'role':'minor','text':prior},{'role':'assistant','text':
                      'Te ascult.' if lang=='ro' else 'Я слухаю.'}, {'role':'minor','text':text}]
                key=f'{split}-{lang}-{ci}-{bi}-{s}-{t}-{a}-{signal}-{layout}'
                blocks.append({'case_id':key,'language':lang,'turns':turns,'focus':2,
                    'attributes':attrs,'family':f'{lang}-concern-{ci}',
                    'ordinary_family':f'{lang}-ordinary-{bi}',
                    'pair_group':f'{split}-{lang}-{ci}-{bi}-{s}-{t}-{signal}',
                    'fictional':True,'source':'controlled_composition_v8'})
        random.Random(800+list(SPLITS).index(split)).shuffle(blocks)
        # Deduplicate entire input: ordinary rows may repeat across concern IDs.
        seen=set();unique=[]
        for r in blocks:
            key=json.dumps(r['turns'],ensure_ascii=False)
            if key not in seen:seen.add(key);unique.append(r)
        if len(unique)<count//2:
            raise ValueError(f'{split}/{lang}: only {len(unique)} distinct scenarios')
        result.extend(unique[:count//2])
    return result

def write(root):
    root.mkdir(parents=True,exist_ok=True)
    manifest={'synthetic_only':True,'independent_language_review':False,
              'description':'Controlled compositional scenarios; counts are not independent semantic families.',
              'splits':{}}
    seen=set()
    for split in SPLITS:
        rows=generate(split);raw=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in rows).encode()
        for r in rows:
            key=json.dumps(r['turns'],ensure_ascii=False)
            if key in seen:raise ValueError('Input leakage across splits')
            seen.add(key)
        path=root/(split+'.jsonl.gz')
        if path.exists():raise ValueError('Refusing to overwrite frozen data')
        path.write_bytes(gzip.compress(raw,mtime=0))
        manifest['splits'][split]={'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
          'concern_families':len(set(r['family'] for r in rows)),
          'ordinary_families':len(set(r['ordinary_family'] for r in rows))}
    (root/'data-freeze.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))

def load(root,split):
    path=Path(root)/(split+'.jsonl.gz');manifest=json.loads((Path(root)/'data-freeze.json').read_text())
    if hashlib.sha256(path.read_bytes()).hexdigest()!=manifest['splits'][split]['sha256']:raise ValueError('Data hash mismatch')
    return [json.loads(line) for line in gzip.decompress(path.read_bytes()).decode().splitlines()]

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();write(a.output)
