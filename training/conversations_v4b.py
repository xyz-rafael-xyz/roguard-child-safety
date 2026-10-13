"""Broader direct-message and contextual study after V4 source-shift diagnosis."""
import json
from pathlib import Path
from training.conversations_v4 import generate as original, render
from training.generate_messages_v3b import generate as legacy
ROOT=Path(__file__).resolve().parents[1]
FRAMES={
 'ro':{'train':['Ce spun mai jos este despre mine.','Într-un text inventat apare următoarea replică. Nu este despre mine.','Nu mai vreau să detaliez acum.'],
       'dev':['Vreau să îți descriu ce simt eu.','Comentez cuvintele unui personaj dintr-un dialog imaginar.','Poate lăsăm discuția pe altă dată.'],
       'test':['Așa încerc să exprim propria mea stare.','Discut numai formularea dintr-o scenă inventată pentru un exercițiu.','Mai bine nu continuăm cu subiectul ăsta.']},
 'uk':{'train':['Наступними словами описую себе.','У вигаданому тексті є така репліка. Вона не про мене.','Зараз не хочу розповідати докладніше.'],
       'dev':['Хочу описати власні почуття.','Коментую слова персонажа з уявного діалогу.','Може, залишимо розмову на інший раз.'],
       'test':['Так я намагаюся висловити власний стан.','Обговорюю лише формулювання у вигаданій сцені для вправи.','Краще не продовжувати цю тему.']}}

def generate(split):
 bank=json.loads((ROOT/'data/synthetic/v4b_native_pairs.json').read_text())
 rows=original('train') if split=='train' else []
 def append(lang,family,discourse,label,texts,repeat=1):
  for exposure in range(repeat):
   rows.append({'id':f'{family}-{discourse}-{exposure}','family':family,'setting_group':family,
                'need_group':family,'language':lang,'split':split,'label':label,'discourse':discourse,
                'source':'project-authored-v4b','exposure':exposure,
                'turns':[{'role':'minor','text':text} for text in texts]})
 for lang in ('ro','uk'):
  f=FRAMES[lang][split]
  for index,(positive,negative) in enumerate(bank[lang][split]):
   family=f'v4b-{lang}-{split}-{index}';repeat=8 if split=='train' else 1
   for discourse,label,texts in [('direct',1,[positive]),('ordinary',0,[negative]),('personal',1,[f[0],positive]),('fiction',0,[f[1],positive]),('retraction',1,[positive,f[2]])]:
    append(lang,family,discourse,label,texts,repeat)
 if split=='train':
  # Previous tests are explicitly consumed training material. Only canonical
  # old stems are used; repeated exposure is declared, not new authorship.
  old=[r for r in legacy('train') if r['pair_id'].endswith('-0-0')]+legacy('dev')+legacy('test')
  for row in old:
   lang=row['language'];f=FRAMES[lang]['train'];family='consumed-'+row['pair_id']
   append(lang,family,'legacy_direct',row['label'],[row['text']],8)
   if row['label']:
    append(lang,family,'legacy_fiction',0,[f[1],row['text']],8)
    append(lang,family,'legacy_retraction',1,[row['text'],f[2]],8)
 return rows
