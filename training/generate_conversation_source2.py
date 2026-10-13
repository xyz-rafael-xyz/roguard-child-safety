"""Simpler second-source interface after the multi-pair generator failed schema."""
import argparse,json,hashlib
from pathlib import Path
from training.generate_conversation_challenge import MODEL,REVISION
SCENES={
 'ro':['un hobby nou','o zi obișnuită la școală','o activitate în grup','o grijă greu de explicat','începerea unei conversații','nevoia de a fi ascultat','o schimbare de rutină','o întâlnire cu prietenii'],
 'uk':['нове хобі','звичайний день у школі','групове заняття','хвилювання, яке важко пояснити','початок розмови','потреба бути вислуханим','зміна звичного розпорядку','зустріч із друзями']}

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,default=32);args=p.parse_args()
 if args.output.exists():raise ValueError('Preserve prior generations')
 from mlx_lm import load,generate
 from mlx_lm.sample_utils import make_sampler
 import mlx.core as mx
 model,tokenizer=load(args.model);mx.random.seed(20261014)
 rows=[];rejected=[];requests=[]
 for lang in ('ro','uk'):
  for index,topic in enumerate(SCENES[lang]):
   for label in (1,0):
    if len(requests)>=args.limit:break
    if lang=='ro':
     purpose=('Persoana vorbește despre sine și cere sprijin emoțional pentru o grijă ușoară.' if label else 'Persoana discută numai despre un personaj fictiv pentru o temă de scriere. Nu cere sprijin pentru sine.')
     prompt=f'Scrie în română un exemplu complet inventat despre {topic}. {purpose} Răspunde doar cu un obiect JSON cu două chei: "context" și "message". Ambele valori sunt mesaje scurte ale aceleiași persoane. Fără nume, adrese, violență, abuz sau conținut sexual. Folosește formulări simple și firești, maximum 20 de cuvinte pentru fiecare valoare.'
    else:
     purpose=('Людина говорить про себе й просить емоційної підтримки через легке хвилювання.' if label else 'Людина обговорює лише вигаданого персонажа для письмового завдання. Вона не просить підтримки для себе.')
     prompt=f'Напиши українською повністю вигаданий приклад на тему: {topic}. {purpose} Відповідай лише об’єктом JSON із двома ключами: "context" і "message". Обидва значення — короткі повідомлення тієї самої людини. Без імен, адрес, насильства, описів знущань чи сексуального змісту. Прості природні слова, до 20 слів у кожному значенні.'
    formatted=tokenizer.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,add_generation_prompt=True)
    raw=generate(model,tokenizer,prompt=formatted,max_tokens=600,sampler=make_sampler(temp=.6),verbose=False).strip()
    requests.append({'language':lang,'scene':index,'label':label,'sha256':hashlib.sha256(raw.encode()).hexdigest()})
    with args.output.with_suffix('.raw.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps({'language':lang,'label':label,'text':raw},ensure_ascii=False)+'\n')
    cleaned=raw
    if cleaned.startswith('```json') and cleaned.endswith('```'):cleaned=cleaned[7:-3].strip()
    elif cleaned.startswith('```') and cleaned.endswith('```'):cleaned=cleaned[3:-3].strip()
    try:
     obj=json.loads(cleaned)
     if not isinstance(obj,dict) or set(obj)!={'context','message'} or any(not isinstance(v,str) or not 5<=len(v)<=500 for v in obj.values()):raise ValueError()
     family=f'qwen2-{lang}-{index}'
     rows.append({'id':f'{family}-{label}','family':family,'setting_group':family,'need_group':family,'language':lang,'label':label,'discourse':'personal' if label else 'fiction','turns':[{'role':'minor','text':obj['context']},{'role':'minor','text':obj['message']}],'label_status':'generator_assigned_unvalidated'})
    except (ValueError,TypeError):rejected.append({'language':lang,'scene':index,'label':label})
    print(lang,index,label,'accepted',len(rows),'rejected',len(rejected),flush=True)
 with args.output.open('x',encoding='utf-8') as f:json.dump({'model':MODEL,'revision':REVISION,'seed':20261014,'rows':rows,'rejected':rejected,'requests':requests},f,ensure_ascii=False,indent=2)

if __name__=='__main__':main()
