"""Contrast emotional support with ordinary task help; keep test data separate."""
from training.context_repair_v71 import generate as previous
from training.context_v7 import WRAPPERS,HELD
PHRASES={
'ro':{
 'train':['Aș vrea să povestesc cuiva despre grijile care mă apasă.','O conversație m-ar ajuta să nu mai port singur toate problemele.','Mi-ar face bine să mă asculte cineva când sunt trist.','Caut sprijin pentru lucrurile care mă sperie.','Nu mă descurc cu tot ce simt și vreau să vorbesc cu cineva.','Aș avea nevoie de o discuție despre fricile mele.','Mi-e greu să păstrez în mine toate supărările.','Vreau să împărtășesc cuiva ce mă neliniștește.'],
 'dev':['Am multe temeri despre care aș vrea să discut.','M-ar ajuta să spun cuiva ce mă doare sufletește.'],
 'cal':['Îmi doresc să nu duc singur povara acestor griji.','Caut un om cu care să vorbesc despre starea mea.']},
'uk':{
 'train':['Хотілося б розповісти комусь про тривоги, які мене гнітять.','Розмова допомогла б мені не залишатися з усіма проблемами наодинці.','Мені було б легше, якби хтось вислухав мене, коли сумно.','Шукаю підтримки через те, що мене лякає.','Не справляюся зі своїми почуттями й хочу з кимось поговорити.','Мені потрібна розмова про мої страхи.','Мені важко тримати в собі всі образи.','Хочу поділитися з кимось тим, що мене непокоїть.'],
 'dev':['Маю багато страхів, про які хотілося б поговорити.','Мені допомогло б розповісти комусь про душевний біль.'],
 'cal':['Хочу не нести тягар цих переживань наодинці.','Шукаю людину, з якою можна поговорити про мій стан.']}}

def support(split):
 rows=[]
 for lang in ('ro','uk'):
  for i,text in enumerate(PHRASES[lang][split]):
   def add(texts,frame):
    rows.append(dict(case_id=f'support-{split}-{lang}-{len(rows)}',language=lang,turns=[dict(role='minor',text=t) for t in texts],frame=frame,signal=1,family=f'support-{split}-{lang}-{i}',pair_group=f'support-{split}-{lang}-{i}',fictional=True,surface='original',source='v71_support'))
   add([text],'SELF_CURRENT')
   for k,label in enumerate(('SELF_CURRENT','SELF_PAST','OTHER_PERSON','QUOTED','CORRECTION','UNCERTAIN')):
    contexts=WRAPPERS[lang][k] if split=='train' else [HELD[lang][split][k]]
    for context in contexts:
     for texts in ([context,text],[text+' '+context],[text,context]):add(texts,label)
 return rows

def generate(split):
 return previous(split)+support(split)*(4 if split=='train' else 1)
