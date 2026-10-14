"""Additional scope-only negatives and correction/quotation contrasts."""
from training.context_v71 import generate as original
from training.context_v71 import TEXT
from training.context_v7 import CONCERNS

WRAPPERS={
'ro': {
 'train':[
  ('SELF_CURRENT','Vorbesc în nume propriu despre viața mea de acum.'),
  ('SELF_CURRENT','Îți spun ce se întâmplă cu mine în prezent.'),
  ('SELF_CURRENT','Aici descriu propria mea experiență.'),
  ('CORRECTION','Îmi corectez mesajul anterior: era vocea unui personaj, nu a mea.'),
  ('CORRECTION','Revin asupra afirmației: am redat vorbele unui erou fictiv.'),
  ('CORRECTION','Rectific ce am scris: nu era viața mea, ci o scenă inventată.'),
  ('CORRECTION','M-am exprimat greșit mai devreme; citam o replică.'),
  ('QUOTED','Aceste vorbe sunt rostite de un personaj din scenariul meu.'),
  ('QUOTED','Scriu un dialog imaginar pentru spectacol.'),
  ('ORDINARY','Vreau să aflu informații despre un subiect nou.')],
 'dev': [('SELF_CURRENT','Descriu ceea ce trăiesc eu personal astăzi.'),('CORRECTION','Reformulez cele de mai sus: personajul vorbea, nu eu.'),('QUOTED','Este replica unui actor într-o scenă fictivă.')],
 'cal': [('SELF_CURRENT','Mă refer la propriile întâmplări din prezent.'),('CORRECTION','Repar o confuzie din mesajul trecut: relatam un fragment fictiv.'),('QUOTED','Cuvintele aparțin unui rol din piesa scrisă de mine.')]
},
'uk': {
 'train':[
  ('SELF_CURRENT','Говорю від себе про своє теперішнє життя.'),
  ('SELF_CURRENT','Розкажу, що відбувається зі мною в цей час.'),
  ('SELF_CURRENT','Тут я описую власний досвід.'),
  ('CORRECTION','Виправляю попереднє повідомлення: це був голос героя, а не мій.'),
  ('CORRECTION','Повернуся до свого твердження: я переказував слова вигаданого героя.'),
  ('CORRECTION','Уточню те, що написав: це не моє життя, а вигадана сцена.'),
  ('CORRECTION','Раніше я висловився неточно: це була цитата.'),
  ('QUOTED','Ці слова промовляє персонаж мого сценарію.'),
  ('QUOTED','Пишу уявний діалог для вистави.'),
  ('ORDINARY','Хочу дізнатися більше про нову тему.')],
 'dev': [('SELF_CURRENT','Описую те, що особисто переживаю сьогодні.'),('CORRECTION','Переформулюю сказане вище: говорив персонаж, а не я.'),('QUOTED','Це репліка актора у вигаданій сцені.')],
 'cal': [('SELF_CURRENT','Маю на увазі події з власного життя в теперішній час.'),('CORRECTION','Виправлю непорозуміння з минулого повідомлення: я переказував вигаданий уривок.'),('QUOTED','Ці слова належать герою п’єси, яку я пишу.')]
}}

def repairs(split):
 rows=[]
 indices={'train':range(12),'dev':range(12,16),'cal':range(16,20)}[split]
 bi={'train':range(12),'dev':range(24,28),'cal':range(32,36)}[split]
 for lang in ('ro','uk'):
  def add(texts,frame,signal):
   rows.append(dict(case_id=f'repair-{split}-{lang}-{len(rows)}',language=lang,turns=[dict(role='minor',text=t) for t in texts],frame=frame,signal=signal,family=f'repair-{split}-{lang}',pair_group=f'repair-{split}-{lang}',fictional=True,surface='original',source='v71_repair'))
  for frame,wrapper in WRAPPERS[lang][split]:
   add([wrapper],frame,0)
   for text,signal in [(TEXT[lang][i],0) for i in bi]+[(CONCERNS[lang][i],1) for i in indices]:
    if frame=='ORDINARY' and signal:continue
    for texts in ([wrapper,text],[text+' '+wrapper],[text,wrapper]):add(texts,frame,signal)
 return rows

def generate(split):
 return original(split)+repairs(split)*(2 if split=='train' else 1)
