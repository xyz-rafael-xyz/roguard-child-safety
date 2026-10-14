"""Topic shifts and direct concerns, separate from inspected v7.1 test text."""
from training.context_support_v71 import generate as previous
from training.context_v71 import TEXT
from training.context_v7 import CONCERNS
PIVOTS={
'ro':{'train':['Las povestea deoparte; acum îți spun ce simt eu. ','Nu mai vorbesc despre personaj, ci despre propria mea viață. ','Aș vrea să schimb discuția și să îți spun ceva personal. '], 'dev':['Închei exercițiul; urmează o mărturisire despre mine. '], 'cal':['Dincolo de tema de mai sus, chiar eu trec prin asta. ']},
'uk':{'train':['Залишу історію осторонь; тепер скажу, що відчуваю я. ','Я вже не про персонажа, а про власне життя. ','Хочу перейти до іншої розмови й розповісти щось особисте. '], 'dev':['Завершу вправу; далі розповім дещо про себе. '], 'cal':['Окрім завдання вище, я справді сам через це проходжу. ']}}
DIRECT={
'ro':{
 'train':['Nu am ce să mănânc și nu este nimeni care să aibă grijă de mine.','Sunt singur de mult timp și nu pot ajunge la un adult.','Am rămas fără mâncare și nu știu de unde să iau.','Un om mai mare îmi cere să păstrez secretă întâlnirea noastră.','Cineva mă lovește și îmi spune să nu povestesc.','Un coleg mă amenință și nu știu ce să fac.','Mă gândesc să mă rănesc și vreau ajutor.','Mă urmărește cineva și sunt speriat.'],
 'dev':['Nu am mâncare de câteva zile și nimeni nu mă ajută.','O persoană mă obligă să fac lucruri care mă sperie.'],
 'cal':['Mi-e foame și sunt lăsat singur fără hrană.','Sunt amenințat și am nevoie de ajutor ca să fiu în siguranță.']},
'uk':{
 'train':['Мені нічого їсти, і немає нікого, хто б про мене подбав.','Я вже довго сам і не можу зв’язатися з дорослим.','У мене закінчилася їжа, і я не знаю, де її взяти.','Старша людина просить тримати нашу зустріч у таємниці.','Хтось мене б’є і каже нікому не розповідати.','Однокласник погрожує мені, і я не знаю, що робити.','Думаю про те, щоб поранити себе, і хочу допомоги.','Хтось переслідує мене, і я наляканий.'],
 'dev':['У мене вже кілька днів немає їжі, і ніхто не допомагає.','Хтось змушує мене робити речі, які мене лякають.'],
 'cal':['Я голодний і залишився сам без їжі.','Мені погрожують, і я потребую допомоги, щоб бути в безпеці.']}}
BENIGN={
'ro':{'train':['Mi-e foame, iar tata pregătește cina.','Sunt singur în camera mea și citesc liniștit.','Am rămas fără creioane; cumpăr altele mâine.','Monstrul din joc mă urmărește; cum câștig?'], 'dev':['Aștept să fie gata prânzul și între timp fac tema.','Personajul din joc a pierdut o viață.'], 'cal':['Mama mi-a lăsat o gustare pe masă.','Dragonul din poveste îl urmărește pe cavaler.']},
'uk':{'train':['Я голодний, а тато готує вечерю.','Я сам у своїй кімнаті й спокійно читаю.','У мене закінчилися олівці; завтра куплю інші.','Монстр у грі переслідує мене; як перемогти?'], 'dev':['Чекаю, поки приготується обід, і роблю домашнє завдання.','Персонаж у грі втратив одне життя.'], 'cal':['Мама залишила мені перекус на столі.','Дракон у казці переслідує лицаря.']}}

def transfer(split):
 rows=[]
 for lang in ('ro','uk'):
  def add(texts,frame,signal):
   rows.append(dict(case_id=f'transfer-{split}-{lang}-{len(rows)}',language=lang,turns=[dict(role='minor',text=t) for t in texts],frame=frame,signal=signal,family=f'transfer-{split}-{lang}',pair_group=f'transfer-{split}-{lang}',fictional=True,surface='original',source='v71_transfer'))
  indices={'train':range(12),'dev':range(12,16),'cal':range(16,20)}[split]
  topics=[TEXT[lang][i] for i in ({'train':[5,7,20,21],'dev':[27,31],'cal':[34,39]}[split])]
  for topic in topics:
   for pivot in PIVOTS[lang][split]:
    for text in [CONCERNS[lang][i] for i in indices]+DIRECT[lang][split]:add([topic,pivot+text],'SELF_CURRENT',1)
  for text in DIRECT[lang][split]:add([text],'SELF_CURRENT',1)
  for text in BENIGN[lang][split]:
   add([text],'ORDINARY',0)
   for pivot in PIVOTS[lang][split]:add([pivot+text],'SELF_CURRENT',0)
 return rows

def generate(split):return previous(split)+transfer(split)*(4 if split=='train' else 1)
