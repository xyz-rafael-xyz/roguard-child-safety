"""Frozen factorial holdout: unseen content and frames, with benign controls."""
from training.context_v7 import LABELS
DATA={
'ro':{
'concerns':['Mi-ar prinde bine o discuție ca să nu țin toate grijile în mine.','Mă simt departe de ceilalți, deși suntem în aceeași clasă.','Nu reușesc să le spun celor din jur că mă simt rău sufletește.','Îmi doresc să găsesc o persoană care să mă sprijine.','Simt că grijile sunt prea multe pentru mine.','Aș vrea să mă ajute cineva să îmi pun gândurile în ordine.'],
'benign':['Îmi place să fac modele din lut.','Mă bucur când învăț despre insecte.','Știu să pregătesc masa pentru cină.','Aș vrea să cunosc mai multe povești despre călătorii.','Am reușit să cos un nasture.','Îmi place să inventez ghicitori.'],
'frames':['Vorbesc în nume propriu despre ce trăiesc acum.','Aduc vorba despre o experiență de-a mea de altădată, nu de acum.','Redau aici spusele altcuiva, care mi-a explicat propria experiență.','Rândurile acestea sunt ale unui personaj pe care l-am creat pentru o temă.','Rectific ce am scris mai sus: era vocea unui personaj, nu o relatare despre mine.','Nu știu încă dacă acesta este felul potrivit de a descrie ce simt.','Am doar o întrebare practică despre această activitate.']},
'uk':{
'concerns':['Мені допомогла б розмова, щоб не тримати всі переживання в собі.','Почуваюся далеко від інших, хоча ми в одному класі.','Не вдається розповісти людям довкола, що мені душевно зле.','Хочу знайти людину, яка мене підтримає.','Відчуваю, що турбот для мене забагато.','Хотілося б, щоб хтось допоміг мені впорядкувати думки.'],
'benign':['Мені подобається ліпити фігурки з глини.','Радію, коли дізнаюся про комах.','Умію накривати стіл до вечері.','Хотілося б знати більше історій про подорожі.','Мені вдалося пришити ґудзик.','Мені подобається вигадувати загадки.'],
'frames':['Говорю від власного імені про те, що переживаю зараз.','Згадую власний давній досвід, а не теперішній.','Передаю тут слова іншої людини, яка пояснила мені свій досвід.','Ці рядки належать персонажеві, якого я створив для завдання.','Уточнюю написане вище: це був голос персонажа, а не розповідь про мене.','Ще не знаю, чи саме так варто описати мої почуття.','Маю лише практичне запитання про цю справу.']}}
def generate():
    out=[]
    for lang,data in DATA.items():
        for n in range(6):
            for k,frame in enumerate(LABELS):
                for signal in ([0] if frame=='ORDINARY' else [0,1]):
                    text=data['concerns' if signal else 'benign'][n];context=data['frames'][k]
                    for f,texts in enumerate(([context,text],[text+' '+context],[text,context])):
                        for surface in ('original','lower'):
                            out.append(dict(case_id=f'factorial-{lang}-{n}-{k}-{signal}-{f}-{surface}',language=lang,turns=[dict(role='minor',text=t if surface=='original' else t.lower()) for t in texts],frame=frame,signal=signal,family=f'factorial-{lang}-{n}',pair_group=f'factorial-{lang}-{n}-{signal}-{f}-{surface}',fictional=True,surface=surface,source='authored_factorial_final'))
    return out
