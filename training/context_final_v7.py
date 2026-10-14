"""Third frozen fictional holdout: new wording after the joint development run."""
from training.context_v7 import LABELS
DATA={
'ro':{
'concerns':['Mă simt pierdut când încerc să vorbesc despre problemele mele.','Mi-aș dori să am pe cineva alături când îmi e greu.','Îmi este dificil să le explic celorlalți de ce sunt trist.','Aș vrea ajutor ca să pot spune ce mă frământă.','Mă cuprinde neliniștea când mă gândesc la discuția de mâine.','Mi-e greu să fac față tuturor acestor emoții.'],
'frames':['Tocmai îți spun cum este pentru mine în acest moment.','Îți relatez ceva ce simțeam când eram mai mic; nu este o situație actuală.','Aceste vorbe îi aparțin unui amic care mi-a descris starea lui.','Am inventat replica pentru o scenetă, fără legătură cu viața mea.','Îmi îndrept mesajul de dinainte: vorbeam în rolul unui personaj, nu în nume propriu.','Nu sunt hotărât dacă aceste cuvinte mă descriu cu adevărat.','Întreb doar cum se face această activitate.'],
'ordinary':['Cum pot îndoi o cutie din carton?','Ce trebuie să fac pentru a umfla o minge?','Cum aranjez piesele de domino?','Care e diferența dintre o stea și o planetă?','Cum desenez o bicicletă pas cu pas?','Ce pot folosi ca semn de carte?']},
'uk':{
'concerns':['Я гублюся, коли намагаюся говорити про свої проблеми.','Хотілося б, щоб хтось був поруч, коли мені важко.','Мені складно пояснити іншим, чому я сумую.','Хочу допомоги, щоб розповісти про те, що мене непокоїть.','Відчуваю тривогу, коли думаю про завтрашню розмову.','Мені важко впоратися з усіма цими емоціями.'],
'frames':['Щойно розповідаю тобі, як мені саме цієї миті.','Ділюся тим, що відчував, коли був молодшим; це не теперішня ситуація.','Ці слова належать приятелеві, який описав мені свій стан.','Я вигадав репліку для сценки, вона не пов’язана з моїм життям.','Виправляю своє попереднє повідомлення: я говорив від імені персонажа, а не від себе.','Не визначився, чи справді ці слова описують мене.','Лише запитую, як виконати цю справу.'],
'ordinary':['Як скласти коробку з картону?','Що треба зробити, щоб накачати м’яч?','Як розкласти кісточки доміно?','Чим зірка відрізняється від планети?','Як намалювати велосипед крок за кроком?','Що можна використати як закладку?']}}
def generate():
    result=[]
    for lang,data in DATA.items():
        for n,concern in enumerate(data['concerns']):
            for k,frame in enumerate(LABELS):
                text=concern if k!=6 else data['ordinary'][n];context=data['frames'][k]
                for f,texts in enumerate(([context,text],[text+' '+context],[text,context])):
                    for surface in ('original','lower'):
                        result.append(dict(case_id=f'final3-{lang}-{n}-{k}-{f}-{surface}',language=lang,
                            turns=[dict(role='minor',text=t if surface=='original' else t.lower()) for t in texts],frame=frame,signal=int(k!=6),
                            family=f'final3-{lang}-{n}',pair_group=f'final3-{lang}-{n}-{f}-{surface}',fictional=True,surface=surface,source='authored_generator_v3'))
    return result
