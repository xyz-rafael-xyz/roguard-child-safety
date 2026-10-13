"""Third fresh synthetic test after consuming prior two tests for hard examples."""
from training.conversations_v6 import generate as earlier,FRAMES
BANK={
'ro':[
('Azi la cercul de robotică m-am simțit neînțeles. Mi-ar prinde bine să povestesc cuiva.','Azi la cercul de robotică vreau să aflu cum funcționează un senzor.'),
('Înainte să încep cursul de înot sunt neliniștit și aș avea nevoie de o discuție.','Înainte să încep cursul de înot vreau să aflu ce echipament îmi trebuie.'),
('Mă simt singur când ceilalți fac planuri fără mine. Aș vrea să mă asculte cineva.','Vreau să fac un plan pentru o machetă de carton.'),
('Nu știu cum să povestesc ce mă întristează în ultima vreme. Mă ajuți să vorbesc despre asta?','Nu știu cum să povestesc pe scurt finalul unei cărți.'),
('Aș vrea să spun că mă apasă ceva, dar îmi e greu să încep. Poți să mă asculți?','Aș vrea să încep un proiect despre sistemul solar.'),
('Mă încurcă emoțiile când sunt cu grupul nou. Putem sta puțin de vorbă despre ce simt?','Cu grupul nou vreau să organizăm o vânătoare de comori.')],
'uk':[
('Сьогодні на робототехніці відчув, що мене не розуміють. Хотілося б комусь розповісти.','Сьогодні на робототехніці хочу дізнатися, як працює датчик.'),
('Перед першим заняттям із плавання тривожно, і мені потрібна розмова.','Перед першим заняттям із плавання хочу дізнатися, що взяти із собою.'),
('Почуваюся самотньо, коли інші будують плани без мене. Хочу, щоб мене вислухали.','Хочу скласти план картонного макета.'),
('Не знаю, як розповісти про те, що останнім часом засмучує. Допоможеш почати розмову?','Не знаю, як коротко переказати кінцівку книжки.'),
('Хочу сказати, що мене щось турбує, але важко почати. Можеш вислухати?','Хочу почати проєкт про Сонячну систему.'),
('У новій групі хвилювання заважає спілкуватися. Можемо трохи поговорити про мої почуття?','З новою групою хочемо організувати пошук скарбів.')]
}
def generate(split):
    if split!='test':return earlier(split)
    rows=[]
    for lang,pairs in BANK.items():
        fiction,retract,personal=FRAMES[lang]
        for i,(pos,neg) in enumerate(pairs):
            for form,label,texts in [('direct',1,[pos]),('ordinary',0,[neg]),('fiction',0,[fiction,pos]),('retraction',1,[pos,retract]),('switch',1,[fiction,neg,personal,pos])]:
                rows.append(dict(id=f'v6c-{lang}-{i}-{form}',family=f'v6c-{lang}-{i}',language=lang,label=label,split=split,discourse=form,turns=[dict(role='minor',text=t) for t in texts]))
    return rows
