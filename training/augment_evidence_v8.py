"""Development-motivated training-only paraphrases; no held-out content used."""
import gzip,json,random
from pathlib import Path
from training.evidence_v8 import load,CONCERNS,TEXT,SUBJECTS,TIMES,ASSERTIONS

PHRASES={
'ro':{
'subject':[
 ['Îți spun ce simt eu.','Aici este vorba de propria mea viață.','Eu sunt cel care spune asta.','Acestea sunt cuvintele mele despre mine.','Îmi descriu trăirile.','Mă descriu pe mine.'],
 ['Un alt copil mi-a spus asta.','O colegă vorbește despre ea însăși.','Citez ce zice un prieten despre el.','Sunt vorbele altei persoane.','Altcineva mi-a povestit asta.','Mesajul este de la un coleg.'],
 ['Am inventat asta pentru o poveste.','Replica este fictivă, pentru teatru.','Un personaj din scenariu spune asta.','Scriu ficțiune pentru școală.','Acesta este un dialog imaginar.','Citez un personaj dintr-o carte.']],
'time':[
 ['Se petrece în acest moment.','Vorbesc despre ce se întâmplă azi.','Așa este acum.','Este ceva din prezent.','Mă refer la această perioadă.','Chiar acum lucrurile sunt așa.'],
 ['Asta se petrecea cândva.','Situația a fost așa în trecut.','Îmi amintesc cum era mai demult.','Vorbesc despre o etapă deja încheiată.','Așa era cu mult timp în urmă.','Este o relatare din anii trecuți.'],
 ['Imaginează-ți că ar fi așa.','Este un scenariu ipotetic.','Mă întreb cum ar fi dacă s-ar întâmpla.','Să presupunem că ar fi așa.','Discutăm o posibilitate, nu un fapt.','E un exemplu despre ce s-ar putea întâmpla.']],
'assertion':[
 ['Descrierea este corectă:','Confirm că este așa:','Asta este ceea ce spun:','Afirmația pe care o fac este:','Da, aceasta este descrierea:','Susțin această afirmație:'],
 ['Nu confirm această afirmație:','Nu este așa cum scrie aici:','Nu se aplică următoarea descriere:','Este greșit să se spună asta:','Nu, acest lucru nu este adevărat:','Aceasta nu este situația:'],
 ['Nu pot confirma acest lucru:','Nu îmi dau seama dacă este așa:','Poate că da, poate că nu:','Nu pot spune cu certitudine:','Mă întreb dacă următoarele sunt adevărate:','Încerc să înțeleg dacă se aplică:'],
 ['Îmi retrag cuvintele anterioare:','Nu mai afirm ceea ce am spus:','Renunț la declarația de mai înainte:','Vreau să retractez ce am zis:','Îndrept ce am declarat mai devreme:','Mă răzgândesc în privința afirmației anterioare:']]},
'uk':{
'subject':[
 ['Розповідаю, що відчуваю я.','Тут ідеться про моє власне життя.','Це я кажу про себе.','Це мої слова про мене.','Описую свої переживання.','Описую самого себе.'],
 ['Це сказала мені інша дитина.','Однокласниця говорить про себе.','Цитую слова друга про нього самого.','Це слова іншої людини.','Хтось інший розповів мені це.','Це повідомлення від однокласника.'],
 ['Я вигадав це для оповідання.','Репліка вигадана для театру.','Це каже персонаж сценарію.','Пишу художній текст для школи.','Це уявний діалог.','Цитую персонажа з книжки.']],
'time':[
 ['Це трапляється в цю мить.','Говорю про те, що відбувається сьогодні.','Зараз усе саме так.','Це стосується теперішнього часу.','Маю на увазі нинішній період.','Просто зараз обставини такі.'],
 ['Це відбувалося колись.','Такою була ситуація в минулому.','Згадую, як було давно.','Говорю про етап, який уже закінчився.','Так було багато часу тому.','Це розповідь про минулі роки.'],
 ['Уявімо, що було б саме так.','Це гіпотетичний сценарій.','Думаю, як було б, якби це сталося.','Припустімо, що було б так.','Обговорюємо можливість, а не факт.','Це приклад того, що могло б статися.']],
'assertion':[
 ['Опис правильний:','Підтверджую, що це так:','Ось що я кажу:','Моє твердження таке:','Так, ось цей опис:','Підтримую це твердження:'],
 ['Не підтверджую це твердження:','Усе не так, як написано тут:','Цей опис не підходить:','Було б неправильно сказати таке:','Ні, це неправда:','Ситуація не така:'],
 ['Не можу цього підтвердити:','Не розумію, чи це так:','Можливо, так, а можливо, ні:','Не можу сказати напевно:','Міркую, чи правдиве таке твердження:','Намагаюся зрозуміти, чи це підходить:'],
 ['Відкликаю свої попередні слова:','Більше не стверджую те, що казав:','Відмовляюся від попередньої заяви:','Хочу відкликати сказане:','Виправляю свою попередню заяву:','Змінюю думку щодо попереднього твердження:']]}}

def main():
 root=Path('research/v8-evidence/data');rng=random.Random(820);out=[]
 for row in load(root,'train'):
  row=json.loads(json.dumps(row));attrs=row['attributes'];lang=row['language'];old=row['turns'][2]['text'];body=old[old.index('«')+1:old.rindex('»')]
  pieces=[rng.choice(PHRASES[lang]['subject'][SUBJECTS.index(attrs['subject'])]),rng.choice(PHRASES[lang]['time'][TIMES.index(attrs['time'])]),rng.choice(PHRASES[lang]['assertion'][ASSERTIONS.index(attrs['assertion'])])+' «'+body+'»']
  rng.shuffle(pieces);row['turns'][2]['text']=' '.join(pieces);row['case_id']+='-paraphrase';row['source']='training_only_authored_paraphrases_v8';out.append(row)
 path=root/'train-paraphrases.jsonl.gz'
 if path.exists():raise ValueError('Refusing overwrite')
 path.write_bytes(gzip.compress(('\n'.join(json.dumps(r,ensure_ascii=False) for r in out)+'\n').encode(),mtime=0))
 print('Additional training scenarios',len(out))
if __name__=='__main__':main()
