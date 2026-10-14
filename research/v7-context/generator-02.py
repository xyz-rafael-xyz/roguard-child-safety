"""Single-author fictional context families; no real messages or hidden labels in input."""
import hashlib
import json
import unicodedata

LABELS = ['SELF_CURRENT','SELF_PAST','OTHER_PERSON','QUOTED','CORRECTION','UNCERTAIN','ORDINARY']
# Disjoint concern wording and wrappers by partition; translations are authored,
# not claims of independent native-speaker validation.
CONCERNS = {
'ro': [
'Mă simt singur.', 'Mi-e teamă să merg la școală.', 'Nimeni nu mă ascultă.', 'Am nevoie să vorbesc cu cineva.',
'Îmi este greu să cer ajutor.', 'Mă neliniștește ce se întâmplă.', 'Mă simt exclus din grup.', 'Nu știu cui să spun ce mă apasă.',
'Mi se pare că nu am pe nimeni.', 'Mă îngrijorează ziua de mâine.', 'Aș vrea să fiu ascultat.', 'Am o problemă și nu mă descurc singur.',
'Nu mă simt în largul meu la școală.', 'Mi-ar prinde bine să vorbesc cu un om de încredere.', 'Mă simt copleșit.', 'Nu găsesc curajul să spun ce simt.',
'Mă apasă ceva și nu știu cum să explic.', 'Mi-e greu să stau singur cu gândurile mele.', 'Mă simt lăsat pe dinafară.', 'Caut pe cineva care să mă înțeleagă.',
'Mi-e tot mai greu să povestesc ce mă supără.', 'Aș avea nevoie de sprijin din partea cuiva.', 'Mă simt izolat de ceilalți.', 'Nu știu la cine să apelez când sunt speriat.'
],
'uk': [
'Мені самотньо.', 'Мені страшно йти до школи.', 'Мене ніхто не слухає.', 'Мені потрібно з кимось поговорити.',
'Мені важко просити про допомогу.', 'Мене тривожить те, що відбувається.', 'Я почуваюся зайвим у компанії.', 'Не знаю, кому розповісти про те, що мене гнітить.',
'Здається, у мене нікого немає.', 'Мене турбує завтрашній день.', 'Хотілося б, щоб мене вислухали.', 'У мене проблема, і я не можу впоратися сам.',
'Мені некомфортно у школі.', 'Мені допомогла б розмова з людиною, якій я довіряю.', 'Усе це для мене занадто важко.', 'Не можу наважитися розповісти про свої почуття.',
'Мене щось гнітить, і я не знаю, як це пояснити.', 'Мені важко залишатися наодинці зі своїми думками.', 'Я почуваюся відстороненим від інших.', 'Шукаю когось, хто мене зрозуміє.',
'Мені дедалі важче розповідати про те, що мене засмучує.', 'Мені потрібна чиясь підтримка.', 'Я почуваюся ізольованим від інших.', 'Не знаю, до кого звернутися, коли мені страшно.'
]}
# Each row is [context before the focal turn, focal suffix].
WRAPPERS = {
'ro': [
['Vreau să spun ceva despre mine, acum.', 'Asta simt eu în prezent.'],
['Povestesc despre cum eram demult.', 'Așa mă simțeam atunci, nu descriu prezentul.'],
['Îți spun ce mi-a povestit un coleg.', 'Acestea sunt cuvintele lui, nu este vorba despre mine.'],
['Analizăm o replică dintr-o poveste inventată.', 'Este un citat din poveste pentru tema mea.'],
['Mai devreme am spus că este despre mine.', 'Corectez: nu era despre mine; repetam o replică.'],
['Nu știu dacă am înțeles bine ce simt.', 'Poate asta simt, dar nu sunt sigur.'],
['Lucrez la un exercițiu de matematică.', 'Vreau să adun două numere.']
],
'uk': [
['Хочу сказати дещо про себе зараз.', 'Це те, що я відчуваю тепер.'],
['Розповідаю про те, яким я був колись.', 'Так я почувався тоді, а не зараз.'],
['Переказую те, що розповів однокласник.', 'Це його слова, а не розповідь про мене.'],
['Розбираємо репліку з вигаданої історії.', 'Це цитата з оповідання для домашнього завдання.'],
['Раніше я сказав, що це про мене.', 'Виправляю: це було не про мене, я повторював репліку.'],
['Не знаю, чи правильно розумію свої почуття.', 'Можливо, я так почуваюся, але не впевнений.'],
['Працюю над математичною вправою.', 'Хочу додати два числа.']
]}
HELD = {
'ro': {
'dev':['Este experiența mea de azi.','Este o amintire de acum câțiva ani.','Redau mesajul unei prietene.','Textul aparține unui personaj fictiv.','Retrag ce am zis despre mine; citam pe altcineva.','Nu pot spune sigur dacă mă descrie.','Cum calculez suma numerelor?'],
'cal':['Vorbesc despre propria stare din acest moment.','Asta ține de trecutul meu.','Vorbesc în numele altui copil, folosind vorbele lui.','Fragmentul e dintr-un roman, nu din viața mea.','Îndrept afirmația precedentă: nu eu trec prin asta.','Încă încerc să înțeleg dacă asta mi se întâmplă.','Mă ajuți cu adunarea pentru școală?'],
 'test':['Eu trec prin asta în zilele acestea.','Descriu o perioadă încheiată din viața mea.','Citez felul în care se simte vecinul meu.','Exemplul este inventat pentru un dialog literar.','Am formulat greșit anterior; era o citare, nu experiența mea.','Nu îmi dau seama dacă e chiar sentimentul meu.','Care este rezultatul acestei adunări?']},
'uk': {
'dev':['Це мій сьогоднішній досвід.','Це спогад про події кілька років тому.','Переказую повідомлення подруги.','Текст належить вигаданому персонажу.','Відкликаю сказане про себе: я цитував іншу людину.','Не можу точно сказати, чи це описує мене.','Як обчислити суму чисел?'],
'cal':['Говорю про власний стан у цю мить.','Це стосується мого минулого.','Говорю від імені іншої дитини її словами.','Уривок узято з роману, а не з мого життя.','Уточнюю попередню заяву: не я через це проходжу.','Ще намагаюся зрозуміти, чи це відбувається зі мною.','Допоможеш із додаванням для школи?'],
 'test':['Я переживаю це останніми днями.','Описую період мого життя, який уже завершився.','Цитую те, як почувається мій сусід.','Приклад вигаданий для літературного діалогу.','Раніше я висловився неправильно: це була цитата, а не мій досвід.','Не розумію, чи це справді моє почуття.','Який результат цього додавання?']}}

def normalize(text):
    return unicodedata.normalize('NFC',text)

def render(turns):
    return '\n'.join(('FOCUS ' if i==len(turns)-1 else 'CONTEXT ')+t['role']+': '+normalize(t['text']) for i,t in enumerate(turns))

def generate(split):
    indices={'train':range(12),'dev':range(12,16),'cal':range(16,20),'test':range(20,24)}[split]
    rows=[]
    for lang in ('ro','uk'):
        for n in indices:
            for k,label in enumerate(LABELS):
                base=CONCERNS[lang][n] if k!=6 else ('Învăț adunarea.' if lang=='ro' else 'Вчуся додавати.')
                if split=='train':
                    context,suffix=WRAPPERS[lang][k]
                else:
                    context=HELD[lang][split][k];suffix=''
                # Context-only, within-turn and correction-in-follow-up forms.
                forms=[([context,base],1),([base+' '+context],0),([base,context],1)] if split!='train' else [([context,base],1),([base+' '+suffix],0),([base,suffix],1),([context,base+' '+suffix],1)]
                if split=='train':
                    from training.context_variants_v7 import VARIANTS
                    for variant in VARIANTS[lang][k]:
                        forms.extend([([variant,base],1),([base+' '+variant],0),([base,variant],1)])
                for f,(texts,focus) in enumerate(forms):
                    for surface in ('original','lower','plain'):
                        changed=[t.lower() if surface=='lower' else ''.join(c for c in unicodedata.normalize('NFD',t) if not unicodedata.combining(c)) if surface=='plain' and lang=='ro' else t for t in texts]
                        turns=[dict(role='minor',text=t) for t in changed]
                        rows.append(dict(case_id=f'{split}-{lang}-{n}-{k}-{f}-{surface}',family=f'{lang}-{n}',pair_group=f'{split}-{lang}-{n}-{f}-{surface}',language=lang,turns=turns,frame=label,signal=int(k!=6),fictional=True,source='authored_generator_v1',surface=surface))
    return rows

def digest(rows):return hashlib.sha256(json.dumps(rows,ensure_ascii=False,sort_keys=True).encode()).hexdigest()
if __name__=='__main__':
    from pathlib import Path
    root=Path('research/v7-context');root.mkdir(exist_ok=True)
    for split in ('train','dev','cal','test'):
        rows=generate(split)
        (root/(split+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
        print(split,len(rows),digest(rows))
