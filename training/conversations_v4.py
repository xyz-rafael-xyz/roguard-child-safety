"""Invented support-seeking scenarios; family and wording splits are explicit.

No row comes from a reported case. Settings x needs are combinations, not
independent authors. All dialogue frames within a split share provenance.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

# Native-language phrases, not translations of real messages. Kept low-detail.
SETTINGS = {
 'ro': {
  'train': ['la clubul de lectură', 'în pauza de prânz', 'la cursul de desen', 'după antrenament', 'în tabăra de vară', 'la repetiția corului', 'în autobuzul școlar', 'la atelierul de ceramică', 'în timpul excursiei', 'la cercul de șah', 'în echipa de proiect', 'în sala de sport', 'la biblioteca școlii', 'la cursul de dans', 'în grupul de colegi', 'la concursul de științe', 'la ora de muzică', 'la activitatea de voluntariat', 'la clubul de robotică', 'în curtea școlii', 'la cursul de înot', 'la întâlnirea de weekend', 'la cercul de fotografie', 'în grupa de teatru'],
  'dev': ['la atelierul de grădinărit', 'în noua clasă', 'la cercul de astronomie', 'în sala de lectură', 'la repetiția orchestrei', 'la întâlnirea de orientare', 'în echipa de dezbateri', 'la atelierul de origami'],
  'test': ['la clubul de drumeție', 'în grupa de limbi străine', 'la atelierul de film', 'la repetiția de dans popular', 'în echipa revistei școlare', 'la cercul de geografie', 'la atelierul de programare', 'în grupa de escaladă']},
 'uk': {
  'train': ['у книжковому клубі', 'під час обідньої перерви', 'на занятті з малювання', 'після тренування', 'у літньому таборі', 'на репетиції хору', 'у шкільному автобусі', 'на занятті з кераміки', 'під час екскурсії', 'у шаховому гуртку', 'у проєктній команді', 'у спортивній залі', 'у шкільній бібліотеці', 'на занятті з танців', 'у групі однокласників', 'на науковому конкурсі', 'на уроці музики', 'під час волонтерської зустрічі', 'у гуртку робототехніки', 'на шкільному подвір’ї', 'на занятті з плавання', 'на зустрічі у вихідні', 'у фотогуртку', 'у театральній групі'],
  'dev': ['на занятті із садівництва', 'у новому класі', 'в астрономічному гуртку', 'у читальній залі', 'на репетиції оркестру', 'на ознайомчій зустрічі', 'у команді дебатів', 'на занятті з оригамі'],
  'test': ['у туристичному клубі', 'у групі іноземних мов', 'на занятті з кіно', 'на репетиції народних танців', 'у команді шкільного журналу', 'у географічному гуртку', 'на занятті з програмування', 'у групі зі скелелазіння']}}
NEEDS = {
 'ro': {
  'train': ['îmi este greu să cer sprijin', 'mă simt singur și aș vrea să vorbesc cu cineva', 'nu prea știu cum să spun ce mă apasă', 'am nevoie de ajutor ca să explic ce simt', 'mă tem că nu voi fi luat în serios', 'aș vrea să vorbesc cu un adult în care am încredere', 'mi-ar prinde bine să mă asculte cineva', 'nu mă simt în largul meu și caut sprijin', 'mă frământă ceva și nu știu cui să mă adresez', 'nu găsesc curajul să cer ajutor', 'mă simt copleșit și am nevoie de o pauză', 'vreau să discut despre o îngrijorare'],
  'dev': ['mi-e greu să duc singur grijile astea', 'aș avea nevoie de cineva care să mă înțeleagă', 'mă tot gândesc cum să încep o discuție despre ce simt', 'nu știu cum să cer să fiu ascultat', 'am o neliniște despre care aș vrea să vorbesc', 'îmi doresc să găsesc o persoană de încredere'],
  'test': ['nu mai vreau să țin toate grijile doar pentru mine', 'caut pe cineva cu care să pot vorbi deschis', 'mă tot opresc când încerc să spun ce mă deranjează', 'aș avea nevoie de puțin sprijin ca să fac primul pas', 'îmi vine greu să pun în cuvinte ce mă neliniștește', 'aș vrea să nu trec singur prin starea asta']},
 'uk': {
  'train': ['мені важко попросити підтримки', 'мені самотньо і хочеться з кимось поговорити', 'не знаю, як сказати про те, що мене непокоїть', 'мені потрібна допомога, щоб пояснити свої почуття', 'боюся, що мене не сприймуть серйозно', 'хочу поговорити з дорослою людиною, якій довіряю', 'мені було б легше, якби хтось мене вислухав', 'мені некомфортно і потрібна підтримка', 'мене дещо турбує, а до кого звернутися — не знаю', 'не вистачає сміливості попросити допомоги', 'мені зараз забагато всього, потрібна перерва', 'хочу обговорити те, що викликає тривогу'],
  'dev': ['мені важко залишатися з цими переживаннями наодинці', 'хочеться знайти когось, хто мене зрозуміє', 'усе думаю, як почати розмову про свої почуття', 'не знаю, як попросити, щоб мене вислухали', 'є неспокій, про який хочеться поговорити', 'хочу знайти людину, якій можна довіритися'],
  'test': ['не хочу тримати всі переживання лише в собі', 'шукаю когось, із ким можна поговорити відверто', 'щоразу зупиняюся, коли пробую сказати, що мене бентежить', 'мені знадобилася б підтримка, щоб зробити перший крок', 'важко дібрати слова для того, що мене непокоїть', 'не хочеться переживати цей стан наодинці']}}
FRAMES = {
 'ro': {
  'train': ['Vorbesc despre mine și despre ce simt.', 'Scriu o poveste. Următoarea replică aparține unui personaj inventat.', 'Poți să îmi spui mai mult, doar dacă vrei?', 'Lasă, poate n-ar fi trebuit să spun asta.', 'Analizez o replică dintr-un exercițiu, nu povestesc despre mine.', 'Vreau să înțeleg sensul expresiei.', 'Mă simt bine. Întreb doar despre program și materiale.'],
  'dev': ['Este o situație prin care trec eu.', 'Pregătesc o scenă imaginară. Vorbește personajul, nu eu.', 'Ce ai vrea să înțeleagă persoana care te ascultă?', 'Nu mai contează, uită ce am zis.', 'Textul de mai jos este un citat din tema de literatură.', 'Explică-mi construcția acestei propoziții.', 'Totul e în regulă cu mine. Caut informații despre orar.'],
  'test': ['Îți descriu starea mea de acum, din viața mea.', 'Inventez dialogul pentru o piesă. Cuvintele următoare sunt ale unui rol fictiv.', 'Putem continua în ritmul tău.', 'Mai bine mă opresc aici. Nu vreau să mai discut.', 'Am de comentat următorul fragment fictiv la școală.', 'Mă interesează analiza gramaticală a fragmentului.', 'Sunt liniștit și nu caut sprijin personal. Am doar o întrebare organizatorică.']},
 'uk': {
  'train': ['Говорю про себе і свої почуття.', 'Пишу оповідання. Наступні слова належать вигаданому персонажу.', 'Можеш розповісти більше, лише якщо хочеш.', 'Та ні, мабуть, не варто було про це казати.', 'Розбираю репліку з вправи, а не розповідаю про себе.', 'Хочу зрозуміти значення цього вислову.', 'У мене все добре. Питаю лише про розклад і матеріали.'],
  'dev': ['Це ситуація, яку переживаю особисто.', 'Готую уявну сцену. Це слова персонажа, а не мої.', 'Що має зрозуміти людина, яка тебе слухає?', 'Уже неважливо, забудь мої слова.', 'Текст нижче — цитата із завдання з літератури.', 'Поясни будову цього речення.', 'Зі мною все гаразд. Шукаю інформацію про час занять.'],
  'test': ['Описую те, що відчуваю зараз у власному житті.', 'Вигадую діалог для п’єси. Далі говорить уявний герой.', 'Можемо продовжувати в зручному для тебе темпі.', 'Краще на цьому зупинюся. Більше не хочу це обговорювати.', 'Маю прокоментувати цей вигаданий уривок для шкільного завдання.', 'Мене цікавить граматичний розбір уривка.', 'Я почуваюся спокійно й особистої підтримки не шукаю. Є лише організаційне запитання.']}}


def render(turns):
    return '\n'.join(f"{turn['role']}: {turn['text']}" for turn in turns)


def generate(split, augment=True):
    if split not in ('train', 'dev', 'test'):
        raise ValueError('Unknown split')
    rows = []
    for language in ('ro', 'uk'):
        frames = FRAMES[language][split]
        for si, setting in enumerate(SETTINGS[language][split]):
            for ni, need in enumerate(NEEDS[language][split]):
                family = f'{language}-{split}-s{si}-n{ni}'
                sentence = f'{setting.capitalize()}, {need}.'
                minor = lambda text: {'role': 'minor', 'text': text}
                variants = [
                    ('personal', 1, [minor(frames[0]), minor(sentence)]),
                    ('fiction', 0, [minor(frames[1]), minor(sentence)]),
                    ('retraction', 1, [minor(sentence), {'role': 'assistant', 'text': frames[2]}, minor(frames[3])]),
                    ('quotation', 0, [minor(frames[4]), minor(sentence), minor(frames[5])]),
                    ('direct', 1, [minor(sentence)]),
                    ('ordinary', 0, [minor(f'{setting.capitalize()}. {frames[6]}')]),
                ]
                for discourse, label, turns in variants:
                    styles = range(3) if split == 'train' and augment else range(1)
                    for style in styles:
                        transformed = []
                        for turn in turns:
                            text = turn['text']
                            if style == 1:
                                text = text.lower().replace('.', '').replace(',', '')
                            if style == 2:
                                if language == 'ro':
                                    text = ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))
                                else:
                                    text = text.replace('і', 'i').replace('’', "'")
                                text = text.replace(' ', '  ')
                            transformed.append({**turn, 'text': text})
                        ident = f'{family}-{discourse}-{style}'
                        rows.append({'id': ident, 'family': family, 'setting_group': f'{language}-{split}-s{si}',
                                     'need_group': f'{language}-{split}-n{ni}', 'language': language,
                                     'split': split, 'label': label, 'discourse': discourse,
                                     'source': 'project-authored-v4-templates', 'style': style,
                                     'turns': transformed})
    # Ordinary messages repeat across need slots: keep a single copy per setting/style.
    unique = {}
    for row in rows:
        key = (row['language'], render(row['turns']))
        if key in unique and unique[key]['label'] != row['label']:
            raise ValueError('Contradictory duplicate')
        unique.setdefault(key, row)
    return list(unique.values())


def digest(rows):
    return hashlib.sha256(json.dumps(rows, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=['train','dev','test'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = generate(args.split)
    with args.output.open('x', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps({'split': args.split, 'rows': len(rows), 'sha256': digest(rows)}))

if __name__ == '__main__':
    main()
