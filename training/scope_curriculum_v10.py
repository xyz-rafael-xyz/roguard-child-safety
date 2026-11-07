"""Developer-authored matched assertion training; no inference language rules."""
import argparse
from collections import Counter
import gzip
import itertools
import json
from pathlib import Path
import random

from roguard.trajectory_data import file_sha, prefix, read_jsonl, write_jsonl
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory
from training.evidence_v8 import CONCERNS, TEXT, SUBJECTS, TIMES, ASSERTIONS
from training.prepare_instruction_sft_v8 import target
from training.prepare_semantic_augmentation_v10 import key


FRAMES = {
    'ro': {
        'subject': [
            ['Cuvintele citate se referă la mine.', 'Eu sunt persoana descrisă în citat.', 'Vorbesc despre propriile mele trăiri.', 'Descrierea mă privește pe mine personal.'],
            ['Redau vorbele unei colege despre ea.', 'Citatul descrie situația unui prieten, nu a mea.', 'O altă persoană mi-a trimis aceste cuvinte despre ea.', 'Transmit relatarea unui coleg despre propria lui viață.'],
            ['Citatul aparține unui personaj inventat.', 'Aceasta este o replică din povestirea mea fictivă.', 'Descriu numai un personaj imaginar dintr-un scenariu.', 'Textul citat este ficțiune scrisă pentru o piesă.']],
        'time': [
            ['Descrierea privește momentul prezent.', 'Mă refer la situația de astăzi.', 'Perioada descrisă este cea de acum.', 'Este vorba despre ceea ce se petrece în prezent.'],
            ['Descrierea privește o perioadă încheiată din trecut.', 'Mă refer la o situație de acum câțiva ani.', 'Perioada descrisă s-a încheiat demult.', 'Povestesc despre ce se petrecea atunci, în trecut.'],
            ['Situația este doar o ipoteză pentru discuție.', 'Mă refer la un scenariu care ar putea exista.', 'Este un exercițiu ipotetic despre o posibilă situație.', 'Explorăm ce s-ar întâmpla într-o împrejurare imaginară.']],
        'assertion': [
            ['Susțin că descrierea este adevărată.', 'Confirm cele scrise în fragment.', 'Afirmația din citat este cea pe care o susțin.', 'Da, descrierea corespunde situației.', 'Declar că este adevărat ce spune citatul.', 'Îmi asum afirmația citată.', 'Consider corectă descrierea de mai jos.', 'Sunt sigur că descrierea se aplică.', 'Afirm fără îndoială cele descrise.', 'Ceea ce susțin este exact descrierea citată.', 'Mențin această afirmație.', 'Confirm în mod explicit conținutul descrierii.'],
            ['Resping descrierea; nu o consider adevărată.', 'Citatul este fals în privința situației descrise.', 'Dezmințesc afirmația din fragment.', 'Nu accept drept adevărat ce spune citatul.', 'Susțin că descrierea nu corespunde realității.', 'Contrazic afirmația citată.', 'Este neadevărat conținutul acestui fragment.', 'Exclud că descrierea ar fi adevărată.', 'Spun clar că descrierea este falsă.', 'Citatul nu descrie situația respectivă.', 'Nu este corect ceea ce se afirmă în fragment.', 'Refuz această descriere ca fiind neadevărată.'],
            ['S-ar putea ca descrierea să fie adevărată, dar nu sunt sigur.', 'Este posibil să fie așa; încă am îndoieli.', 'Nu exclud această posibilitate și nu o pot confirma.', 'Poate există ceva adevărat în descriere, însă nu știu sigur.', 'Îmi pare posibil ce spune citatul, fără să pot decide.', 'Există șansa să fie adevărat; rămân nesigur.', 'Bănuiesc că descrierea se aplică, dar mă pot înșela.', 'Oscilez între a crede și a respinge această descriere.', 'Nu am o concluzie despre adevărul afirmației.', 'Presupun că s-ar putea să fie adevărat, fără certitudine.', 'Mă îndoiesc și nu știu dacă să confirm sau să infirm.', 'Încă nu pot decide dacă afirmația este adevărată ori falsă.'],
            ['Retractez acum afirmația pe care am făcut-o anterior.', 'Îmi retrag explicit declarația din mesajul precedent.', 'Nu mai susțin citatul pe care l-am afirmat mai devreme.', 'Anulez propria mea afirmație anterioară.', 'Renunț acum la cele afirmate de mine înainte.', 'Retrag confirmarea pe care am dat-o în mesajul anterior.', 'Vreau să retrag ce am declarat în primul mesaj.', 'Îmi revoc afirmația precedentă despre această situație.', 'Acum retractez descrierea pe care o susținusem.', 'Declarația mea anterioară este retrasă.', 'Nu mai mențin afirmația făcută de mine anterior.', 'Îmi iau înapoi afirmația dinainte.']]},
    'uk': {
        'subject': [
            ['Цитовані слова стосуються мене.', 'Я є людиною, описаною в цитаті.', 'Говорю про власні переживання.', 'Цей опис стосується особисто мене.'],
            ['Передаю слова однокласниці про неї саму.', 'Цитата описує ситуацію друга, а не мою.', 'Інша людина надіслала мені ці слова про себе.', 'Переказую розповідь однокласника про його власне життя.'],
            ['Цитата належить вигаданому персонажу.', 'Це репліка з мого вигаданого оповідання.', 'Описую лише уявного персонажа сценарію.', 'Цитований текст є вигадкою для п’єси.']],
        'time': [
            ['Опис стосується теперішнього моменту.', 'Маю на увазі сьогоднішню ситуацію.', 'Описаний період триває зараз.', 'Ідеться про те, що відбувається в теперішньому часі.'],
            ['Опис стосується завершеного періоду в минулому.', 'Маю на увазі ситуацію кілька років тому.', 'Описаний період давно закінчився.', 'Розповідаю про те, що відбувалося тоді, у минулому.'],
            ['Ситуація є лише гіпотезою для обговорення.', 'Маю на увазі сценарій, який міг би існувати.', 'Це гіпотетична вправа про можливу ситуацію.', 'Розглядаємо, що сталося б за уявних обставин.']],
        'assertion': [
            ['Стверджую, що опис правдивий.', 'Підтверджую написане у фрагменті.', 'Підтримую саме твердження з цитати.', 'Так, опис відповідає ситуації.', 'Заявляю, що зміст цитати правдивий.', 'Беру на себе відповідальність за це твердження.', 'Вважаю наведений опис правильним.', 'Я впевнений, що цей опис підходить.', 'Без сумніву стверджую описане.', 'Підтримую саме те, що сказано в цитаті.', 'Залишаюся при цьому твердженні.', 'Прямо підтверджую зміст опису.'],
            ['Відкидаю опис і не вважаю його правдивим.', 'Цитата неправдива щодо описаної ситуації.', 'Спростовую твердження з фрагмента.', 'Не приймаю зміст цитати за правду.', 'Стверджую, що опис не відповідає дійсності.', 'Заперечую цитоване твердження.', 'Зміст цього фрагмента неправдивий.', 'Виключаю можливість, що опис правдивий.', 'Чітко кажу, що опис хибний.', 'Цитата не описує відповідну ситуацію.', 'Те, що стверджується у фрагменті, неправильне.', 'Відкидаю цей опис як неправдивий.'],
            ['Опис може бути правдивим, але я не впевнений.', 'Це можливо, однак у мене ще є сумніви.', 'Не виключаю такої можливості й не можу її підтвердити.', 'Можливо, в описі є правда, проте я не знаю напевно.', 'Зміст цитати здається можливим, але я не можу вирішити.', 'Є шанс, що це правда; я залишаюся невпевненим.', 'Припускаю, що опис підходить, але можу помилятися.', 'Вагаюся між тим, щоб повірити опису й відкинути його.', 'Не маю висновку щодо правдивості твердження.', 'Припускаю, що це могло б бути правдою, без упевненості.', 'Сумніваюся і не знаю, підтвердити чи спростувати це.', 'Поки не можу вирішити, правдиве це твердження чи хибне.'],
            ['Зараз відкликаю твердження, яке зробив раніше.', 'Прямо відкликаю заяву з попереднього повідомлення.', 'Більше не підтримую цитату, яку стверджував раніше.', 'Скасовую власне попереднє твердження.', 'Тепер відмовляюся від того, що стверджував раніше.', 'Відкликаю підтвердження з попереднього повідомлення.', 'Хочу відкликати те, що заявив у першому повідомленні.', 'Відкликаю попередню заяву про цю ситуацію.', 'Зараз відмовляюся від опису, який раніше підтримував.', 'Мою попередню заяву відкликано.', 'Більше не дотримуюся твердження, зробленого раніше.', 'Беру назад своє попереднє твердження.']]}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selector', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise ValueError('Refusing to overwrite a scope curriculum')
    selector = TrajectoryModel(a.selector); inputs = {}; old_train = set(); protected = set()
    for name in ('shared-tail-corpus-164', 'grounding-tail-corpus-174', 'referent-tail-corpus-176', 'reported-tail-corpus-195'):
        path = Path('research/v9-stabilization')/name/'inputs.jsonl.gz'; inputs[str(path)] = file_sha(path)
        for entry in map(json.loads, gzip.open(path, 'rt', encoding='utf-8')):
            row = entry['row']; payload = {'language': row['language'], 'turns': row['turns'][:row['focus']+1]}
            (old_train if entry['partition'] == 'train' else protected).add(key(payload))
    earlier = Path('research/v10-evidence/semantic-training-12/train.jsonl'); inputs[str(earlier)] = file_sha(earlier)
    old_train.update(key(r) for r in read_jsonl(earlier))
    paths = [Path('research/v10-evidence/data')/(name+'.jsonl') for name in ('dev', 'calibration', 'challenge', 'v9-regression')]
    paths.append(Path('research/v10-evidence/broad-v9-regression/cases.jsonl'))
    for path in paths:
        inputs[str(path)] = file_sha(path)
        for case in read_jsonl(path):
            for point in case['checkpoints']:
                payload = prefix(case, point['focus']); view, inference = prepare_memory(payload, selector, 1200, preserve_complete=True)
                if view['mandatory_overflow']:
                    raise ValueError('Unsupported protected inference prefix')
                protected.update((key(payload), key(inference)))
    rng = random.Random(1020); rows = []; excluded = Counter(); seen = set(old_train)
    for lang in ('ro', 'uk'):
        bank = FRAMES[lang]
        for s, t, signal, variant in itertools.product(range(3), range(3), range(2), range(24)):
            family = 'scope20-'+lang+'-'+str(s)+'-'+str(t)+'-'+str(signal)+'-'+str(variant)
            body = rng.choice(CONCERNS[lang][:12] if signal else TEXT[lang][:24])
            subject = bank['subject'][s][variant % 4]; temporal = bank['time'][t][(variant//4+t) % 4]
            quotation = '«'+body+'»'
            for assertion in range(4):
                wording = bank['assertion'][assertion][variant % 12]
                if variant < 12:
                    focal = ' '.join((subject, temporal, wording, quotation))
                else:
                    focal = ' '.join((quotation, subject, temporal, wording))
                prior = ' '.join((subject, temporal, bank['assertion'][0][variant % 12], quotation)) if assertion == 3 else TEXT[lang][variant % 24]
                attributes = {'subject': SUBJECTS[s], 'time': TIMES[t], 'assertion': ASSERTIONS[assertion],
                    'source': ('DIRECT', 'REPORTED', 'FICTION')[s], 'relation': 'RETRACTS' if assertion == 3 else 'NEW', 'concern': str(signal)}
                row = {'case_id': family+'-'+str(assertion), 'family': family, 'language': lang,
                    'turns': [{'role': 'minor', 'text': prior},
                              {'role': 'assistant', 'text': 'Am citit mesajul.' if lang == 'ro' else 'Я прочитав повідомлення.'},
                              {'role': 'minor', 'text': focal}], 'focus': 2, 'attributes': attributes,
                    'partition': 'train', 'source': 'scope_curriculum_v10', 'synthetic_only': True,
                    'independently_annotated': False}
                row['review'] = target(row); identity = key(row)
                if identity in protected:
                    excluded['protected_prefix'] += 1
                elif identity in seen:
                    excluded['previous_or_duplicate_training_prefix'] += 1
                else:
                    seen.add(identity); rows.append(row)
    a.output.mkdir(parents=True); write_jsonl(a.output/'train.jsonl', rows)
    report = {'format': 'roguard-scope-curriculum-1', 'source_sha256': file_sha(Path(__file__)),
        'amendment_sha256': file_sha('research/v10-evidence/protocol-amendment-17.json'), 'inputs': inputs,
        'selector_sha256': selector.identity, 'training_data_sha256': file_sha(a.output/'train.jsonl'),
        'training_rows': len(rows), 'families': len({r['family'] for r in rows}),
        'counts': dict(Counter(r['language']+':'+r['attributes']['assertion']+':'+str(r['review']) for r in rows)),
        'excluded': dict(excluded), 'independent_language_review': False, 'new_held_out_claim': False,
        'limitation': 'Synthetic authored TRAIN only. Exact exclusion does not establish author or family independence.'}
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    (a.output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k: v for k, v in report.items() if k != 'inputs'}, indent=2))


if __name__ == '__main__':
    main()
