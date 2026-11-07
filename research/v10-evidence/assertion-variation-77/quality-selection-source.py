"""Explicit author-curated TRAIN slice; no independent semantic certification."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


ALLOW = {
    'ro': {
        'Am nevoie de ajutor cu împărțirea.', 'Am nevoie să vorbesc cu cineva.',
        'Am o problemă și nu mă descurc singur.', 'Aș vrea să fiu ascultat.',
        'Mi se pare că nu am pe nimeni.', 'Mi-e teamă că am calculat greșit aria.',
        'Mi-e teamă să merg la școală.', 'Mă neliniștește ce se întâmplă.',
        'Mă simt exclus din grup.', 'Mă simt singur.', 'Mă îngrijorează ziua de mâine.',
        'Nimeni nu mă ascultă.', 'Nu găsesc piesa care lipsește din puzzle.',
        'Nu mai țin minte regula de ortografie.', 'Nu pot să deschid fișierul cu tema.',
        'Nu înțeleg cerința exercițiului.', 'Nu știu cui să spun ce mă apasă.',
        'Nu știu cum să desenez o bicicletă.', 'Scriu o replică pentru un spectacol de păpuși.',
        'Vreau să scriu o scenă pentru teatrul școlii.', 'Îmi este greu să cer ajutor.',
        'Îmi place să construiesc modele de avioane.'},
    'uk': {
        'Боюся, що неправильно обчислив площу.', 'Здається, у мене нікого немає.',
        'Мене ніхто не слухає.', 'Мене тривожить те, що відбувається.',
        'Мене турбує завтрашній день.', 'Мені важко просити про допомогу.',
        'Мені подобається будувати моделі літаків.', 'Мені потрібна допомога з діленням.',
        'Мені потрібно з кимось поговорити.', 'Мені самотньо.', 'Мені страшно йти до школи.',
        'Не знаю, кому розповісти про те, що мене гнітить.', 'Не знаю, як намалювати велосипед.',
        'Не можу відкрити файл із домашнім завданням.', 'Не можу знайти деталь пазла.',
        "Не пам'ятаю правило правопису.", 'Не розумію умову вправи.',
        'Пишу репліку для лялькової вистави.', 'У мене проблема, і я не можу впоратися сам.',
        'Хотілося б, щоб мене вислухали.', 'Хочу написати сценку для шкільного театру.',
        'Я почуваюся зайвим у компанії.'}}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write(path, rows):
    with path.open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False)+'\n')


def main():
    parser = argparse.ArgumentParser()
    for name in ('private', 'plan', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    a = parser.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite selected data')
    plan = json.loads(a.plan.read_text())
    assert sha(Path(__file__)) == plan['selection_source_sha256']
    for name, expected in plan['private'].items():
        assert sha(a.private/name) == expected, name
    original = a.private/'assertion-variation-77/actual'
    treatment = read(original/'additional-train.jsonl'); control = read(original/'matched-control.jsonl')
    decisions = {}; observed = {lang: set() for lang in ALLOW}
    for row in treatment:
        key = (row['language'], row['parent']['case_id'])
        if key in decisions: continue
        core = row['turns'][row['focus']]['text'].split('«')[1].split('»')[0]
        observed[row['language']].add(core)
        reasons = []
        if core not in ALLOW[row['language']]: reasons.append('core_outside_explicit_current_first_person_declarative_allowlist')
        if row['attributes']['subject'] != 'SELF': reasons.append('outer_subject_not_self')
        if row['attributes']['time'] != 'CURRENT': reasons.append('outer_time_not_current')
        decisions[key] = {'language': row['language'], 'parent_case_id': row['parent']['case_id'],
                          'core': core, 'attributes': row['attributes'], 'selected': not reasons, 'exclusions': reasons}
    assert all(len(words) == 22 and words <= observed[lang] for lang, words in ALLOW.items())
    chosen = {key for key, row in decisions.items() if row['selected']}
    selected = [r for r in treatment if (r['language'], r['parent']['case_id']) in chosen]
    selected_control = [r for r in control if (r['language'], r['parent']['case_id']) in chosen]
    assert len(selected) == len(selected_control) == len(chosen)*8
    assert [r['pair_id'] for r in selected] == [r['pair_id'] for r in selected_control]
    runtime = read(a.private/'assertion-variation-77/verification/runtime-inputs.jsonl')
    selected_runtime = [r for r in runtime if (r['language'], r['parent']['case_id']) in chosen]
    assert [r['case_id'] for r in selected_runtime] == [r['case_id'] for r in selected]
    a.output.mkdir(parents=True)
    for name, rows in (('additional-train', selected), ('matched-control', selected_control),
                       ('runtime-inputs', selected_runtime), ('parent-decisions', list(decisions.values()))):
        write(a.output/(name+'.jsonl'), rows)
    counts = Counter((r['language'], r['attributes']['assertion'], r['review']) for r in selected)
    report = {'complete': True, 'source_sha256': sha(Path(__file__)), 'plan_sha256': sha(a.plan),
              'full_expansion_training_admitted': False, 'selected_parents': len(chosen),
              'excluded_parents': len(decisions)-len(chosen), 'rows_per_arm': len(selected),
              'unique_cores_by_language': {lang: len({d['core'] for d in decisions.values() if d['selected'] and d['language'] == lang}) for lang in ALLOW},
              'counts': [{'language': k[0], 'assertion': k[1], 'review': k[2], 'rows': n} for k, n in sorted(counts.items())],
              'exclusion_reasons': dict(Counter(reason for d in decisions.values() for reason in d['exclusions'])),
              'artifacts': {p.name: sha(p) for p in sorted(a.output.glob('*.jsonl'))},
              'scope': 'Author-curated first-person CURRENT declarative subset. All parent labels, context and factors are preserved. Exclusions are conservative and do not establish that every excluded label is wrong.',
              'training_performed': False, 'training_admitted': False, 'independent_language_or_label_validation': False,
              'encoder_loaded': False, 'reserved_test_opened': False, 'qualified': False}
    (a.output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
