"""Matched TRAIN-only assertion wording; generated counts are correlated variants."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import json
from pathlib import Path
import re
import unicodedata

from roguard.assurance import digest
from roguard.evidence import ONTOLOGY
from roguard.trajectory_data import file_sha, prefix, write_jsonl
from training.evidence_v8 import ASSERTIONS, WORDS
from training.prepare_instruction_sft_v8 import target


# Each stance has two operators without the standalone negation marker and two
# with it. Both quote positions use exactly the same self-contained operator.
# These are author judgments, not independently annotated natural-language gold.
OPERATORS = {
    'ro': {
        'AFFIRMED': [
            'Confirm că descrierea din citat corespunde situației.',
            'Susțin ca adevărată descrierea din citat.',
            'Nu resping descrierea din citat; o confirm ca adevărată.',
            'Nu am dubii: descrierea din citat este adevărată.'],
        'NEGATED': [
            'Declar falsă descrierea din citat.',
            'Resping ca falsă descrierea din citat.',
            'Descrierea din citat nu este adevărată.',
            'Nu susțin descrierea din citat; afirm că este falsă.'],
        'UNCERTAIN': [
            'Sunt indecis dacă descrierea din citat este adevărată sau falsă.',
            'Încă oscilez între a considera descrierea din citat adevărată și a o considera falsă.',
            'Nu pot stabili dacă descrierea din citat este adevărată sau falsă.',
            'Nu am ajuns la o concluzie despre adevărul descrierii din citat.'],
        'RETRACTED': [
            'Retractez ceea ce am afirmat mai devreme în descrierea citată.',
            'Îmi retrag afirmația anterioară, reprodusă în citat.',
            'Nu mai susțin afirmația de mai devreme, reprodusă în citat; o retrag.',
            'Nu îmi mențin afirmația anterioară din citat; o retractez.']},
    'uk': {
        'AFFIRMED': [
            'Підтверджую, що опис у цитаті відповідає дійсності.',
            'Стверджую, що опис у цитаті правдивий.',
            'Не відкидаю опис у цитаті; підтверджую його правдивість.',
            'Не маю сумнівів: опис у цитаті правдивий.'],
        'NEGATED': [
            'Стверджую, що опис у цитаті хибний.',
            'Відкидаю опис у цитаті як хибний.',
            'Опис у цитаті не відповідає дійсності.',
            'Не підтримую опис у цитаті; стверджую, що він хибний.'],
        'UNCERTAIN': [
            'Вагаюся, чи опис у цитаті правдивий, чи хибний.',
            'Досі вагаюся між тим, щоб вважати опис у цитаті правдивим, і тим, щоб вважати його хибним.',
            'Не можу визначити, чи опис у цитаті правдивий, чи хибний.',
            'Не дійшов висновку щодо правдивості опису в цитаті.'],
        'RETRACTED': [
            'Відкликаю те, що раніше стверджував у наведеному описі.',
            'Відкликаю своє попереднє твердження, наведене в цитаті.',
            'Більше не підтримую попереднє твердження, наведене в цитаті; відкликаю його.',
            'Не залишаю чинним своє попереднє твердження в цитаті; відкликаю його.']}}


def normalized(text):
    return ' '.join(unicodedata.normalize('NFC', text).casefold().split())


def negation(text):
    return bool(re.search(r'\b(?:nu|не)\b', normalized(text)))


def decompose(row):
    if row['source'] != 'controlled_composition_v8' or not row['case_id'].startswith('train-'):
        raise ValueError('Only original controlled TRAIN parents are eligible')
    text = row['turns'][row['focus']]['text']
    if text.count('«') != 1 or text.count('»') != 1:
        raise ValueError('Expected exactly one quotation')
    assertion = row['attributes']['assertion']
    old = WORDS[row['language']]['assertion'][0][ASSERTIONS.index(assertion)]
    before, middle, after = text.partition(old+' «')
    if not middle or text.count(old+' «') != 1:
        raise ValueError('Expected the exact TRAIN assertion operator')
    core, closing, tail = after.partition('»')
    if not core or not closing or '«' in before:
        raise ValueError('Malformed controlled quotation')
    return before, core, tail


def transform(row, operator_index, placement):
    if operator_index not in range(4) or placement not in ('before', 'after'):
        raise ValueError('Unknown wording factor')
    before, core, tail = decompose(row)
    operator = OPERATORS[row['language']][row['attributes']['assertion']][operator_index]
    if negation(operator) != (operator_index >= 2):
        raise ValueError('Operator marker balance differs')
    quote = '«'+core+'»'
    clause = operator+' '+quote if placement == 'before' else quote+' '+operator
    changed = deepcopy(row)
    changed['turns'][changed['focus']]['text'] = before+clause+tail
    return changed


def load_parents(root, recovered_path, plan):
    sources = {}
    for relative in plan['historical_sources']:
        with gzip.open(root/relative, 'rt') as stream:
            for line in stream:
                row = json.loads(line)['row']
                key = (row['language'], row['case_id'])
                if key in sources:
                    raise ValueError('Duplicate historical source')
                sources[key] = row
    choices = {}; protected = set(); counts = Counter()
    with recovered_path.open() as stream:
        for line in stream:
            recovered = json.loads(line)
            payload = recovered['payload']
            protected.add(normalized(payload['turns'][-1]['text']))
            if recovered['partition'] != 'train':
                continue
            row = sources.get((recovered['language'], recovered['case_id']))
            if row is None or row['source'] != 'controlled_composition_v8':
                continue
            if (digest(row) != recovered['source_row_sha256']
                    or prefix(row, row['focus']) != payload
                    or target(row) != recovered['gold']
                    or [ONTOLOGY[k].index(row['attributes'][k]) for k in ONTOLOGY] != recovered['fields']):
                raise ValueError('Recovered TRAIN source or gold differs')
            _, core, _ = decompose(row)
            counts[row['language']] += 1
            # One parent per actual core and gold-world combination. Contexts
            # are selected by stable source hash, never model error or DEV data.
            key = (row['language'], normalized(core), tuple(recovered['fields']))
            if key not in choices or digest(row) < digest(choices[key][0]):
                choices[key] = (row, recovered)
    return [choices[k] for k in sorted(choices)], protected, dict(counts)


def build(parents, protected):
    treatment = []; control = []; seen = set(); selected = []
    for row, recovered in parents:
        before, core, tail = decompose(row)
        identity = {'language': row['language'], 'case_id': row['case_id'],
                    'source_row_sha256': digest(row), 'feature_index': recovered['feature_index'],
                    'token_sha256': recovered['token_sha256'], 'fields': recovered['fields'],
                    'gold': recovered['gold'], 'core_sha256': digest(normalized(core)),
                    'family': row.get('family'), 'ordinary_family': row.get('ordinary_family')}
        selected.append(identity)
        for op in range(4):
            for placement in ('before', 'after'):
                pair_id = 'assertion77:'+row['language']+':'+digest(identity)[:20]+':'+str(op)+':'+placement
                changed = transform(row, op, placement)
                text = changed['turns'][changed['focus']]['text']
                if normalized(text) in protected:
                    raise ValueError('Treatment focal text exactly overlaps existing TRAIN/DEV')
                key = digest(prefix(changed, changed['focus']))
                if key in seen:
                    raise ValueError('Duplicate treatment input')
                seen.add(key)
                factors = {'operator_index': op, 'quote_placement': placement,
                           'operator_negation_marker': op >= 2}
                for arm, example, rows in (('treatment', changed, treatment), ('control', deepcopy(row), control)):
                    example.update(case_id=pair_id+':'+arm, partition='train', pair_id=pair_id,
                                   arm=arm, parent=identity, factors=factors,
                                   review=target(row), synthetic_only=True, independently_annotated=False)
                    actual_text = example['turns'][example['focus']]['text']
                    example['literal_pre_quote_negation'] = negation(actual_text.partition('«')[0])
                    rows.append(example)
    return treatment, control, selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private', 'plan', 'study', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.study.exists():
        raise ValueError('Refusing to overwrite an experiment')
    plan = json.loads(args.plan.read_text())
    for base, mapping in ((args.root, plan['public']), (args.private, plan['private'])):
        for relative, expected in mapping.items():
            if file_sha(base/relative) != expected:
                raise ValueError('Bound input differs: '+relative)
    parents, protected, eligible = load_parents(args.root, args.private/plan['recovered_inputs'], plan)
    treatment, control, selected = build(parents, protected)
    args.output.mkdir(parents=True); args.study.mkdir(parents=True)
    for name, rows in (('additional-train', treatment), ('matched-control', control), ('parents', selected)):
        write_jsonl(args.output/(name+'.jsonl'), rows)
    counts = Counter((r['language'], r['attributes']['assertion'], r['factors']['operator_negation_marker'],
                      r['factors']['quote_placement'], r['review']) for r in treatment)
    report = {'complete': True, 'format': 'roguard-matched-assertion-variation-1',
              'source_sha256': file_sha(__file__), 'plan_sha256': file_sha(args.plan),
              'eligible_original_train_rows': eligible, 'selected_parents': len(parents),
              'unique_cores_by_language': {lang: len({r['core_sha256'] for r in selected if r['language'] == lang}) for lang in ('ro', 'uk')},
              'rows_per_arm': len(treatment), 'variants_per_parent': 8,
              'artifacts': {p.name: file_sha(p) for p in sorted(args.output.glob('*.jsonl'))},
              'counts': [{'language': k[0], 'assertion': k[1], 'operator_negation_marker': k[2],
                          'quote_placement': k[3], 'review': k[4], 'rows': n} for k, n in sorted(counts.items())],
              'exact_treatment_overlap_with_existing_focal_texts': 0,
              'encoder_loaded': False, 'training_performed': False, 'training_admitted': False,
              'independent_language_or_label_review': False, 'reserved_test_opened': False, 'qualified': False,
              'limitations': ['Counts are correlated rewrites of existing TRAIN cores, not independent meanings.',
                             'Labels are intended invariances of controlled synthetic parents and are not independently validated.',
                             'The control repeats existing inputs eight times per selected parent.',
                             'Artifact reproduction, source/label conservation, token envelope and operator balance must be audited before any prospective fitting plan.']}
    (args.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (args.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: report[k] for k in ('eligible_original_train_rows', 'selected_parents', 'unique_cores_by_language', 'rows_per_arm')}))


if __name__ == '__main__':
    main()
