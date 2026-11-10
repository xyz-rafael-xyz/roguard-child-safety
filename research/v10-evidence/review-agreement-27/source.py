"""Development-gated frozen review agreement; no fitting on regression labels."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.gemma4_contract import classifier_tokens
from roguard.gemma4_review import _score
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl, prefix, input_identity
from roguard.trajectory_evaluation import checkpoint_id
from training.native_fields_v9 import targets


ROOT = Path('research/v10-evidence')
RULES = ('agreement', 'contract-routed-agreement')


def decide(record, rule):
    if rule not in RULES:
        raise ValueError('Unregistered agreement rule')
    if rule == 'contract-routed-agreement' and record['outside_original_contract']:
        return record['attention']
    return int(record['v9'] == 1 and record['readout'] == 1)


def summarize(records, rule):
    errors = Counter(); totals = Counter(); positives = Counter(); routed = Counter()
    fixed = introduced = 0
    for row in records:
        language = row['language']; value = decide(row, rule)
        totals[language] += 1; positives[language] += row['gold']
        errors[language] += value != row['gold']
        fixed += row['v9'] != row['gold'] and value == row['gold']
        introduced += row['v9'] == row['gold'] and value != row['gold']
        routed[language] += rule == 'contract-routed-agreement' and row['outside_original_contract']
    if set(totals) != {'ro', 'uk'} or any(not 0 < positives[l] < totals[l] for l in totals):
        raise ValueError('Bilingual class coverage missing')
    return {'rows': sum(totals.values()), 'errors': sum(errors.values()), 'fixed': fixed, 'introduced': introduced,
        'groups': {l: {'total': totals[l], 'errors': errors[l], 'attention_routed': routed[l]} for l in ('ro', 'uk')},
        'selection_rank': [max(errors.values()), sum(errors.values()), RULES.index(rule)]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'features', 'readout_artifacts', 'attention_artifacts', 'study'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists():
        raise ValueError('Refusing to overwrite agreement study')
    parent_study = ROOT/'base-reference-features-18b'
    parent = json.loads((parent_study/'report.json').read_text())
    readout_study = ROOT/'semantic-readout-11'; attention_study = ROOT/'semantic-tail-14'
    regression = ROOT/'semantic-regression-11'; attention_regression = ROOT/'semantic-tail-regression-14'
    reports = {str(path): json.loads((path/'report.json').read_text()) for path in
               (parent_study, readout_study, attention_study, regression, attention_regression)}
    if any(not r['complete'] for r in reports.values()):
        raise ValueError('Incomplete parent experiment')
    for lang, record in parent['languages'].items():
        for suffix, expected in record['artifacts'].items():
            if file_sha(a.features/(lang+'-'+suffix)) != expected:
                raise ValueError('Original feature binding differs')
    specifications = {}
    for kind, study, artifacts in (('readout', readout_study, a.readout_artifacts),
                                    ('attention', attention_study, a.attention_artifacts)):
        specifications[kind] = {}
        for lang in ('ro', 'uk'):
            spec = json.loads((study/(lang+'-review.json')).read_text())
            if spec['adapter_sha256'] != file_sha(artifacts/(lang+'-selected.safetensors')):
                raise ValueError('Frozen component artifact differs')
            specifications[kind][lang] = spec
    a.study.mkdir(parents=True)
    frozen = LocalFrozenV9(a.base, Path('research/v9-stabilization/reported-tail-fit-196'), a.study/'v9-binding')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.base), trust_remote_code=False, local_files_only=True)
    import numpy as np
    metadata = {}; teacher = {}; readout_dev = {}; attention_dev = {}
    for lang in ('ro', 'uk'):
        rows = read_jsonl(a.features/(lang+'-rows.jsonl'))
        z = np.load(a.features/(lang+'-logits.npy'), allow_pickle=False)
        for row, values in zip(rows, z):
            if row['partition'] == 'dev':
                key = (lang, row['case_id'])
                if key in metadata:
                    raise ValueError('Duplicate eligible development input')
                metadata[key] = row
                teacher[key] = int(_score(values, frozen.models[lang]._spec) >= frozen.models[lang]._spec['threshold'])
        for kind, study, destination in (('readout', readout_study, readout_dev), ('attention', attention_study, attention_dev)):
            for row in read_jsonl(study/(lang+'-development-predictions.jsonl')):
                key = (lang, row['case_id'])
                if key in destination or row['predicted'] != int(row['score'] >= specifications[kind][lang]['threshold']):
                    raise ValueError('Frozen development prediction binding differs')
                destination[key] = row
    if len(metadata) != 6668 or set(attention_dev) != set(metadata):
        raise ValueError('Eligible development inventory differs')
    protocol = {'format': 'roguard-review-agreement-1', 'source_sha256': file_sha(Path(__file__)),
        'amendment_sha256': file_sha(ROOT/'protocol-amendment-25.json'),
        'parent_reports': {path: file_sha(Path(path)/'report.json') for path in reports},
        'component_specs': {str(study/(lang+'-review.json')): file_sha(study/(lang+'-review.json'))
                            for study in (readout_study, attention_study) for lang in ('ro', 'uk')},
        'rules': list(RULES), 'inputs': {}, 'fitting_performed': False, 'threshold_selection_performed': False,
        'regression_informed_motivation': True, 'independent_test': False, 'stable_release_ready': False}
    def outside(payload):
        return len(payload['turns']) > 12 or len(classifier_tokens(tokenizer, payload)) > 1024
    records = []; original_keys = set()
    for name in ('shared-tail-corpus-164', 'grounding-tail-corpus-174', 'referent-tail-corpus-176', 'reported-tail-corpus-195'):
        path = Path('research/v9-stabilization')/name/'inputs.jsonl.gz'; protocol['inputs'][str(path)] = file_sha(path)
        for entry in map(json.loads, gzip.open(path, 'rt', encoding='utf-8')):
            if entry['partition'] != 'dev':
                continue
            row = entry['row']; key = (row['language'], row['case_id']); old = metadata[key]
            if old['token_sha256'] != entry['token_sha256'] or old['fields'] != targets([row])[0]:
                raise ValueError('Original development row differs')
            if readout_dev[key]['gold'] != old['gold'] or attention_dev[key]['gold'] != old['gold']:
                raise ValueError('Development labels differ between components')
            payload = {'language': row['language'], 'turns': row['turns'][:row['focus']+1]}
            records.append({'case_id': row['case_id'], 'language': row['language'], 'gold': old['gold'],
                'v9': teacher[key], 'readout': readout_dev[key]['predicted'], 'attention': attention_dev[key]['predicted'],
                'outside_original_contract': outside(payload)})
            original_keys.add(key)
    if len(original_keys) != 6341 or set(readout_dev) != original_keys:
        raise ValueError('Original DEV coverage differs')
    def trajectory_records(population):
        path = ROOT/'data'/(population+'.jsonl') if population != 'broad' else ROOT/'broad-v9-regression/cases.jsonl'
        cases = read_jsonl(path); protocol['inputs'][str(path)] = file_sha(path)
        parents = {}
        for kind, study, suffix in (('v9', regression, 'baseline'), ('readout', regression, 'candidate'),
                                     ('attention', attention_regression, 'candidate')):
            parent_protocol = json.loads((study/'protocol.json').read_text())
            if parent_protocol['inputs'][population] != file_sha(path):
                raise ValueError('Trajectory parent input differs')
            pred_path = study/(population+'-'+suffix+'.jsonl')
            protocol['inputs'][str(pred_path)] = file_sha(pred_path)
            values = read_jsonl(pred_path); parents[kind] = {r['checkpoint_id']: r for r in values}
            if len(parents[kind]) != len(values):
                raise ValueError('Duplicate trajectory prediction')
        output = []
        for case in cases:
            for point in case['checkpoints']:
                cid = checkpoint_id(case, point); lang = case['language']; payload = prefix(case, point['focus'])
                if any(parents[kind][cid]['input_digest'] != input_identity(payload) for kind in parents):
                    raise ValueError('Trajectory prefix identity differs')
                if len({parents[kind][cid]['inference_input_digest'] for kind in parents}) != 1:
                    raise ValueError('Components evaluated different memory views')
                for kind in parents:
                    spec = frozen.models[lang]._spec if kind == 'v9' else specifications[kind][lang]
                    value = parents[kind][cid]
                    if value['predicted'] != int(value['score'] >= spec['threshold']):
                        raise ValueError('Saved trajectory decision differs from its threshold')
                output.append({'case_id': cid, 'language': lang, 'gold': point['review'],
                    **{kind: parents[kind][cid]['predicted'] for kind in parents},
                    'outside_original_contract': outside(payload)})
        if any(len(v) != len(output) for v in parents.values()):
            raise ValueError('Unexpected trajectory prediction inventory')
        return output
    trajectory_dev = trajectory_records('dev')
    eligible = [r for r in trajectory_dev if (r['language'], r['case_id']) in metadata]
    if len(eligible) != 327 or len(trajectory_dev)-len(eligible) != 21:
        raise ValueError('Prior training-overlap exclusion differs')
    for row in eligible:
        key = (row['language'], row['case_id'])
        if row['gold'] != metadata[key]['gold'] or row['v9'] != teacher[key] or row['attention'] != attention_dev[key]['predicted']:
            raise ValueError('Eligible trajectory development binding differs')
    records.extend(eligible)
    summaries = {rule: summarize(records, rule) for rule in RULES}
    selected = min(RULES, key=lambda rule: summaries[rule]['selection_rank'])
    selection = {'complete': True, 'selected': selected, 'development': summaries, 'new_regression_combination_scored': False}
    selection_path = a.study/'selection.json'; selection_path.write_text(json.dumps(selection, indent=2)+'\n')
    selection_sha = file_sha(selection_path)
    write_jsonl(a.study/'development-components.jsonl', records)
    report = {'complete': True, 'selection_sha256': selection_sha, 'selected': selected, 'development': summaries,
              'binary_component_only': True, 'independent_test': False, 'stable_release_ready': False, 'regression': {}}
    if summaries[selected]['errors'] == 0:
        for population in ('broad', 'challenge', 'v9-regression'):
            rows = trajectory_records(population)
            write_jsonl(a.study/(population+'-predictions.jsonl'), [{**r, 'predicted': decide(r, selected)} for r in rows])
            report['regression'][population] = summarize(rows, selected)
    if file_sha(selection_path) != selection_sha:
        raise ValueError('Agreement selection changed after regression scoring')
    report['decision'] = 'REJECT' if summaries[selected]['errors'] or any(r['errors'] for r in report['regression'].values()) else 'BINARY_ENGINEERING_PASS_ONLY'
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
