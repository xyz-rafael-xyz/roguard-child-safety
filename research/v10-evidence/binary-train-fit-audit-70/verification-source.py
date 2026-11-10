"""Recount the retained CPU audit using source metadata and standard Python."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Refuse overwrite'
    from roguard.evidence import ONTOLOGY
    fields = list(ONTOLOGY)
    folder = a.private / 'binary-train-fit-audit-70'
    report = json.loads((folder / 'actual/report.json').read_text())
    plan = json.loads((folder / 'plan.json').read_text())
    assert report['complete'] and report['cpu_only']
    assert report['plan_sha256'] == sha(folder / 'plan.json')
    assert report['bindings'] == plan['bindings']
    assert report['source_sha256'] == sha(a.private / 'audit-binary-train-fit-70.py')
    assert report['predictions_sha256'] == sha(folder / 'actual/train-predictions.jsonl')
    assert report['encoder_loaded'] is report['training_launched'] is report['development_logits_evaluated'] is False
    assert report['reserved_test_opened'] is report['held_out_accuracy_claimed'] is report['stable_release_ready'] is False
    feature_report_path = a.root / 'research/v10-evidence/crossed-review-features-62/report.json'
    assert sha(feature_report_path) == plan['bindings']['feature_report']
    extraction = json.loads(feature_report_path.read_text())
    source = {}
    for lang in ('ro', 'uk'):
        path = a.private / 'crossed-review-features-62' / (lang + '-rows.jsonl')
        assert sha(path) == extraction['languages'][lang]['artifacts']['rows.jsonl']
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row['partition'] == 'train':
                key = (lang, row['case_id'])
                assert key not in source
                source[key] = row
    actual = [json.loads(line) for line in (folder / 'actual/train-predictions.jsonl').read_text().splitlines()]
    assert len(actual) == report['prediction_rows'] == len(source) == 28929
    assert len({(r['language'], r['case_id']) for r in actual}) == len(actual)
    grouped = defaultdict(list)
    overlap = {}
    for row in actual:
        lang = row['language']
        original = source[(lang, row['case_id'])]
        for field in ('partition', 'group', 'gold'):
            assert row[field] == original[field]
        assert row['field_gold'] == original['fields']
        for model in ('v9', 'candidate'):
            pred = row[model]
            assert len(pred['fields']) == len(fields)
            assert all(type(v) is int and 0 <= v < len(ONTOLOGY[field]) for field, v in zip(fields, pred['fields']))
            assert type(pred['binary']) is int and pred['binary'] in (0, 1)
            assert isinstance(pred['score'], (int, float)) and math.isfinite(pred['score']) and 0 <= pred['score'] <= 1
            threshold = report['languages'][lang]['modes'][model]['threshold']
            assert pred['binary'] == int(pred['score'] >= threshold)
        origin = {'controlled-synthetic-scope-composition': 'generated56',
                  'controlled-crossed-scope-composition': 'crossed61'}.get(row['group'], 'historical')
        for group in ('all', 'origin:' + origin, 'source:' + row['group']):
            grouped[(lang, group)].append(row)
    for lang in ('ro', 'uk'):
        records = report['languages'][lang]
        assert records['cached_v9_maximum_logit_error'] <= 1e-3
        assert records['cached_v9_binary_disagreements'] == 0
        assert set(records['groups']) == {group for language, group in grouped if language == lang}
        for group, summary in records['groups'].items():
            rows = grouped[(lang, group)]
            recounted = {'rows': len(rows)}
            for model in ('v9', 'candidate'):
                wrong = [r for r in rows if r[model]['binary'] != r['gold']]
                recounted[model] = {
                    'binary_errors': len(wrong),
                    'false_positive': sum(r['gold'] == 0 for r in wrong),
                    'false_negative': sum(r['gold'] == 1 for r in wrong),
                    'joint_field_errors': sum(r[model]['fields'] != r['field_gold'] for r in rows),
                    'field_errors': {f: sum(r[model]['fields'][j] != r['field_gold'][j] for r in rows) for j, f in enumerate(fields)},
                }
            recounted['fixed'] = sum(r['v9']['binary'] != r['gold'] and r['candidate']['binary'] == r['gold'] for r in rows)
            recounted['introduced'] = sum(r['v9']['binary'] == r['gold'] and r['candidate']['binary'] != r['gold'] for r in rows)
            assert recounted == summary, (lang, group)
        rows = grouped[(lang, 'all')]
        overlap[lang] = {
            'binary_errors_with_correct_field_argmax': sum(r['candidate']['binary'] != r['gold'] and r['candidate']['fields'] == r['field_gold'] for r in rows),
            'binary_and_joint_field_errors': sum(r['candidate']['binary'] != r['gold'] and r['candidate']['fields'] != r['field_gold'] for r in rows),
            'joint_field_errors_with_correct_binary': sum(r['candidate']['binary'] == r['gold'] and r['candidate']['fields'] != r['field_gold'] for r in rows),
        }
    proof = {
        'complete': True, 'verified': True, 'source_sha256': sha(Path(__file__)),
        'report_sha256': sha(folder / 'actual/report.json'), 'plan_sha256': sha(folder / 'plan.json'),
        'predictions_sha256': report['predictions_sha256'],
        'rows': len(actual), 'language_source_groups_verified': len(grouped),
        'candidate_error_overlap': overlap,
        'scope': 'Exact TRAIN identities, source gold, saved-score threshold decisions and aggregate recount. CPU numerical parity is a recorded check from the bound audit, not independently rerun here.',
        'model_inference_performed': False, 'reserved_test_opened': False,
        'qualification_changed': False, 'stable_release_ready': False,
    }
    with a.output.open('x') as stream:
        stream.write(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
