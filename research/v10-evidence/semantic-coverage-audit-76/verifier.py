"""Independently recount every semantic-support audit record and saved logit."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite verification')
    import numpy as np
    from roguard.trajectory_data import file_sha, read_jsonl
    from roguard.evidence import ONTOLOGY
    from roguard.gemma4_review import _score
    public = a.root/'research/v10-evidence/semantic-coverage-audit-76'
    private = a.private/'semantic-coverage-audit-76/actual'
    report = json.loads((private/'report.json').read_text()); plan = json.loads((public/'plan.json').read_text())
    assert report['complete'] and report['plan_sha256'] == file_sha(public/'plan.json')
    assert report['source_sha256'] == file_sha(public/'source.py')
    for base, bindings in ((a.root, plan['public']), (a.private, plan['private'])):
        for name, value in bindings.items(): assert file_sha(base/name) == value
    assert report['predictions_sha256'] == file_sha(private/'predictions.jsonl')
    records = read_jsonl(private/'predictions.jsonl')
    by_position = {(row['language'], row['feature_index']): row for row in records}
    assert len(records) == len(by_position) == 37766
    fields = list(ONTOLOGY); sizes = [len(ONTOLOGY[name]) for name in fields]
    summaries = {}
    for lang in ('ro', 'uk'):
        metadata = read_jsonl(a.private/'crossed-review-features-62'/(lang+'-rows.jsonl'))
        teacher = np.load(a.private/'crossed-review-features-62'/(lang+'-logits.npy'), allow_pickle=False)
        spec = json.loads((a.root/'research/v10-evidence/dropout-readout-75'/(lang+'-review.json')).read_text())
        old_spec = json.loads((a.root/'research/v10-evidence/crossed-review-features-62/v9-binding'/(lang+'-review.json')).read_text())
        support = Counter(tuple(row['fields']) for row in metadata if row['partition'] == 'train')
        saved = report['languages'][lang]
        assert saved['train_worlds'] == len(support)
        assert saved['dev_worlds'] == len({tuple(row['fields']) for row in metadata if row['partition'] == 'dev'})
        assert saved['all_ontology_worlds'] == int(np.prod(sizes))
        assert saved['train_world_counts'] == [{'fields': list(world), 'rows': count} for world, count in sorted(support.items())]
        for split in ('train', 'dev'):
            indices = [i for i, row in enumerate(metadata) if row['partition'] == split]
            summary = saved['partitions'][split]
            path = private/(lang+'-'+split+'-logits.npy')
            assert file_sha(path) == summary['logits_sha256']
            logits = np.load(path, allow_pickle=False)
            assert logits.shape == (len(indices), sum(sizes)+2) and np.isfinite(logits).all()
            if split == 'dev':
                assert np.array_equal(logits, np.load(a.private/'dropout-readout-75'/(lang+'-selected-dev-logits.npy'), allow_pickle=False))
            totals = Counter(); confusions = [Counter() for _ in fields]; wording = Counter()
            for j, i in enumerate(indices):
                row = metadata[i]; actual = by_position[(lang, i)]
                assert all(actual[k] == row[k] for k in ('case_id', 'partition', 'group', 'gold', 'fields', 'token_sha256'))
                predicted = []; offset = 0
                for size in sizes:
                    predicted.append(int(np.argmax(logits[j, offset:offset+size]))); offset += size
                assert predicted == actual['predicted_fields']
                assert actual['train_world_support'] == support.get(tuple(row['fields']), 0)
                score = _score(logits[j], spec); baseline_score = _score(teacher[i], old_spec)
                assert score == actual['candidate_score'] and baseline_score == actual['v9_score']
                assert actual['candidate'] == int(score >= spec['threshold']) and actual['v9'] == int(baseline_score >= old_spec['threshold'])
                wrong = [gold != pred for gold, pred in zip(row['fields'], predicted)]
                totals['binary_errors'] += actual['candidate'] != row['gold']
                totals['v9_binary_errors'] += actual['v9'] != row['gold']
                totals['joint_field_errors'] += any(wrong)
                totals['binary_errors_with_all_fields_correct'] += actual['candidate'] != row['gold'] and not any(wrong)
                unseen = tuple(row['fields']) not in support
                totals['rows_with_unseen_worlds'] += unseen; totals['joint_errors_with_unseen_worlds'] += unseen and any(wrong)
                totals['formal_uncertain_to_affirmed_errors'] += row['group'] == 'formal' and row['fields'][2] == 2 and predicted[2] == 0
                key = (actual['source_kind'], row['group'], ONTOLOGY['assertion'][row['fields'][2]],
                       actual['has_opening_quote'], actual['negation_before_first_quote'])
                wording[(*key, 'rows')] += 1; wording[(*key, 'assertion_errors')] += wrong[2]
                for k, field in enumerate(fields): confusions[k][(ONTOLOGY[field][row['fields'][k]], ONTOLOGY[field][predicted[k]])] += 1
            assert summary['rows'] == len(indices)
            for name in ('binary_errors', 'v9_binary_errors', 'joint_field_errors', 'binary_errors_with_all_fields_correct',
                         'rows_with_unseen_worlds', 'joint_errors_with_unseen_worlds', 'formal_uncertain_to_affirmed_errors'):
                assert summary[name] == totals[name]
            for k, field in enumerate(fields):
                assert {(v['gold'], v['predicted']): v['rows'] for v in summary['field_confusions'][field]} == dict(confusions[k])
                assert summary['field_errors'][k] == sum(n for (g, v), n in confusions[k].items() if g != v)
            expected_wording = {}
            for group in summary['wording_groups']:
                key = tuple(group[k] for k in ('source_kind', 'group', 'assertion_gold', 'has_opening_quote', 'negation_before_first_quote'))
                for metric in ('rows', 'assertion_errors'): expected_wording[(*key, metric)] = group[metric]
            assert expected_wording == dict(wording)
            summaries[lang+':'+split] = {'rows': len(indices), **dict(totals)}
    result = {'verified': True, 'source_sha256': file_sha(__file__), 'report_sha256': file_sha(private/'report.json'),
              'predictions_sha256': file_sha(private/'predictions.jsonl'), 'rows_verified': len(records),
              'summaries': summaries, 'encoder_loaded': False, 'new_head_inference_performed': False, 'qualified': False,
              'scope': 'Exact source hashes, every metadata position/gold label, saved logit field/score prediction, '
                       'semantic support, confusion and wording-group recount. Descriptor semantics and causal claims are not validated.'}
    a.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
