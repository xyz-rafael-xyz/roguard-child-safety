"""Recount every independently assessed75 DEV identity and paired decision."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite verification')
    import numpy as np
    from roguard.gemma4_review import _score
    from roguard.trajectory_data import file_sha, read_jsonl
    e = a.root/'research/v10-evidence'; study = e/'dropout-readout-75'
    assessment = e/'dropout-readout-assessment-75'; report = json.loads((assessment/'report.json').read_text())
    proof = json.loads((e/'dropout-readout-qualification-75/verification.json').read_text())
    assert proof['verified'] and proof['report_sha256'] == file_sha(study/'report.json')
    assert report['selection_report_sha256'] == file_sha(study/'report.json') and report['complete']
    assert report['source_sha256'] == file_sha(a.root/'eval/assess_crossed_review_v10.py')
    path = a.private/'dropout-readout-assessment-75/predictions.jsonl'
    assert report['predictions_sha256'] == file_sha(path)
    values = read_jsonl(path); lookup = {(v['language'], v['case_id']): v for v in values}
    assert len(values) == len(lookup) == 8837
    groups = {}; maximum = 0.
    for lang in ('ro', 'uk'):
        features = a.private/'crossed-review-features-62'
        metadata = read_jsonl(features/(lang+'-rows.jsonl')); dev = [i for i, v in enumerate(metadata) if v['partition'] == 'dev']
        saved = read_jsonl(study/(lang+'-development-predictions.jsonl'))
        teacher = np.load(features/(lang+'-logits.npy'), allow_pickle=False)
        spec = json.loads((study/(lang+'-review.json')).read_text())
        original_spec = json.loads((e/'crossed-review-features-62/v9-binding'/(lang+'-review.json')).read_text())
        for i, reference in zip(dev, saved, strict=True):
            row = metadata[i]; value = lookup[(lang, row['case_id'])]
            assert value['gold'] == row['gold'] and value['token_sha256'] == row['token_sha256']
            assert reference['case_id'] == row['case_id'] and reference['gold'] == row['gold']
            assert value['candidate'] == reference['predicted'] == int(value['candidate_score'] >= spec['threshold'])
            delta = abs(value['candidate_score']-reference['score']); maximum = max(maximum, delta)
            assert delta <= 1e-5
            old = _score(teacher[i], original_spec)
            assert abs(old-value['v9_score']) < 1e-12 and value['v9'] == int(old >= original_spec['threshold'])
            origin = ('crossed' if row['case_id'].startswith('composition61:') else
                      'generated' if row.get('family', '').startswith('composition56:') else 'historical')
            assert value['origin'] == origin
            group = groups.setdefault(origin+':'+lang, dict(rows=0, candidate_errors=0, v9_errors=0, fixed=0, introduced=0))
            bad, old_bad = value['candidate'] != row['gold'], value['v9'] != row['gold']
            group['rows'] += 1; group['candidate_errors'] += bad; group['v9_errors'] += old_bad
            group['fixed'] += old_bad and not bad; group['introduced'] += bad and not old_bad
    assert groups == report['groups'] and report['batch_single_decision_disagreements'] == 0
    assert 0 <= report['maximum_batch_single_logit_error'] <= 1e-3
    assert report['decision'] == proof['decision'] == 'REJECT'
    result = {'verified': True, 'source_sha256': file_sha(__file__), 'assessment_sha256': file_sha(assessment/'report.json'),
              'checkpoint_verification_sha256': file_sha(e/'dropout-readout-qualification-75/verification.json'),
              'predictions_sha256': file_sha(path), 'rows_verified': len(values), 'groups': groups,
              'maximum_batch_single_score_difference': maximum, 'encoder_loaded': False,
              'new_inference_performed': False, 'qualified': False}
    a.output.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
