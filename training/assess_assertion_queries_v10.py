"""Recompute zero-shot assertion diagnostics against the frozen v9 head."""
import argparse
from collections import Counter
import json
from pathlib import Path

from roguard.trajectory_data import file_sha, read_jsonl
from training.probe_assertion_query_v10 import LABELS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--probes', type=Path, nargs='+', required=True)
    for name in ('reference_study', 'reference_private', 'study'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists():
        raise ValueError('Refusing to overwrite assessment')
    import numpy as np
    reference = json.loads((a.reference_study/'report.json').read_text())
    if not reference['complete']:
        raise ValueError('Incomplete reference features')
    original = {}
    for lang, record in reference['languages'].items():
        for suffix, expected in record['artifacts'].items():
            if file_sha(a.reference_private/(lang+'-'+suffix)) != expected:
                raise ValueError('Reference artifact differs')
        rows = read_jsonl(a.reference_private/(lang+'-rows.jsonl'))
        z = np.load(a.reference_private/(lang+'-logits.npy'), allow_pickle=False)
        if z.shape != (len(rows), 19) or not np.isfinite(z).all():
            raise ValueError('Reference logits differ')
        for row, values in zip(rows, z):
            key = (lang, row['token_sha256'])
            if key in original:
                raise ValueError('Duplicate reference input')
            original[key] = (row, LABELS[int(values[6:10].argmax())])
    summaries = {}; sources = {}; selected_sha = None
    for probe in a.probes:
        protocol = json.loads((probe/'protocol.json').read_text())
        report = json.loads((probe/'report.json').read_text())
        if not report['complete'] or report['predictions_sha256'] != file_sha(probe/'predictions.jsonl'):
            raise ValueError('Incomplete or changed probe predictions')
        current_sha = file_sha(probe/'selected-development.jsonl')
        if current_sha != protocol['selected_development_sha256'] or selected_sha not in (None, current_sha):
            raise ValueError('Probe development sample differs')
        selected_sha = current_sha
        for path, expected in protocol['inputs'].items():
            if file_sha(path) != expected:
                raise ValueError('Probe source input differs')
        selected = read_jsonl(probe/'selected-development.jsonl')
        lookup = {(e['row']['language'], e['token_sha256']): e for e in selected}
        if len(lookup) != 256 or len(selected) != 256 or any(e['partition'] != 'dev' for e in selected):
            raise ValueError('Invalid selected development inventory')
        predictions = read_jsonl(probe/'predictions.jsonl'); seen = set(); groups = {}
        for prediction in predictions:
            lang = prediction['language']; prompt = prediction['prompt']
            key = (lang, prediction['original_token_sha256']); binding = (prompt, *key)
            if binding in seen or key not in lookup or prompt not in protocol['prompts']:
                raise ValueError('Unexpected or duplicated query prediction')
            seen.add(binding); entry = lookup[key]; row = entry['row']
            baseline_row, baseline = original[key]
            gold = row['attributes']['assertion']; predicted = prediction['predicted']
            if (baseline_row['partition'] != 'dev' or baseline_row['case_id'] != row['case_id']
                    or baseline_row['fields'][2] != LABELS.index(gold)
                    or prediction['case_id'] != row['case_id'] or prediction['gold'] != gold
                    or predicted not in LABELS):
                raise ValueError('Query and baseline row identity differ')
            counts = groups.setdefault(lang, {}).setdefault(prompt, Counter())
            counts['total'] += 1; counts['baseline_correct'] += baseline == gold
            counts['query_correct'] += predicted == gold
            counts['fixed'] += baseline != gold and predicted == gold
            counts['introduced'] += baseline == gold and predicted != gold
        if len(seen) != len(lookup)*len(protocol['prompts']):
            raise ValueError('Missing query predictions')
        for lang, prompts in groups.items():
            for prompt, counts in prompts.items():
                if counts['total'] != 128 or counts['query_correct'] != report['groups'][lang][prompt]['correct']:
                    raise ValueError('Reported accuracy differs from recomputed predictions')
        summaries[probe.name] = groups
        sources[probe.name] = {name: file_sha(probe/name) for name in ('protocol.json', 'report.json', 'predictions.jsonl')}
    a.study.mkdir(parents=True)
    result = {'format': 'roguard-assertion-query-assessment-1', 'complete': True,
        'source_sha256': file_sha(Path(__file__)), 'reference_report_sha256': file_sha(a.reference_study/'report.json'),
        'probe_sources': sources, 'selected_development_sha256': selected_sha, 'comparisons': summaries,
        'decision': 'REJECT', 'reason': 'Zero-shot query errors remain; this exposed development diagnostic cannot qualify a release.',
        'fitting_performed': False, 'independent_test': False, 'stable_release_ready': False}
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(summaries, indent=2))


if __name__ == '__main__':
    main()
