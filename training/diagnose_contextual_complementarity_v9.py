"""Compare already selected native/contextual scores on development only."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
import math
from pathlib import Path

from training.budgeted_native_frontier_v9 import frontier


def blend(native, semantic, weight, mode):
    if not 0 <= weight <= 1 or not all(math.isfinite(x) and 0 <= x <= 1 for x in (native, semantic)):
        raise ValueError('Invalid blend input')
    if mode == 'probability':
        return weight * native + (1 - weight) * semantic
    if mode != 'logit':
        raise ValueError('Unknown blend mode')
    def logit(value):
        value = min(1 - 1e-12, max(1e-12, value))
        return math.log(value) - math.log1p(-value)
    value = weight * logit(native) + (1 - weight) * logit(semantic)
    return 1 / (1 + math.exp(-value))


def read_rows(path):
    raw = path.read_bytes()
    if path.suffix == '.gz':
        raw = gzip.decompress(raw)
    return [json.loads(line) for line in raw.decode().splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', type=Path, required=True)
    args = parser.parse_args()
    study = args.study
    freeze = json.loads((study/'source-freeze.json').read_text())
    for path, expected in freeze.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source or evidence differs: ' + path)
    protocol = json.loads((study/'protocol.json').read_text())
    native = Path(protocol['native'])
    semantic = Path(protocol['semantic'])
    contextual_rows = read_rows(Path(protocol['contextual_inputs']))
    output = study/'actual'
    output.mkdir(exist_ok=False)
    report = dict(complete=False, languages={}, synthetic_only=True,
                  new_neural_inference=False, calibration_inferred=False,
                  fresh_test_inferred=False, stable_release_ready=False)
    for language in ('ro', 'uk'):
        inputs = {}
        scores = {}
        for population in ('dev', 'scope_dev', 'stress'):
            path = native/language/(population+'.jsonl.gz')
            if not path.exists():
                continue
            for row in read_rows(path):
                if row['case_id'] in inputs:
                    raise ValueError('Ambiguous native input identity')
                inputs[row['case_id']] = row
            for row in read_rows(native/language/(population+'-predictions.jsonl')):
                if row['case_id'] in scores:
                    raise ValueError('Duplicate native prediction')
                scores[row['case_id']] = row
        expected = {r['row']['case_id']: r for r in contextual_rows
                    if r['partition'] == 'dev' and r['row']['language'] == language}
        predictions = read_rows(semantic/'actual'/(language+'-dev-predictions.jsonl'))
        if len(predictions) != len(expected) or {r['case_id'] for r in predictions} != set(expected):
            raise ValueError('Development populations differ')
        for row in predictions:
            key = row['case_id']
            if inputs[key] != expected[key]['row'] or scores[key]['gold'] != row['gold'] or row['group'] != expected[key]['group']:
                raise ValueError('Input content, label or group differs')
        variants = []
        selected = None
        best = None
        configurations = [('native', 1.), ('semantic', 0.)]
        configurations += [(mode, w) for mode in protocol['modes'] for w in protocol['native_weights']]
        for mode, weight in configurations:
            values = [scores[r['case_id']]['score'] if mode == 'native' else r['score'] if mode == 'semantic'
                      else blend(scores[r['case_id']]['score'], r['score'], weight, mode) for r in predictions]
            population = [dict(gold=r['gold'], group=r['group'], score=v) for r, v in zip(predictions, values)]
            result = frontier(population)
            variants.append(dict(mode=mode, native_weight=weight, frontier=result))
            rank = tuple(result['rank'])
            if best is None or rank < best:
                best = rank
                selected = variants[-1]
                selected_predictions = [dict(case_id=r['case_id'], language=language, **v,
                    predicted=int(v['score'] >= result['threshold'])) for r, v in zip(predictions, population)]
        (output/(language+'-dev-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in selected_predictions))
        report['languages'][language] = dict(selected=selected, comparisons=variants)
        print(language, json.dumps(selected), flush=True)
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                  development_passes=all(v['selected']['frontier']['counts_pass'] for v in report['languages'].values()))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
