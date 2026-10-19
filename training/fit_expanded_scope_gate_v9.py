"""Revisit the quotation scope gate with audited expanded training coverage."""
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import warnings

from roguard.evidence import disposition
from training.quote_scope_v9 import scope_text
from training.review_data_v8 import input_key
from training.diagnose_contextual_complementarity_v9 import read_rows
from training.budgeted_native_frontier_v9 import frontier


def main():
    import numpy as np
    from scipy.sparse import hstack
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from threadpoolctl import threadpool_limits
    from training.score_scope_gate_v9 import score as exported_score

    root = Path('research/v9-stabilization')
    study = root/'expanded-scope-gate-147'
    freeze = json.loads((study/'source-freeze.json').read_text())
    for path, expected in freeze.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source or data differs: ' + path)
    for name, expected in json.loads((study/'software.json').read_text()).items():
        if version(name) != expected:
            raise ValueError('Dependency version differs: ' + name)
    protocol = json.loads((study/'protocol.json').read_text())
    output = study/'actual'
    output.mkdir(exist_ok=False)
    warnings.simplefilter('error', ConvergenceWarning)
    threadpool_limits(4)
    contextual = read_rows(root/'gemma4-readout-141/inputs.jsonl.gz')
    report = dict(complete=False, languages={}, synthetic_only=True,
                  independently_annotated=False, calibration_inferred=False,
                  fresh_test_inferred=False, new_foundation_inference=False,
                  stable_release_ready=False)
    for language in ('ro', 'uk'):
        native = root/'native-scope-operators-131'/language
        training = read_rows(native/'train.jsonl.gz')
        training += read_rows(root/'indirect-distress-135'/(language+'-train.jsonl.gz'))
        training += read_rows(root/'freeform-scope-136'/(language+'-train.jsonl.gz'))
        protected = []
        for population in ('dev', 'cal', 'scope_dev', 'exposed', 'stress'):
            path = native/(population+'.jsonl.gz')
            if path.exists():
                protected += read_rows(path)
        reserved = read_rows(root/'paired-scope-117/data'/(language+'-test.jsonl'))
        def normalized(row):
            return dict(row, focus=row.get('focus', len(row['turns'])-1))
        training = [normalized(r) for r in training]
        protected = [normalized(r) for r in protected+reserved]
        if len(training) != protocol['training_rows'][language]:
            raise ValueError('Training population differs')
        if {input_key(r) for r in training} & {input_key(r) for r in protected}:
            raise ValueError('Training overlaps protected full input')
        dev_entries = [r for r in contextual if r['partition'] == 'dev' and r['row']['language'] == language]
        dev = [r['row'] for r in dev_entries]
        labels = np.array([int(disposition({k: {'label': '1' if k == 'concern' else v}
                         for k, v in r['attributes'].items()})[0] != 'CONTINUE') for r in training])
        gold = [int(disposition({k: {'label': v} for k, v in r['attributes'].items()})[0] != 'CONTINUE') for r in dev]
        bases = {}
        native_predictions = []
        for population in ('dev', 'scope_dev', 'stress'):
            path = native/(population+'-predictions.jsonl')
            if path.exists():
                native_predictions += read_rows(path)
        paths = dict(semantic=root/'gemma4-structured-143/actual', blend146=root/'contextual-complementarity-146/actual')
        sources = dict(native=native_predictions, **{name: read_rows(path/(language+'-dev-predictions.jsonl')) for name, path in paths.items()})
        for name, predictions in sources.items():
            lookup = {r['case_id']: r for r in predictions}
            if len(lookup) != len(predictions) or set(lookup) != {r['case_id'] for r in dev}:
                raise ValueError('Prediction identities differ')
            if any(lookup[r['case_id']]['gold'] != y for r, y in zip(dev, gold)):
                raise ValueError('Prediction labels differ')
            bases[name] = np.array([lookup[r['case_id']]['score'] for r in dev])
        vectorizers = []
        blocks = dict(train=[], dev=[])
        for focus, analyzer, ngrams, size in ((True, 'char', (2, 5), 16000), (True, 'word', (1, 3), 12000), (False, 'word', (1, 2), 6000)):
            vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams, min_df=2, max_features=size, sublinear_tf=True)
            blocks['train'].append(vec.fit_transform([scope_text(r, focus) for r in training]))
            blocks['dev'].append(vec.transform([scope_text(r, focus) for r in dev]))
            vectorizers.append(dict(focus=focus, analyzer=analyzer, ngram_range=ngrams,
                                    vocabulary=vec.vocabulary_, idf=vec.idf_.tolist()))
        matrices = {s: hstack(v).tocsr() for s, v in blocks.items()}
        abstract_train = {(scope_text(r, True), scope_text(r, False)) for r in training}
        collisions = sum((scope_text(r, True), scope_text(r, False)) in abstract_train for r in dev)
        entry = dict(training_rows=len(training), full_input_protected_overlap=0,
                     abstracted_dev_views_also_in_training=collisions,
                     abstraction_warning='Shared scope frames do not establish generalization to new frames',
                     selected=None, comparisons=[])
        best = None
        for name, values in bases.items():
            result = frontier([dict(gold=y, score=float(s), group=r['group']) for y, s, r in zip(gold, values, dev_entries)])
            if best is None or tuple(result['rank']) < best:
                best = tuple(result['rank'])
                entry['selected'] = dict(kind='unchanged_baseline', baseline=name, frontier=result)
        for c in protocol['regularization_C']:
            classifier = LogisticRegression(C=c, max_iter=1000, class_weight='balanced', random_state=1147).fit(matrices['train'], labels)
            scope = classifier.predict_proba(matrices['dev'])[:, 1]
            for name, baseline in bases.items():
                for power in protocol['scope_powers']:
                    values = baseline * scope ** power
                    population = [dict(gold=y, score=float(s), group=r['group']) for y, s, r in zip(gold, values, dev_entries)]
                    result = frontier(population)
                    record = dict(C=c, baseline=name, power=power, frontier=result)
                    entry['comparisons'].append(record)
                    if tuple(result['rank']) < best:
                        best = tuple(result['rank'])
                        spec = dict(format='roguard-scope-gate-experiment-v1', vectorizers=vectorizers,
                                    weights=classifier.coef_[0].tolist(), intercept=float(classifier.intercept_[0]),
                                    selected=dict(power=power, baseline=name, C=c, threshold=result['threshold']))
                        # Round-trip actual JSON serialization before independent reconstruction.
                        blob = gzip.compress(json.dumps(spec, default=int, separators=(',', ':')).encode(), mtime=0)
                        restored = json.loads(gzip.decompress(blob))
                        error = float(np.max(np.abs(exported_score(restored, dev, baseline)-values)))
                        if error > 1e-12:
                            raise ValueError('Portable scope gate differs')
                        (output/(language+'-scope-gate.json.gz')).write_bytes(blob)
                        entry['selected'] = dict(kind='scope_gate', **record, export_max_error=error,
                                                 artifact_sha256=hashlib.sha256(blob).hexdigest())
                        predictions = [dict(case_id=r['row']['case_id'], language=language, **p,
                                       predicted=int(p['score'] >= result['threshold'])) for r, p in zip(dev_entries, population)]
                        (output/(language+'-dev-predictions.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in predictions))
            print(language, 'C', c, json.dumps(entry['selected']), flush=True)
        report['languages'][language] = entry
        (output/'development-progress.json').write_text(json.dumps(report, indent=2)+'\n')
    report.update(complete=True, finished_utc=datetime.now(timezone.utc).isoformat(),
                  development_passes=all(v['selected']['frontier']['counts_pass'] for v in report['languages'].values()))
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
