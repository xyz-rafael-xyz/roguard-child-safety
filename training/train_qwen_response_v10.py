"""A separately pinned pretrained response encoder and fixed isolated-draft fits."""
import argparse
import json
from pathlib import Path
import time
import unicodedata

from roguard.assurance import digest
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from roguard.trajectory_model import RESPONSE_LABELS
from training.train_neural_response_v10 import PROMPT, prediction_rows
from training.train_response_readout_v10 import summarize


def query_tokens(tokenizer, draft):
    messages = [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': json.dumps(
        {'prior_turns': [], 'focal_message': {'role': 'assistant', 'text': unicodedata.normalize('NFC', draft)}}, ensure_ascii=False)}]
    letters = 'ABCDE'; ids = [tokenizer.encode(v, add_special_tokens=False) for v in letters]
    variants = [tokenizer.apply_chat_template(messages+[{'role': 'assistant', 'content': v}],
        tokenize=True, return_dict=False, enable_thinking=False) for v in letters]
    if any(len(v) != 1 for v in ids) or len({len(v) for v in variants}) != 1:
        raise ValueError('Unaligned response answer tokens')
    positions = [i for i, values in enumerate(zip(*variants)) if len(set(values)) > 1]
    if len(positions) != 1 or [v[positions[0]] for v in variants] != [v[0] for v in ids]:
        raise ValueError('Response answer token position differs')
    return variants[0][:positions[0]], [v[0] for v in ids]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'data', 'contrast_study', 'study', 'private'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite Qwen response comparison')
    manifest_path = Path('research/v9-stabilization/qwen-control-87/model-manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if manifest['revision'] != '938d8919941c6e7efd3c7150eff7fe9d12afa631':
        raise ValueError('Unexpected Qwen revision')
    for name, record in manifest['files'].items():
        path = a.base/name
        if path.stat().st_size != record['bytes'] or file_sha(path) != record['sha256']:
            raise ValueError('Pinned Qwen file differs: '+name)
    data = json.loads((a.data/'report.json').read_text())
    contrast = json.loads((a.contrast_study/'protocol.json').read_text())
    for name in ('train', 'dev'):
        if file_sha(a.data/(name+'.jsonl')) != data['artifacts'][name]:
            raise ValueError('Original response data differs')
    if file_sha(a.contrast_study/'additional-train.jsonl') != contrast['additional_train_sha256']:
        raise ValueError('Contrast training inputs differ')
    all_rows = read_jsonl(a.data/'train.jsonl')+read_jsonl(a.data/'dev.jsonl')+read_jsonl(a.contrast_study/'additional-train.jsonl')
    rows = [r for r in all_rows if not r['context']]
    if len(rows) != 204 or len({(r['language'], r['draft']) for r in rows}) != 204:
        raise ValueError('Expected 204 unique isolated TRAIN/DEV drafts')
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    protocol = {'format': 'roguard-qwen-response-fit-1', 'source_sha256': file_sha(Path(__file__)),
        'amendment_sha256': file_sha('research/v10-evidence/protocol-amendment-35.json'),
        'base_manifest_sha256': file_sha(manifest_path), 'data_report_sha256': file_sha(a.data/'report.json'),
        'contrast_protocol_sha256': file_sha(a.contrast_study/'protocol.json'), 'prompt': PROMPT,
        'labels': list(RESPONSE_LABELS), 'draft_only': True, 'contextual_factual_support_established': False,
        'independent_validation': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    import mlx.core as mx
    from mlx_lm import load
    from sklearn.linear_model import LogisticRegression
    mx.set_memory_limit(10*1024**3); mx.set_cache_limit(128*1024**2)
    started = time.monotonic(); engine, tokenizer = load(str(a.base), tokenizer_config={'trust_remote_code': False})
    engine.set_dtype(mx.float32); model = engine.language_model
    def extract(population):
        features = []; native = []; identities = []
        for i, row in enumerate(population):
            tokens, labels = query_tokens(tokenizer, row['draft'])
            if not tokens or len(tokens) > 1024:
                raise ValueError('Unsupported response input envelope')
            hidden = model.model(mx.array([tokens]))[:, -1:, :]
            tied = getattr(model, 'tie_word_embeddings', model.args.tie_word_embeddings)
            logits = model.model.embed_tokens.as_linear(hidden) if tied else model.lm_head(hidden)
            softcap = getattr(model, 'final_logit_softcapping', None)
            if softcap is not None:
                logits = softcap*mx.tanh(logits/softcap)
            chosen = logits[0, 0, labels]; mx.eval(hidden, chosen)
            x = np.asarray(hidden)[0, 0].copy(); z = np.asarray(chosen).copy()
            if not np.isfinite(x).all() or not np.isfinite(z).all():
                raise ValueError('Nonfinite Qwen response outputs')
            features.append(x); native.append(z)
            identities.append({'case_id': row['case_id'], 'partition': row['partition'], 'query_token_sha256': digest(tokens)})
            del hidden, logits, chosen; mx.clear_cache()
            if (i+1) % 20 == 0:
                print(json.dumps({'phase': 'features', 'rows': i+1, 'total': len(population), 'seconds': time.monotonic()-started}), flush=True)
        return np.asarray(features), np.asarray(native, dtype=np.float64), identities
    x, native, identities = extract(rows)
    if len({r['query_token_sha256'] for r in identities}) != 204:
        raise ValueError('Duplicate actual response model inputs')
    np.save(a.private/'features.npy', x, allow_pickle=False); np.save(a.private/'native-logits.npy', native, allow_pickle=False)
    write_jsonl(a.study/'feature-inputs.jsonl', identities)
    def normalize(values):
        values = values.astype(np.float64)
        return values/np.sqrt(np.maximum(np.mean(values*values, axis=1, keepdims=True), 1e-12))
    x = normalize(x)
    train = np.array([i for i, r in enumerate(rows) if r['partition'] == 'train'])
    dev = np.array([i for i, r in enumerate(rows) if r['partition'] == 'dev'])
    if len(train) != 164 or len(dev) != 40:
        raise ValueError('Isolated response partition counts differ')
    gold = np.array([RESPONSE_LABELS.index(r['gold']) for r in rows]); dev_rows = [rows[i] for i in dev]
    selected = {'kind': 'native', 'development': summarize(dev_rows, prediction_rows(native[dev]))}; candidates = [selected]
    for c in (1e-5, 1e-4, 1e-3, .01, .1, 1.):
        estimator = LogisticRegression(C=c, class_weight='balanced', solver='lbfgs', max_iter=2000, tol=1e-9, random_state=1037)
        estimator.fit(x[train], gold[train])
        if not np.array_equal(estimator.classes_, np.arange(5)):
            raise ValueError('Unexpected response class order')
        path = a.private/('C-'+str(c)+'.npz'); np.savez(path, weights=estimator.coef_, bias=estimator.intercept_)
        restored = np.load(path, allow_pickle=False); z = x[dev]@restored['weights'].T+restored['bias']
        predictions = prediction_rows(z)
        probabilities = np.array([[r['probabilities'][label] for label in RESPONSE_LABELS] for r in predictions])
        error = float(np.max(np.abs(probabilities-estimator.predict_proba(x[dev]))))
        if error > 1e-10 or not np.array_equal(z.argmax(axis=1), estimator.predict(x[dev])):
            raise ValueError('Restored response head differs from fitted estimator')
        record = {'kind': 'logistic', 'C': c, 'converged': bool(np.all(estimator.n_iter_ < 2000)),
            'development': summarize(dev_rows, predictions), 'head_sha256': file_sha(path), 'maximum_export_probability_error': error}
        candidates.append(record)
        if record['converged'] and record['development']['selection_rank'] < selected['development']['selection_rank']:
            selected = record; (a.private/'selected.npz').write_bytes(path.read_bytes())
        print(json.dumps(record), flush=True)
    selection_path = a.study/'selection.json'
    selection_path.write_text(json.dumps({'complete': True, 'selected': selected, 'candidates': candidates,
        'features_sha256': file_sha(a.private/'features.npy'), 'native_logits_sha256': file_sha(a.private/'native-logits.npy'),
        'diagnostic_opened': False}, indent=2)+'\n'); selection_sha = file_sha(selection_path)
    if file_sha(a.data/'old-diagnostic.jsonl') != data['artifacts']['old-diagnostic']:
        raise ValueError('Original diagnostic data differs')
    diagnostic = read_jsonl(a.data/'old-diagnostic.jsonl')
    if any(r['context'] for r in diagnostic):
        raise ValueError('Expected isolated old diagnostic drafts')
    dx, dz, dids = extract(diagnostic); np.save(a.private/'diagnostic-features.npy', dx, allow_pickle=False)
    np.save(a.private/'diagnostic-native-logits.npy', dz, allow_pickle=False)
    write_jsonl(a.study/'diagnostic-feature-inputs.jsonl', dids)
    dx = normalize(dx); metrics = {}
    for name, population, values, base_logits in (('development', dev_rows, x[dev], native[dev]), ('old-diagnostic', diagnostic, dx, dz)):
        if selected['kind'] == 'native':
            z = base_logits
        else:
            head = np.load(a.private/'selected.npz', allow_pickle=False); z = values@head['weights'].T+head['bias']
        predictions = prediction_rows(z); metrics[name] = summarize(population, predictions)
        write_jsonl(a.study/(name+'-predictions.jsonl'), [{**r, 'prediction': pred, 'logits': v.tolist()}
            for r, pred, v in zip(population, predictions, z)])
    if file_sha(selection_path) != selection_sha:
        raise ValueError('Response selection changed after diagnostic access')
    report = {'complete': True, 'selected': selected, 'selection_sha256': selection_sha, 'metrics': metrics,
        'diagnostic_features_sha256': file_sha(a.private/'diagnostic-features.npy'),
        'diagnostic_native_logits_sha256': file_sha(a.private/'diagnostic-native-logits.npy'),
        'seconds': time.monotonic()-started, 'decision': 'REJECT' if any(m['errors'] for m in metrics.values()) else 'RESPONSE_ENGINEERING_PASS_ONLY',
        'draft_only': True, 'contextual_factual_support_established': False, 'independent_validation': False, 'stable_release_ready': False}
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
