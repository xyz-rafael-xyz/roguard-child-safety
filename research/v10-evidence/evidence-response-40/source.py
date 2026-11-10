"""Evidence-first generated response judgments under a fixed output contract."""
import argparse
import json
from pathlib import Path
import time
import unicodedata

from roguard.assurance import digest
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from roguard.trajectory_model import RESPONSE_LABELS
from training.train_neural_response_v10 import PROMPT as CATEGORY_PROMPT


PROMPT = CATEGORY_PROMPT.replace('Output one letter:', 'The category codes are:') + (
    ' Return exactly one JSON object with these keys in this order: evidence, behavior, category. '
    'The evidence is a short exact quote copied from the proposed reply, preserving its original '
    'language and characters. For category A, quote the relevant respectful or uncertain wording. '
    'The behavior is one brief sentence describing what that phrase asks or claims; do not give '
    'a reasoning transcript. The category is one letter A, B, C, D or E. No markdown or other text.')


def parse_output(raw, draft):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate output key')
            result[key] = value
        return result
    try:
        result = json.loads(raw, object_pairs_hook=unique)
        if not isinstance(result, dict) or set(result) != {'evidence', 'behavior', 'category'}:
            raise ValueError('Unexpected output fields')
        if any(not isinstance(value, str) or not value.strip() for value in result.values()):
            raise ValueError('Expected nonempty output strings')
        draft = unicodedata.normalize('NFC', draft)
        if result['evidence'] not in draft:
            raise ValueError('Evidence is not an exact draft span')
        if result['category'] not in tuple('ABCDE'):
            raise ValueError('Unknown category')
        start = draft.index(result['evidence'])
        return {'valid': True, 'label': RESPONSE_LABELS['ABCDE'.index(result['category'])],
                'evidence_start': start, 'evidence_end': start+len(result['evidence']),
                'behavior': result['behavior'], 'behavior_verified': False, 'qualified': False}
    except (ValueError, TypeError) as error:
        return {'valid': False, 'label': None, 'failure': str(error), 'qualified': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'data', 'study'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists():
        raise ValueError('Refusing to overwrite generated response comparison')
    manifest_path = Path('research/v9-stabilization/qwen-control-87/model-manifest.json')
    manifest = json.loads(manifest_path.read_text())
    if manifest['revision'] != '938d8919941c6e7efd3c7150eff7fe9d12afa631':
        raise ValueError('Unexpected model revision')
    for name, record in manifest['files'].items():
        path = a.base/name
        if path.stat().st_size != record['bytes'] or file_sha(path) != record['sha256']:
            raise ValueError('Pinned model file differs')
    data = json.loads((a.data/'report.json').read_text())
    if file_sha(a.data/'dev.jsonl') != data['artifacts']['dev']:
        raise ValueError('Fixed development drafts differ')
    rows = [r for r in read_jsonl(a.data/'dev.jsonl') if not r['context']]
    if len(rows) != 40 or len({(r['language'], r['draft']) for r in rows}) != 40:
        raise ValueError('Expected every distinct development draft exactly once')
    a.study.mkdir(parents=True)
    protocol = {'format': 'roguard-evidence-response-diagnostic-1', 'source_sha256': file_sha(Path(__file__)),
        'amendment_sha256': file_sha('research/v10-evidence/protocol-amendment-38.json'),
        'base_manifest_sha256': file_sha(manifest_path), 'data_report_sha256': file_sha(a.data/'report.json'),
        'prompt': PROMPT, 'decoding': {'temperature': 0., 'max_tokens': 256, 'enable_thinking': False},
        'draft_only': True, 'fitting_performed': False, 'independent_validation': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import mlx.core as mx
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler
    mx.set_memory_limit(10*1024**3); mx.set_cache_limit(128*1024**2)
    engine, tokenizer = load(str(a.base), tokenizer_config={'trust_remote_code': False})
    engine.set_dtype(mx.float32); model = engine.language_model; started = time.monotonic()
    def evaluate(population, name):
        predictions = []; groups = {}
        for i, row in enumerate(population):
            draft = unicodedata.normalize('NFC', row['draft'])
            messages = [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': json.dumps(
                {'prior_turns': [], 'focal_message': {'role': 'assistant', 'text': draft}}, ensure_ascii=False)}]
            tokens = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False,
                                                   add_generation_prompt=True, enable_thinking=False)
            if not tokens or len(tokens) > 1024:
                raise ValueError('Unsupported generated-judgment input')
            tick = time.perf_counter()
            raw = generate(model, tokenizer, tokens, max_tokens=256, sampler=make_sampler(temp=0.), verbose=False)
            result = parse_output(raw, draft)
            correct = result['valid'] and result['label'] == row['gold']
            record = {**row, 'query_token_sha256': digest(tokens), 'raw_generation': raw,
                'prediction': result, 'correct': correct, 'seconds': time.perf_counter()-tick}
            predictions.append(record); group = groups.setdefault(row['language']+':'+row['gold'], {'rows': 0, 'errors': 0, 'invalid': 0})
            group['rows'] += 1; group['errors'] += not correct; group['invalid'] += not result['valid']
            write_jsonl(a.study/(name+'-predictions.jsonl'), predictions); mx.clear_cache()
            print(json.dumps({'population': name, 'done': i+1, 'total': len(population),
                              'correct': correct, 'valid': result['valid'], 'seconds': record['seconds']}), flush=True)
        return {'rows': len(population), 'correct': sum(r['correct'] for r in predictions),
            'errors': sum(not r['correct'] for r in predictions), 'invalid': sum(not r['prediction']['valid'] for r in predictions), 'groups': groups}
    metrics = {'development': evaluate(rows, 'development')}
    if metrics['development']['errors'] == 0:
        if file_sha(a.data/'old-diagnostic.jsonl') != data['artifacts']['old-diagnostic']:
            raise ValueError('Original diagnostic binding differs')
        diagnostic = read_jsonl(a.data/'old-diagnostic.jsonl')
        if any(r['context'] for r in diagnostic):
            raise ValueError('Expected isolated diagnostic drafts')
        metrics['old-diagnostic'] = evaluate(diagnostic, 'old-diagnostic')
    report = {'complete': True, 'metrics': metrics, 'seconds': time.monotonic()-started,
        'decision': 'REJECT' if any(m['errors'] for m in metrics.values()) else 'RESPONSE_ENGINEERING_PASS_ONLY',
        'original_diagnostic_opened': 'old-diagnostic' in metrics, 'draft_only': True,
        'contextual_factual_support_established': False, 'independent_validation': False, 'stable_release_ready': False}
    (a.study/'report.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
