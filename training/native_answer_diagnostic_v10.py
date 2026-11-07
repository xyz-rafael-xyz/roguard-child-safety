"""Project frozen query features through verified pretrained answer-letter rows."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import struct

from roguard.assurance import digest
from roguard.gemma4_contract import FILES, BASE_REVISION
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl
from roguard.trajectory_model import RESPONSE_LABELS
from training.probe_assertion_query_v10 import LABELS


def letter_matrix(base, token_ids):
    import numpy as np
    import mlx.core as mx
    path = base/'model.safetensors'
    with path.open('rb') as stream:
        count = struct.unpack('<Q', stream.read(8))[0]
        if count > 8_000_000:
            raise ValueError('Unexpected model tensor header size')
        header = json.loads(stream.read(count)); offset = 8+count
        def rows(suffix, dtype, width):
            info = header['language_model.model.embed_tokens.'+suffix]
            if info['dtype'] != dtype or info['shape'] != [262144, width]:
                raise ValueError('Unexpected tied embedding representation')
            itemsize = 4 if dtype == 'U32' else 2
            output = []
            for index in token_ids:
                if not 0 <= index < info['shape'][0]:
                    raise ValueError('Answer token outside vocabulary')
                stream.seek(offset+info['data_offsets'][0]+index*width*itemsize)
                raw = stream.read(width*itemsize)
                if len(raw) != width*itemsize:
                    raise ValueError('Incomplete embedding row')
                value = np.frombuffer(raw, dtype='<u4' if dtype == 'U32' else '<u2')
                output.append(value.copy() if dtype == 'U32' else (value.astype(np.uint32)<<16).view(np.float32))
            return np.stack(output)
        weight = rows('weight', 'U32', 320); scales = rows('scales', 'BF16', 40); biases = rows('biases', 'BF16', 40)
    unpacked = ((weight[:, :, None] >> (np.arange(8, dtype=np.uint32)*4)) & 15).reshape(len(token_ids), 2560).astype(np.float32)
    decoded = unpacked*np.repeat(scales, 64, axis=1)+np.repeat(biases, 64, axis=1)
    reference = mx.dequantize(mx.array(weight), mx.array(scales), mx.array(biases), group_size=64, bits=4, mode='affine')
    mx.eval(reference); maximum = float(np.max(np.abs(decoded-np.asarray(reference))))
    if maximum > 1e-7 or not np.isfinite(decoded).all():
        raise ValueError('Tied embedding row dequantization differs')
    return decoded, maximum


def probabilities(x, matrix):
    import numpy as np
    logits = 30*np.tanh((x@matrix.T).astype(np.float64)/30)
    exp = np.exp(logits-logits.max(axis=1, keepdims=True))
    return exp/exp.sum(axis=1, keepdims=True)


def summarize(rows):
    groups = defaultdict(lambda: {'total': 0, 'errors': 0, 'classes': {}})
    for row in rows:
        group = groups[row['language']+':'+row['partition']]
        group['total'] += 1; group['errors'] += row['predicted'] != row['gold']
        item = group['classes'].setdefault(row['gold'], {'total': 0, 'errors': 0})
        item['total'] += 1; item['errors'] += row['predicted'] != row['gold']
    return dict(groups)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'query_features', 'response_features', 'study', 'private'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    a = p.parse_args(); root = Path('research/v10-evidence')
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite native-answer diagnostic')
    if a.base.name != BASE_REVISION:
        raise ValueError('Unexpected base revision')
    for name, binding in FILES.items():
        if (a.base/name).stat().st_size != binding['bytes'] or file_sha(a.base/name) != binding['sha256']:
            raise ValueError('Pinned model asset differs: '+name)
    config = json.loads((a.base/'config.json').read_text())
    if config['text_config']['final_logit_softcapping'] != 30. or not config['text_config']['tie_word_embeddings'] or config['quantization'] != {'group_size': 64, 'bits': 4, 'mode': 'affine'}:
        raise ValueError('Unexpected original vocabulary head')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(a.base), trust_remote_code=False, local_files_only=True)
    ids = [tokenizer.encode(c, add_special_tokens=False) for c in 'ABCDE']
    if any(len(v) != 1 for v in ids):
        raise ValueError('Expected single-token answer letters')
    import numpy as np
    matrix, row_error = letter_matrix(a.base, [v[0] for v in ids])
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    np.save(a.private/'answer-letter-rows.npy', matrix, allow_pickle=False)
    query_study = root/'query-features-24'; response_study = root/'neural-response-26'
    qreport = json.loads((query_study/'report.json').read_text()); rreport = json.loads((response_study/'report.json').read_text())
    if not qreport['complete'] or not rreport['complete']:
        raise ValueError('Incomplete source features')
    qprotocol = json.loads((query_study/'protocol.json').read_text()); rprotocol = json.loads((response_study/'protocol.json').read_text())
    protocol = {'format': 'roguard-native-answer-diagnostic-1', 'source_sha256': file_sha(Path(__file__)),
        'amendment_sha256': file_sha(root/'protocol-amendment-29.json'), 'base_revision': BASE_REVISION,
        'base_weight_sha256': FILES['model.safetensors']['sha256'], 'answer_token_ids': dict(zip('ABCDE', [v[0] for v in ids])),
        'answer_row_sha256': file_sha(a.private/'answer-letter-rows.npy'),
        'query_report_sha256': file_sha(query_study/'report.json'), 'response_report_sha256': file_sha(response_study/'report.json'),
        'assertion_prompt': qprotocol['prompt'], 'response_prompt': rprotocol['prompt'],
        'fitting_performed': False, 'calibration_or_regression_loaded': False, 'stable_release_ready': False}
    (a.study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    query_rows = []; query_lookup = {}
    for lang in ('ro', 'uk'):
        for suffix, expected in qreport['languages'][lang]['artifacts'].items():
            if file_sha(a.query_features/(lang+'-'+suffix)) != expected:
                raise ValueError('Query feature binding differs')
        metadata = read_jsonl(a.query_features/(lang+'-rows.jsonl'))
        x = np.load(a.query_features/(lang+'-features.npy'), mmap_mode='r')
        for start in range(0, len(metadata), 512):
            values = probabilities(x[start:start+512, 2560:], matrix[:4])
            for row, pvalues in zip(metadata[start:start+512], values):
                record = {'case_id': row['case_id'], 'language': lang, 'partition': row['partition'],
                    'gold': LABELS[row['fields'][2]], 'predicted': LABELS[int(pvalues.argmax())],
                    'probabilities': pvalues.tolist(), 'token_sha256': row['token_sha256']}
                query_rows.append(record); query_lookup[(lang, row['token_sha256'])] = record
    probe = root/'assertion-query-22'; probe_protocol = json.loads((probe/'protocol.json').read_text())
    if qprotocol['prompt'] != probe_protocol['prompts']['outer-claim-scope']:
        raise ValueError('Probe and extracted query prompts differ')
    maximum_probe_error = 0.; mismatches = []; checked = 0
    for row in read_jsonl(probe/'predictions.jsonl'):
        if row['prompt'] != 'outer-claim-scope':
            continue
        actual = query_lookup[(row['language'], row['original_token_sha256'])]
        maximum_probe_error = max(maximum_probe_error, float(np.max(np.abs(np.array(actual['probabilities'])-row['conditional_probabilities']))))
        if actual['predicted'] != row['predicted']:
            mismatches.append(row['case_id'])
        checked += 1
    write_jsonl(a.study/'assertion-predictions.jsonl', query_rows)
    selection = json.loads((response_study/'selection.json').read_text())
    if selection['features_sha256'] != file_sha(a.response_features/'features.npy'):
        raise ValueError('Response feature hash differs')
    response_data = root/'response-curriculum-25'
    rows = read_jsonl(response_data/'train.jsonl')+read_jsonl(response_data/'dev.jsonl')
    identities = read_jsonl(response_study/'feature-inputs.jsonl'); rx = np.load(a.response_features/'features.npy', allow_pickle=False)
    if rx.shape != (len(rows), 2560) or len(identities) != len(rows) or len(rows) != 420:
        raise ValueError('Response TRAIN/DEV inventory differs')
    response = []
    for row, identity, pvalues in zip(rows, identities, probabilities(rx, matrix)):
        turns = ([{'role': 'minor', 'text': row['context']}] if row['context'] else [])+[{'role': 'assistant', 'text': row['draft']}]
        if identity['case_id'] != row['case_id'] or identity['partition'] != row['partition'] or identity['input_sha256'] != digest({'language': row['language'], 'turns': turns}):
            raise ValueError('Response effective input identity differs')
        response.append({'case_id': row['case_id'], 'language': row['language'], 'partition': row['partition'],
            'gold': row['gold'], 'predicted': RESPONSE_LABELS[int(pvalues.argmax())], 'probabilities': pvalues.tolist()})
    write_jsonl(a.study/'response-predictions.jsonl', response)
    parity = checked == 256 and not mismatches and maximum_probe_error <= .001
    result = {'complete': True, 'source_reports_verified': True, 'dequantization_maximum_error': row_error,
        'probe_rows_checked': checked, 'probe_prediction_mismatches': mismatches,
        'probe_maximum_probability_error': maximum_probe_error, 'projection_parity_pass': parity,
        'assertion': summarize(query_rows), 'response': summarize(response),
        'decision': 'DIAGNOSTIC_ONLY' if parity else 'INVALID_PROJECTION_PARITY',
        'fitting_performed': False, 'independent_test': False, 'stable_release_ready': False}
    (a.study/'report.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
