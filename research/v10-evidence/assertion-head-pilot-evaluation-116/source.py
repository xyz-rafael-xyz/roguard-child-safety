"""Private prospective postfit evaluator. No encoder, fitting or threshold search.

A registered evaluation plan must bind the completed fit report and selection
snapshot before invocation. Prototype code alone grants no experiment admission.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import io
import json
from pathlib import Path
import sys
import numpy as np

LANGUAGES = ('ro', 'uk')
ARMS = ('control-full_teacher', 'control-mask_wrong_assertion', 'new-full_teacher', 'new-mask_wrong_assertion')
FIELDS = ('subject', 'time', 'assertion', 'source', 'relation', 'concern')
SIZES = (3, 3, 4, 3, 2, 2)
PROBES = ('probe_operator', 'probe_meaning', 'probe_combination', 'probe_both')
NEW_COUNTS = dict(fit=48, probe_operator=32, probe_meaning=32, probe_combination=16, probe_both=16, counterfoil=9)
V9_WEIGHTS = {'ro': '114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09',
              'uk': '0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f'}
PRIMARY, CONTROL = 'new-mask_wrong_assertion', 'control-full_teacher'
SECONDARY = (('new-full_teacher', CONTROL), (PRIMARY, 'control-mask_wrong_assertion'),
             ('control-mask_wrong_assertion', CONTROL), (PRIMARY, 'new-full_teacher'))


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def save_json(path, value):
    with Path(path).open('x') as output:
        json.dump(value, output, indent=2, allow_nan=False)
        output.write('\n')


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')).hexdigest()


def raw_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def pointer(value, path):
    require(path.startswith('/'), 'JSON pointer must start with slash')
    for part in path[1:].split('/'):
        part = part.replace('~1', '/').replace('~0', '~')
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def check_ledger(ledger):
    require(isinstance(ledger, dict) and ledger, 'Empty artifact ledger')
    for record in ledger.values():
        require(sha(record['path']) == record['sha256'], 'Bound source/artifact changed: ' + record['path'])


def check_origins(root, ledger):
    root = Path(root).resolve()
    bound = {Path(r['path']).resolve() for r in ledger.values()}
    for name, module in list(sys.modules.items()):
        if name.split('.')[0] not in ('roguard', 'training', 'eval', 'scripts') or not getattr(module, '__file__', None):
            continue
        base = root / 'src' if name.startswith('roguard') else root
        expected = base.joinpath(*name.split('.'))
        actual = Path(module.__file__).resolve()
        require(actual in {expected.with_suffix('.py').resolve(), (expected / '__init__.py').resolve()}
                and actual in bound, 'Unbound project module origin: ' + name)


def validate_rows(rows, language, *, new):
    expected = NEW_COUNTS if new else dict(train=13174 if language == 'ro' else 15755,
                                         dev=4202 if language == 'ro' else 4635)
    require(dict(Counter(r['partition'] for r in rows)) == expected, 'Population differs')
    identities = set()
    for row in rows:
        require(row['language'] == language and isinstance(row['case_id'], str) and row['case_id']
                and row['case_id'] not in identities, 'Invalid row identity/language')
        identities.add(row['case_id'])
        labels, gold = row['fields'], row['gold']
        require(isinstance(labels, list) and len(labels) == 6 and all(type(v) is int and 0 <= v < s
                for v, s in zip(labels, SIZES)) and type(gold) is int and gold in (0, 1), 'Invalid field/gold')
        if new:
            require(labels == row['proposed_fields'] and gold == row['proposed_gold'], 'Extracted proposals changed')


def match_occurrences(old, new, repeat=32):
    """Independent reconstruction, deliberately does not import the trainer."""
    pools = defaultdict(list)
    for index, row in enumerate(old):
        if row['partition'] == 'train':
            pools[(row['language'], tuple(row['fields']), row['gold'])].append(index)
    for indices in pools.values():
        indices.sort(key=lambda i: old[i]['case_id'])
    offsets, pairs = Counter(), []
    for index, row in enumerate(new):
        if row['partition'] != 'fit':
            continue
        key = (row['language'], tuple(row['fields']), row['gold'])
        require(pools[key], 'Missing matched TRAIN parent')
        for _ in range(repeat):
            parent = pools[key][offsets[key] % len(pools[key])]
            offsets[key] += 1
            pairs.append(dict(occurrence=len(pairs), new_index=index, new_case_id=row['case_id'],
                              parent_index=parent, parent_case_id=old[parent]['case_id'], fields=row['fields'], gold=row['gold']))
    return pairs


def verify_training_arrays(old, new, ox, oz, nx, nz, pairs, record, directory, arm):
    train = [i for i, row in enumerate(old) if row['partition'] == 'train']
    added = [p['new_index'] for p in pairs]
    parents = [p['parent_index'] for p in pairs]
    labels = np.asarray([old[i]['fields'] for i in train] + [new[i]['fields'] for i in added], dtype=np.int32)
    gold = np.asarray([old[i]['gold'] for i in train] + [new[i]['gold'] for i in added], dtype=np.int32)
    control = arm.startswith('control-')
    features = np.concatenate((ox[train], ox[parents] if control else nx[added]))
    teacher = np.concatenate((oz[train], oz[parents] if control else nz[added]))
    for name, array in [('fields', labels), ('gold', gold), ('features', features), ('teacher', teacher)]:
        require(raw_digest(array) == record['train_' + name + '_sha256'], 'Ordered training ' + name + ' changed')
    expected = np.ones(len(train) + len(added), dtype=bool)
    if arm.endswith('mask_wrong_assertion'):
        expected = teacher[:, 6:10].argmax(1) == labels[:, 2]
    mask_path = directory / 'assertion-keep.npy'
    require(sha(mask_path) == record['mask_sha256'], 'Mask bytes changed')
    mask = np.load(mask_path, allow_pickle=False)
    require(mask.dtype == np.bool_ and mask.shape == expected.shape and np.array_equal(mask, expected), 'Mask provenance/order differs')
    require(record['mask_false'] == int((~expected).sum()), 'Mask count differs')
    weights = []
    for column, width in enumerate(SIZES):
        counts = np.bincount(labels[:, column], minlength=width)
        w = np.minimum(4., np.sqrt(len(labels) / np.maximum(1, counts) / width))
        w /= np.mean(w[labels[:, column]])
        weights.append(w.tolist())
    return dict(rows=len(labels), mask_false=int((~expected).sum()), reconstructed_class_weights=weights,
                class_weight_evidence="Reconstruction consistent with bound trainer source; trainer retains no observed weight vector",
                ordered_features_sha256=raw_digest(features), ordered_teacher_sha256=raw_digest(teacher))


def interpretations(logits, spec, scorer):
    require(logits.ndim == 2 and logits.shape[1] == 19 and len(logits) and np.isfinite(logits).all(), 'Invalid logits')
    fields = np.column_stack([p.argmax(1) for p in np.split(logits[:, :17], np.cumsum(SIZES)[:-1], axis=1)])
    binary = np.asarray([int(scorer(row, spec) >= spec['threshold']) for row in logits], dtype=np.int32)
    return fields, binary


def metrics(logits, rows, spec, scorer, references=None):
    fields = np.asarray([r['fields'] for r in rows], dtype=np.int32)
    gold = np.asarray([r['gold'] for r in rows], dtype=np.int32)
    require(len(rows) == len(logits), 'Metric denominator differs')
    predicted, binary = interpretations(logits, spec, scorer)
    errors = predicted != fields
    joint = errors.any(1)
    binary_wrong = binary != gold
    ce = []
    for j, part in enumerate(np.split(logits[:, :17].astype(np.float64), np.cumsum(SIZES)[:-1], axis=1)):
        exp = np.exp(part - part.max(1, keepdims=True))
        prob = exp / exp.sum(1, keepdims=True)
        ce.append(float(-np.log(np.maximum(prob[np.arange(len(rows)), fields[:, j]], 1e-30)).mean()))
    result = dict(rows=len(rows), binary_errors=int(binary_wrong.sum()), joint_errors=int(joint.sum()),
                  assertion_errors=int(errors[:, 2].sum()), field_errors=errors.sum(0).tolist(), mean_field_ce=float(np.mean(ce)))
    if references:
        result['references'] = {}
        for name, (rf, rb) in references.items():
            require(rf.shape == fields.shape and rb.shape == gold.shape, 'Reference denominator differs')
            fe, be = rf != fields, rb != gold
            result['references'][name] = dict(introduced_binary=int((binary_wrong & ~be).sum()),
                fixed_binary=int((~binary_wrong & be).sum()), introduced_joint=int((joint & ~fe.any(1)).sum()),
                fixed_joint=int((~joint & fe.any(1)).sum()), introduced_fields=(errors & ~fe).sum(0).tolist(),
                fixed_fields=(~errors & fe).sum(0).tolist())
        if 'v9' in references:
            result['introduced_v9'] = result['references']['v9']['introduced_binary']
            result['fixed_v9'] = result['references']['v9']['fixed_binary']
        if 'initial75' in references:
            result['introduced_initial75'] = result['references']['initial75']['introduced_binary']
    return result


def metric_rank(metric, epoch):
    return [metric['binary_errors'], metric['joint_errors'], metric['mean_field_ce'], epoch]


def compare_retained(metric, retained, epoch):
    for key in ('rows', 'binary_errors', 'joint_errors', 'field_errors', 'introduced_v9', 'fixed_v9', 'introduced_initial75'):
        require(metric[key] == retained[key], 'Retained DEV metric differs: ' + key)
    require(abs(metric['mean_field_ce'] - retained['mean_field_ce']) <= 1e-12, 'Retained DEV cross entropy differs')
    rank = metric_rank(metric, epoch)
    require(retained['rank'][:2] == rank[:2] and retained['rank'][3] == epoch
            and abs(retained['rank'][2] - rank[2]) <= 1e-12, 'Retained selection rank differs')


def replay_head(checkpoint, features, *, batch_size=256):
    """Independent Torch CPU head, matching the inspected exact-erf MLX GELU."""
    import torch
    from safetensors.numpy import load_file
    weights = checkpoint if isinstance(checkpoint, dict) else load_file(checkpoint)
    require(set(weights) == {'first.weight', 'first.bias', 'last.weight', 'last.bias'}, 'Checkpoint head inventory differs')
    hidden, width = weights['first.weight'].shape
    require(features.ndim == 2 and features.shape[1] == width == 2560 and hidden > 0, 'Head input dimensions differ')
    require(weights['first.bias'].shape == (hidden,) and weights['last.weight'].shape == (19, hidden)
            and weights['last.bias'].shape == (19,), 'Head tensor shapes differ')
    require(all(v.dtype == np.float32 and np.isfinite(v).all() for v in weights.values()), 'Invalid head tensors')
    torch.set_num_threads(2)
    t = {k: torch.from_numpy(v.copy()) for k, v in weights.items()}
    parts = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            x = torch.from_numpy(np.array(features[start:start + batch_size], dtype=np.float32, copy=True))
            norms = (x * x).sum(1, keepdim=True).sqrt()
            require(bool(torch.isfinite(x).all()) and bool((norms > 0).all()), 'Invalid/zero head feature')
            x = x / norms * width ** .5
            h = torch.nn.functional.linear(x, t['first.weight'], t['first.bias'])
            h = h * (1 + torch.erf(h / (2. ** .5))) / 2
            z = torch.nn.functional.linear(h, t['last.weight'], t['last.bias'])
            parts.append(z.numpy().copy())
    out = np.concatenate(parts)
    require(out.shape == (len(features), 19) and np.isfinite(out).all(), 'Invalid CPU replay output')
    return out



def project_bound_checkpoint(path, expected_sha256, features, *, projector=None):
    """Consume the exact verified bytes, not a path reopened after verification."""
    raw = Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha256,
            'Checkpoint changed immediately before projection')
    from safetensors.numpy import load
    tensors = load(raw)
    return (replay_head if projector is None else projector)(tensors, features)


def parity(replayed, retained, spec, scorer, tolerance=1e-3):
    require(replayed.shape == retained.shape and retained.ndim == 2 and np.isfinite(retained).all(), 'Replay shape/value differs')
    error = float(np.max(np.abs(replayed - retained)))
    require(error <= tolerance, 'CPU replay logit tolerance exceeded')
    # Include the raw binary head in addition to the six fields and served score.
    all_heads = lambda z: np.column_stack([p.argmax(1) for p in np.split(z, [3, 6, 10, 13, 15, 17], axis=1)])
    require(np.array_equal(all_heads(replayed), all_heads(retained)), 'CPU seven-head argmax differs')
    require(np.array_equal(interpretations(replayed, spec, scorer)[1], interpretations(retained, spec, scorer)[1]), 'CPU served decisions differ')
    return error


def group_metrics(logits, rows, spec, scorer, references):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups['partition:' + row['partition']].append(i)
        for name in ('operator_family', 'meaning_family', 'paired_meaning_id', 'context_variant', 'panel_pair_id', 'contrast_group'):
            if row.get(name):
                groups[name + ':' + row[name]].append(i)
        groups['assertion:' + row['proposed_attributes']['assertion']].append(i)
        if row.get('operator_family') and row.get('meaning_family'):
            groups['block:' + row['operator_family'] + '::' + row['meaning_family']].append(i)
    result = {}
    for name, indices in sorted(groups.items()):
        refs = {key: (f[indices], b[indices]) for key, (f, b) in references.items()}
        result[name] = metrics(logits[indices], [rows[i] for i in indices], spec, scorer, refs)
    require(sum(result['partition:' + p]['rows'] for p in NEW_COUNTS) == len(rows), 'Group denominators incomplete')
    return result




def retain_grouped_output(path, logits, rows, spec, scorer, references):
    """Retain exact projected logits, then independently rescore those bytes."""
    before = group_metrics(logits, rows, spec, scorer, references)
    with Path(path).open('xb') as output:
        np.save(output, logits, allow_pickle=False)
    raw = Path(path).read_bytes()
    retained = np.load(io.BytesIO(raw), allow_pickle=False)
    require(retained.dtype == logits.dtype and retained.shape == logits.shape
            and np.array_equal(retained, logits), 'Retained logits differ from projected output')
    after = group_metrics(retained, rows, spec, scorer, references)
    require(canonical_digest(before) == canonical_digest(after), 'Retained output rescore differs')
    return dict(logits_sha256=hashlib.sha256(raw).hexdigest(), groups=after)


def verify_selection_barrier(selections, replay_records):
    require(set(selections) == set(LANGUAGES) and all(set(selections[l]) == set(ARMS) for l in LANGUAGES),
            'Selection freeze missing a language/arm')
    expected = {(l, a, epoch) for l in LANGUAGES for a in ARMS for epoch in range(13)}
    actual = [(r['language'], r['arm'], r['epoch']) for r in replay_records]
    require(len(actual) == len(expected) and set(actual) == expected, 'Incomplete/duplicate retained checkpoint replay')
    for record in replay_records:
        require(record['rows'] == (4202 if record['language'] == 'ro' else 4635), 'Retained replay denominator differs')
    require(sum(r['rows'] for r in replay_records) == 459524, 'Retained total differs')
    return True



def guarded_new_scoring(snapshot, selections, replay_records, stage):
    """No baseline or candidate new-data projection may precede this barrier."""
    verify_selection_barrier(selections, replay_records)
    require(all(sha(path) == expected for path, expected in snapshot.items()),
            'Retained artifact changed before new-data scoring')
    for language in LANGUAGES:
        for arm in ARMS:
            selected = selections[language][arm]
            require(snapshot.get(Path(selected['checkpoint'])) == selected['checkpoint_sha256'],
                    'Selected checkpoint lacks replayed snapshot binding')
    return stage()


def verify_prospective_binding(evaluation, fit_plan):
    proposed = evaluation['prospective_design']
    bound = fit_plan['files']['prospective_design']
    require(proposed['sha256'] == fit_plan['prospective_design_sha256'] == bound['sha256']
            and Path(proposed['path']).resolve() == Path(bound['path']).resolve(),
            'Evaluation gate differs from fitted prospective registration')
    require(sha(proposed['path']) == proposed['sha256'], 'Prospective gate document changed')


def primary_gate(results):
    """The v3 primary is fixed; successful secondary arms cannot replace it."""
    failures = []
    for language in LANGUAGES:
        r = results[language]
        c, b = r['arms'][PRIMARY], r['arms'][CONTROL]
        for part in PROBES:
            cm, bm = c['groups']['partition:' + part], b['groups']['partition:' + part]
            require(cm['rows'] == bm['rows'] == NEW_COUNTS[part], 'Probe gate denominator differs')
            for key in ('binary_errors', 'joint_errors', 'assertion_errors'):
                if cm[key] > bm[key]:
                    failures.append(language + '/' + part + '/' + key)
        pooled = lambda arm: sum(arm['groups']['partition:' + p]['assertion_errors'] for p in ('probe_operator', 'probe_both'))
        if pooled(c) >= pooled(b):
            failures.append(language + '/no_strict_held_operator_improvement')
        for reference_name, ref in [('selected_control', b['dev']), ('initial75', r['initial75_dev'])]:
            for key in ('binary_errors', 'joint_errors', 'introduced_v9'):
                if c['dev'][key] > ref[key]:
                    failures.append(language + '/original_dev/' + reference_name + '/' + key)
        panel = c['groups']['partition:counterfoil']
        require(panel['rows'] == 9, 'Counterfoil denominator differs')
        for key in ('binary_errors', 'joint_errors'):
            if panel[key] != 0:
                failures.append(language + '/counterfoil/' + key)
    return dict(escalation_allowed=not failures, failures=failures,
                primary_candidate=PRIMARY, primary_control=CONTROL, qualified=False)


def secondary_deltas(results):
    output = {}
    for candidate, control in SECONDARY:
        output[candidate + '__vs__' + control] = {}
        for language in LANGUAGES:
            a, b = results[language]['arms'][candidate], results[language]['arms'][control]
            output[candidate + '__vs__' + control][language] = {
                part: {key: a['groups']['partition:' + part][key] - b['groups']['partition:' + part][key]
                       for key in ('binary_errors', 'joint_errors', 'assertion_errors')}
                for part in NEW_COUNTS}
    return output


def verify_feature_lineage(evaluation, fit_plan):
    files = fit_plan['files']
    audit = read_json(files['new_feature_audit']['path'])
    require(audit.get('format') == 'roguard-assertion-probe-features-audit-1'
            and audit.get('verification') == 'PASS' and audit.get('qualified') is False
            and audit.get('held_out_test_opened') is False
            and audit.get('token_rows_replayed') == audit.get('head_rows_replayed') == 306,
            'Incomplete new-feature audit')
    for field, key in [('source_sha256', 'new_feature_auditor'), ('audit_plan_sha256', 'new_feature_audit_plan'),
                       ('extraction_plan_sha256', 'new_extraction_plan'), ('producer_report_sha256', 'new_feature_report')]:
        require(audit[field] == files[key]['sha256'], 'New-feature audit provenance differs: ' + field)
    for flag in ('encoder_loaded', 'native_encoder_independently_replayed', 'probe_accuracy_summaries_produced', 'training_performed', 'training_admitted'):
        require(audit.get(flag) is False, 'New-feature audit scope differs: ' + flag)
    require(audit['encoder_weights'] == V9_WEIGHTS, 'Frozen original-v9 teacher differs')
    for language in LANGUAGES:
        observation = audit['languages'][language]
        require(observation['rows'] == observation['native_observations_recounted'] == 153
                and observation['partitions'] == NEW_COUNTS
                and observation['seven_head_argmax_rows_differing'] == observation['binary_decision_differences'] == 0
                and type(observation['maximum_cpu_logit_error']) in (int, float)
                and np.isfinite(observation['maximum_cpu_logit_error']) and 0 <= observation['maximum_cpu_logit_error'] <= 1e-3,
                'New-feature audit language parity differs')
    expected = {language + '-' + filename: files[language + '_' + key]['sha256']
                for language in LANGUAGES for filename, key in [('rows.jsonl', 'new_rows'),
                    ('features.npy', 'new_features'), ('logits.npy', 'new_logits')]}
    require(audit['feature_artifacts'] == expected, 'New-feature audit does not bind actual fit inputs')
    require(audit['encoder_weights'] == {language: files[language + '_v9_weights']['sha256'] for language in LANGUAGES},
            'New-feature teacher differs')
    lineage_records = evaluation['original_lineage']
    require(isinstance(lineage_records, list) and len(lineage_records) == 3, 'Historical lineage incomplete')
    lineage = {record['role']: record for record in lineage_records}
    require(set(lineage) == {'original88', 'audit89', 'warm90'}, 'Historical lineage roles incomplete')
    original_keys = {language + '_' + key for language in LANGUAGES for key in ('original_rows', 'original_features', 'original_logits')}
    initial_keys = {language + '_' + key for language in LANGUAGES for key in ('initial_weights', 'initial_dev_logits', 'initial_dev_rows')}
    expected_bindings = {'original88': original_keys, 'audit89': original_keys | {'original_feature_report'}, 'warm90': initial_keys}
    for role, record in lineage.items():
        document = read_json(files[record['file_key']]['path'])
        statuses = {'/verification': 'PASS', '/qualified': False} if role == 'audit89' else {'/complete': True, '/qualified': False}
        require(all(record['expected_values'].get(k) == v for k, v in statuses.items()), 'Historical completion requirements omitted')
        require(set(record['artifact_bindings'].values()) == expected_bindings[role], 'Historical artifact coverage incomplete')
        for location, value in record['expected_values'].items():
            require(pointer(document, location) == value, 'Historical evidence status differs: ' + role)
        for location, key in record['artifact_bindings'].items():
            require(pointer(document, location) == files[key]['sha256'], 'Historical artifact binding differs: ' + role)
    return audit


def verify_gelu_contract(contract):
    import inspect
    import mlx.nn as nn
    source = inspect.getsource(nn.gelu)
    source_path = Path(inspect.getsourcefile(nn.gelu)).resolve()
    require(str(source_path) == str(Path(contract['path']).resolve()) and sha(source_path) == contract['file_sha256']
            and hashlib.sha256(source.encode()).hexdigest() == contract['function_source_sha256'], 'Installed MLX GELU differs')
    require('mx.erf(x / math.sqrt(2))' in source, 'GELU oracle contract requires re-review')
    return dict(path=str(source_path), file_sha256=sha(source_path), function_source_sha256=contract['function_source_sha256'])


def load_initial_tensors(path, language):
    from safetensors import safe_open
    tensors = {}
    with safe_open(path, framework='np') as source:
        for name in ('first.weight', 'first.bias', 'last.weight', 'last.bias'):
            tensors[name] = source.get_tensor('heads.' + language + '.' + name)
    return tensors


def run(evaluation, output, *, registration):
    """Requires completed, hash-bound real fit artifacts; never fits a model."""
    require(evaluation.get('format') == 'roguard-assertion-head-pilot-evaluation-1'
            and evaluation.get('registered') is True and evaluation.get('held_out_test_opened') is False,
            'Evaluation not separately registered')
    require(sha(registration['path']) == registration['sha256'], 'Evaluation registration changed')
    require(evaluation['source_sha256'] == sha(__file__), 'Executing evaluator differs')
    require(evaluation['replay_tolerance'] == 1e-3, 'Replay tolerance changed')
    require(not Path(output).exists(), 'Refusing overwrite')
    fit_binding = evaluation['fit_plan']
    require(sha(fit_binding['path']) == fit_binding['sha256'], 'Fit registration changed')
    fit_plan = read_json(fit_binding['path'])
    require(fit_plan.get('format') == 'roguard-assertion-head-pilot-1'
            and fit_plan.get('registered') is True and fit_plan.get('synthetic_pilot_admitted') is True,
            'Missing separate synthetic-pilot admission')
    require((fit_plan['epochs'], fit_plan['seed'], fit_plan['batch_size'], fit_plan['repeat']) == (12, 1011, 64, 32), 'Schedule changed')
    ledger = {'fit:' + key: value for key, value in fit_plan['files'].items()}
    ledger.update({'evaluation:' + key: value for key, value in evaluation['files'].items()})
    ledger['evaluation:registration'] = registration
    ledger['evaluation:trainer_source'] = evaluation['trainer_source']
    require(evaluation['trainer_source']['sha256'] == fit_plan['source_sha256'], 'Historical trainer source differs')
    require(any(Path(r['path']).resolve() == Path(evaluation['gelu_contract']['path']).resolve()
                and r['sha256'] == evaluation['gelu_contract']['file_sha256'] for r in fit_plan['files'].values()),
            'Fit registration did not freeze the GELU implementation')
    check_ledger(ledger)
    gelu = verify_gelu_contract(evaluation['gelu_contract'])
    new_audit = verify_feature_lineage(evaluation, fit_plan)
    prospective = evaluation['prospective_design']
    verify_prospective_binding(evaluation, fit_plan)
    design = read_json(prospective['path'])
    require(design['format'] == 'roguard-assertion-head-pilot-proposal-3'
            and design['primary_comparison']['candidate'] == PRIMARY
            and design['primary_comparison']['control'] == CONTROL
            and tuple(map(tuple, design['secondary_comparisons'])) == SECONDARY, 'Primary/secondary contrasts differ')
    from roguard.gemma4_review import _score
    check_origins(fit_plan['project_root'], ledger)
    fit = Path(evaluation['fit_output'])
    report_path, frozen_path = fit / 'report.json', fit / 'selected-before-probes.json'
    require(sha(report_path) == evaluation['fit_report_sha256'] and sha(frozen_path) == evaluation['selection_sha256'], 'Fit/selection snapshot differs')
    report, frozen = read_json(report_path), read_json(frozen_path)
    require(report.get('complete') is True and report.get('fitting_complete') is True
            and report.get('probe_scoring_started') is False and report.get('qualified') is False
            and report['plan_sha256'] == fit_binding['sha256']
            and report['selection_sha256'] == evaluation['selection_sha256'], 'Incomplete fit or unbound selection')
    require(set(report['languages']) == set(frozen) == set(LANGUAGES), 'Language inventory incomplete')
    require(read_json(fit / 'protocol.json') == fit_plan, 'Retained fit protocol changed')
    snapshot = {report_path: evaluation['fit_report_sha256'], frozen_path: evaluation['selection_sha256'],
                Path(fit_binding['path']): fit_binding['sha256'], Path(prospective['path']): prospective['sha256']}
    path = lambda key: Path(fit_plan['files'][key]['path'])
    cached, results, selections, replay_records = {}, {}, {}, []
    for language in LANGUAGES:
        old, new = read_rows(path(language + '_original_rows')), read_rows(path(language + '_new_rows'))
        validate_rows(old, language, new=False)
        validate_rows(new, language, new=True)
        order = canonical_digest([[r['case_id'], r['source_row_sha256'], r['payload_sha256'], r['token_sha256']] for r in new])
        require(order == new_audit['languages'][language]['ordered_inputs_sha256'], 'New-feature audited row order differs')
        arrays = {}
        for key, shape in [('original_features', (len(old), 2560)), ('original_logits', (len(old), 19)),
                           ('new_features', (len(new), 2560)), ('new_logits', (len(new), 19))]:
            value = np.load(path(language + '_' + key), mmap_mode='r', allow_pickle=False)
            require(value.shape == shape and value.dtype == np.float32 and np.isfinite(value).all(), 'Feature array differs')
            arrays[key] = value
        ox, oz, nx, nz = (arrays[k] for k in ('original_features', 'original_logits', 'new_features', 'new_logits'))
        expected_fit = sorted(r['case_id'] for r in new if r['partition'] == 'fit')
        require(expected_fit == sorted(fit_plan['admitted_fit_case_ids'][language]) and len(expected_fit) == 48, 'Fit admission membership differs')
        pairs = match_occurrences(old, new)
        pairs_path = fit / (language + '-paired-occurrences.json')
        require(read_json(pairs_path) == pairs and len(pairs) == 1536, 'Paired occurrence order differs')
        snapshot[pairs_path] = sha(pairs_path)
        spec, v9spec = read_json(path(language + '_initial_spec')), read_json(path(language + '_v9_spec'))
        require(spec['language'] == language and spec['mode'] == 'coupled_policy'
                and spec['threshold'] == (.7517333793124337 if language == 'ro' else .9274248149759536)
                and spec['adapter_sha256'] == sha(path(language + '_initial_weights'))
                and v9spec['adapter_sha256'] == sha(path(language + '_v9_weights')), 'Fixed parent/spec differs')
        exclude = {'adapter_sha256', 'mode', 'threshold'}
        require({k: v for k, v in spec.items() if k not in exclude} == {k: v for k, v in v9spec.items() if k not in exclude}, 'Parent serving contract differs')
        indices = [i for i, row in enumerate(old) if row['partition'] == 'dev']
        dev_rows, dx = [old[i] for i in indices], ox[indices]
        warm_rows = read_rows(path(language + '_initial_dev_rows'))
        order_keys = ('case_id', 'language', 'fields', 'gold', 'token_sha256')
        require([[r[k] for k in order_keys] for r in dev_rows] == [[r[k] for k in order_keys] for r in warm_rows],
                'Original88 DEV order/labels differ from initial75 warm90 reference')
        initial = np.load(path(language + '_initial_dev_logits'), allow_pickle=False)
        refs = {'v9': interpretations(oz[indices], v9spec, _score), 'initial75': interpretations(initial, spec, _score)}
        require(initial.shape == (len(dev_rows), 19), 'Initial75 denominator differs')
        result = dict(arms={}, initial75_dev=metrics(initial, dev_rows, spec, _score, refs))
        results[language] = result
        require(set(report['languages'][language]['arms']) == set(frozen[language]) == set(ARMS), 'Arm inventory incomplete')
        selections[language] = {}
        for arm in ARMS:
            record = report['languages'][language]['arms'][arm]
            directory = fit / language / arm
            require(record['paired_occurrences_sha256'] == snapshot[pairs_path], 'Arm paired-map binding differs')
            data_proof = verify_training_arrays(old, new, ox, oz, nx, nz, pairs, record, directory, arm)
            snapshot[directory / 'assertion-keep.npy'] = record['mask_sha256']
            epochs = record['epochs']
            require(len(epochs) == 13 and [e['epoch'] for e in epochs] == list(range(13)), 'Retained epochs incomplete or reordered')
            checked = []
            for epoch, declared in enumerate(epochs):
                checkpoint = directory / ('epoch-' + str(epoch) + '.safetensors')
                logits_path = directory / ('epoch-' + str(epoch) + '-dev.npy')
                for artifact, key in [(checkpoint, 'checkpoint_sha256'), (logits_path, 'dev_logits_sha256')]:
                    require(sha(artifact) == declared[key], 'Retained checkpoint/logits changed')
                    snapshot[artifact] = declared[key]
                z = np.load(logits_path, allow_pickle=False)
                require(z.dtype == np.float32 and z.shape == (len(dev_rows), 19), 'Retained DEV logits invalid')
                replay = project_bound_checkpoint(checkpoint, declared['checkpoint_sha256'], dx)
                error = parity(replay, z, spec, _score)
                if epoch == 0:
                    parity(replay, initial, spec, _score)
                m = metrics(z, dev_rows, spec, _score, refs)
                compare_retained(m, declared, epoch)
                require(declared['updates'] == epoch * ((data_proof['rows'] + 63) // 64)
                        and declared['presentations'] == epoch * data_proof['rows'], 'Training schedule counters differ')
                checked.append((metric_rank(m, epoch), declared, m, checkpoint))
                replay_records.append(dict(language=language, arm=arm, epoch=epoch, maximum_cpu_logit_error=error,
                                           checkpoint_sha256=declared['checkpoint_sha256'], rows=len(dev_rows)))
            chosen = min(checked, key=lambda item: item[0])
            require(chosen[1] == record['selected'] == frozen[language][arm], 'Selected checkpoint differs from independent ORIGINALDEV reselection')
            selections[language][arm] = dict(epoch=chosen[1]['epoch'], checkpoint=str(chosen[3]),
                                            checkpoint_sha256=chosen[1]['checkpoint_sha256'])
            result['arms'][arm] = dict(dev=chosen[2], data_proof=data_proof)
        cached[language] = (new, nx, nz, spec, v9spec)
    # Mandatory all-eight barrier. Nothing above projects any new feature row.
    verify_selection_barrier(selections, replay_records)
    require(sha(frozen_path) == evaluation['selection_sha256'] and sha(report_path) == evaluation['fit_report_sha256'], 'Selection changed before new scoring')
    check_ledger(ledger)
    check_origins(fit_plan['project_root'], ledger)
    def score_new_stage():
        out = Path(output)
        out.mkdir(parents=True)
        save_json(out / 'verified-selection-before-new-scoring.json', dict(
            fit_report_sha256=evaluation['fit_report_sha256'], selection_sha256=evaluation['selection_sha256'],
            selected=selections, original_dev_checkpoints_replayed=104, original_dev_rows_replayed=459524))
        selection_proof = out / 'verified-selection-before-new-scoring.json'
        snapshot[selection_proof] = sha(selection_proof)
        for language in LANGUAGES:
            rows, features, teacher, spec, v9spec = cached[language]
            initial_logits = replay_head(load_initial_tensors(path(language + '_initial_weights'), language), features)
            refs = {'v9': interpretations(teacher, v9spec, _score), 'initial75': interpretations(initial_logits, spec, _score)}
            language_out = out / language
            language_out.mkdir()
            save_json(language_out / 'row-order.json', [dict(case_id=r['case_id'], partition=r['partition'],
                                                           source_row_sha256=r['source_row_sha256']) for r in rows])
            snapshot[language_out / 'row-order.json'] = sha(language_out / 'row-order.json')
            baseline_groups = {}
            for name, z, current_spec in [('v9', teacher, v9spec), ('initial75', initial_logits, spec)]:
                target = language_out / (name + '-all-new.npy')
                retained = retain_grouped_output(target, z, rows, current_spec, _score, refs)
                snapshot[target] = retained['logits_sha256']
                baseline_groups[name] = retained
            results[language]['new_baselines'] = baseline_groups
            for arm in ARMS:
                chosen = selections[language][arm]
                z = project_bound_checkpoint(chosen['checkpoint'], chosen['checkpoint_sha256'], features)
                target = language_out / (arm + '-all-new.npy')
                retained = retain_grouped_output(target, z, rows, spec, _score, refs)
                snapshot[target] = retained['logits_sha256']
                results[language]['arms'][arm].update(retained)
        return out
    out = guarded_new_scoring(snapshot, selections, replay_records, score_new_stage)
    gate = primary_gate(results)
    require(all(sha(p) == expected for p, expected in snapshot.items()), 'Fit/source changed during independent evaluation')
    check_ledger(ledger)
    check_origins(fit_plan['project_root'], ledger)
    final = dict(format='roguard-assertion-head-pilot-evaluation-1', verification='PASS',
        source_sha256=sha(__file__), evaluation_plan_sha256=registration['sha256'], fit_report_sha256=evaluation['fit_report_sha256'],
        selection_sha256=evaluation['selection_sha256'], selection_proof_sha256=sha(out / 'verified-selection-before-new-scoring.json'),
        new_feature_audit_sha256=sha(path('new_feature_audit')), new_feature_order={l: new_audit['languages'][l]['ordered_inputs_sha256'] for l in LANGUAGES},
        replay_records=replay_records, results=results, gate=gate, secondary_deltas=secondary_deltas(results),
        gelu_contract=gelu, repeated_control='Both explicit_truth_uncertainty siblings are repeated semantic controls, not independent challenges.',
        decision='ALLOW_RESEARCH_ESCALATION' if gate['escalation_allowed'] else 'REJECT_ESCALATION',
        encoder_loaded=False, training_performed=False, held_out_test_opened=False, independently_annotated=False, qualified=False,
        limitations='Independent retained CPU head replay, metric reselection and data/mask proof; not optimizer-step replay, independent semantic gold or release qualification.')
    save_json(out / 'report.json', final)
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--plan-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(sha(args.plan) == args.plan_sha256, 'Evaluation registration differs')
    plan = read_json(args.plan)
    result = run(plan, args.output, registration={'path': str(args.plan), 'sha256': args.plan_sha256})
    require(sha(args.plan) == args.plan_sha256, 'Evaluation registration changed')
    print(json.dumps({k: result[k] for k in ('verification', 'decision', 'qualified')}))


if __name__ == '__main__':
    main()
