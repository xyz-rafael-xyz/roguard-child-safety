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
ARMS = ('new-mask_wrong_assertion','new-mask_wrong_new_fit_semantics')
FIELDS = ('subject', 'time', 'assertion', 'source', 'relation', 'concern')
SIZES = (3, 3, 4, 3, 2, 2)
PROBES = ('probe_operator', 'probe_meaning', 'probe_combination', 'probe_both')
NEW_COUNTS = dict(fit=288, probe_operator=384, probe_meaning=384, probe_combination=96, probe_both=384, counterfoil=4)
V9_WEIGHTS = {'ro': '114d3e89191d06152cceccd080114998901a4f0bf6a9540de8e11dd8b6cf6b09',
              'uk': '0fee43a071d2c07c26e8b3d69cacb4933fd284bc64dbee0d18fe5481e1e4815f'}
PRIMARY, CONTROL = ARMS[1], ARMS[0]
SECONDARY = ()
GATE_CONTRACT = {'primary': 'new-mask_wrong_new_fit_semantics', 'control': 'new-mask_wrong_assertion', 'strict_assertion_partitions': ['probe_operator', 'probe_both'], 'strict_rows_per_language': 768, 'strict_references': ['new-mask_wrong_assertion', 'frozen125-primary', 'frozen125-control-full_teacher', 'unchanged-initial75', 'original-v9'], 'strict_frozen125_primary_assertion_errors': {'ro': 106, 'uk': 102}, 'probe_nonregression_partitions': ['probe_operator', 'probe_meaning', 'probe_combination', 'probe_both'], 'probe_nonregression_metrics': ['binary_errors', 'joint_errors', 'assertion_errors'], 'probe_nonregression_references': ['new-mask_wrong_assertion', 'frozen125-primary', 'frozen125-control-full_teacher'], 'original_dev_nonregression_metrics': ['binary_errors', 'joint_errors', 'introduced_v9'], 'original_dev_nonregression_references': ['new-mask_wrong_assertion', 'frozen125-primary', 'frozen125-control-full_teacher', 'unchanged-initial75'], 'fixed_original_dev_ceilings': {'ro': {'binary_errors': 12, 'joint_errors': 673, 'introduced_v9': 2}, 'uk': {'binary_errors': 1, 'joint_errors': 634, 'introduced_v9': 0}}, 'control_reproduction_required': True, 'control_reproduction_tolerance': 0.001, 'panel_rows': 40, 'panel_alias_rows': 32, 'panel_unique_supplemental_rows': 8, 'panel_binary_errors': 0, 'panel_joint_errors': 0, 'secondary_cannot_replace_primary': True, 'independent_human_validation': False, 'qualification': False}
BASELINE122_ROOT_SHA = '4ae9db78633a02bf362876da65d6e95b948f43d887999b71fbc3d5db680bf7af'
INITIAL75 = {'ro':'1c18db5e5fc26baf4c0739d053c0546338201d76fcb9061e47477c14650535e3',
             'uk':'466a1900655c85a6db809adb384d90fc634357a4f7b72f1415274c4787b3ecb7'}


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
    require(canonical_digest(read_json(path)) == canonical_digest(value), 'Saved JSON metadata differs')


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
            require(canonical_digest(labels) == canonical_digest(row['proposed_fields'])
                    and canonical_digest(gold) == canonical_digest(row['proposed_gold']), 'Extracted proposals changed')


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


def verify_fit_source(report,fit_plan,evaluation,fit):
    require(report['source_sha256']==fit_plan['source_sha256']==evaluation['trainer_source']['sha256']==sha(Path(fit)/'source.py'),
            'Retained fit source differs from registered trainer')


def verify_fit_inventory(fit,report):
    """Exactly114 produced artifacts plus the terminal report; no contradictory state."""
    fit=Path(fit)
    expected={'source.py','protocol.json','all-language-preflight.json','selected-before-probes.json','ro-paired-occurrences.json','uk-paired-occurrences.json'}
    for lang in LANGUAGES:
        for arm in ARMS:
            expected.add(lang+'/'+arm+'/assertion-keep.npy')
            for epoch in range(13):
                expected.add(lang+'/'+arm+'/epoch-'+str(epoch)+'.safetensors')
                expected.add(lang+'/'+arm+'/epoch-'+str(epoch)+'-dev.npy')
    require(len(expected)==114 and set(report['artifacts'])==expected,'Fit artifact declaration inventory differs')
    actual={str(p.relative_to(fit)) for p in fit.rglob('*') if p.is_file()}
    require(actual==expected|{'report.json'},'Fit output has missing/extra/failure artifacts')
    for name,digest in report['artifacts'].items():
        path=fit/name
        require(path.resolve().is_relative_to(fit.resolve()) and sha(path)==digest,'Retained fit artifact changed: '+name)


def verify_training_arrays(old,new,ox,oz,nx,nz,pairs,record,directory,arm):
    """Reconstruct origin-specific Nx7 masks without importing the loss/trainer."""
    require(arm in ARMS,'Unknown semantic-mask arm')
    train=[i for i,row in enumerate(old) if row['partition']=='train'];added=[p['new_index'] for p in pairs]
    labels=np.asarray([old[i]['fields'] for i in train]+[new[i]['fields'] for i in added],dtype=np.int32)
    gold=np.asarray([old[i]['gold'] for i in train]+[new[i]['gold'] for i in added],dtype=np.int32)
    features=np.concatenate((ox[train],nx[added]));teacher=np.concatenate((oz[train],nz[added]))
    for name,array in [('fields',labels),('gold',gold),('features',features),('teacher',teacher)]:
        require(raw_digest(array)==record['train_'+name+'_sha256'],'Ordered training '+name+' changed')
    expected=np.ones((len(labels),7),dtype=np.bool_);offset=0
    for j,width in enumerate(SIZES):
        agrees=teacher[:,offset:offset+width].argmax(1)==labels[:,j]
        if j==FIELDS.index('assertion'):expected[:,j]=agrees
        elif arm==PRIMARY:expected[len(train):,j]=agrees[len(train):]
        offset+=width
    mask_path=directory/'assertion-keep.npy';require(sha(mask_path)==record['mask_sha256'],'Mask bytes changed')
    mask=np.load(mask_path,allow_pickle=False)
    require(mask.dtype==np.bool_ and mask.shape==expected.shape and mask.tobytes()==expected.tobytes(),'Mask provenance/order differs')
    require(mask[:,-1].all(),'Legacy binary teacher must always remain')
    declarations=dict(mask_shape=list(expected.shape),mask_false=int((~expected).sum()),
        mask_false_by_term=(~expected).sum(0).tolist(),mask_false_original_by_term=(~expected[:len(train)]).sum(0).tolist(),
        mask_false_new_fit_by_term=(~expected[len(train):]).sum(0).tolist())
    for key,value in declarations.items():require(canonical_digest(record[key])==canonical_digest(value),'Mask summary differs: '+key)
    weights=[]
    for column,width in enumerate(SIZES):
        counts=np.bincount(labels[:,column],minlength=width);w=np.minimum(4.,np.sqrt(len(labels)/np.maximum(1,counts)/width));w/=np.mean(w[labels[:,column]]);weights.append(w.tolist())
    return dict(rows=len(labels),original_train_occurrences=len(train),new_fit_occurrences=len(added),
        **declarations,reconstructed_class_weights=weights,
        class_weight_evidence='Reconstruction from bound trainer source; no observed saved weight vector',
        ordered_features_sha256=raw_digest(features),ordered_teacher_sha256=raw_digest(teacher))


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
            require(bool(torch.isfinite(x).all()) and bool(torch.isfinite(norms).all())
                    and bool((norms > 0).all()), 'Invalid/nonfinite/zero head feature norm')
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


def verify_control_reproduction(current,prior,declared,spec,scorer):
    error=parity(current,prior,spec,scorer,tolerance=1e-3)
    require(type(declared['argmax_rows_differing']) is int and declared['argmax_rows_differing']==0
            and type(declared['fixed_threshold_decision_differences']) is int and declared['fixed_threshold_decision_differences']==0
            and type(declared['maximum_logit_error']) in (int,float)
            and declared['maximum_logit_error']==error,'Frozen125 control reproduction declaration differs')
    return error


def group_metrics(logits, rows, spec, scorer, references):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups['partition:' + row['partition']].append(i)
        for name in ('operator_family', 'meaning_family', 'paired_meaning_id', 'context_variant', 'panel_pair_id', 'contrast_group', 'register', 'core_type', 'block_id'):
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
            and retained.tobytes()==logits.tobytes(), 'Retained logits differ from projected output')
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
    require(sum(r['rows'] for r in replay_records) == 229762, 'Retained total differs')
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
    """All registered131 gates; no secondary or relaxed historical floors."""
    failures=[]
    for language in LANGUAGES:
        r=results[language];c=r['arms'][PRIMARY];b=r['arms'][CONTROL]
        frozen=r['frozen125_references'];probe_refs=[('selected_control',b),('frozen125-primary',frozen['primary']),('frozen125-control-full_teacher',frozen['full_control'])]
        for name,reference in probe_refs:
            for part in PROBES:
                cm,bm=c['groups']['partition:'+part],reference['groups']['partition:'+part]
                require(cm['rows']==bm['rows']==NEW_COUNTS[part],'Probe gate denominator differs')
                for key in ('binary_errors','joint_errors','assertion_errors'):
                    if cm[key]>bm[key]:failures.append(language+'/'+part+'/'+name+'/'+key)
        pooled=lambda a:sum(a['groups']['partition:'+p]['assertion_errors'] for p in ('probe_operator','probe_both'))
        for name,reference in probe_refs+[('initial75',r['new_baselines']['initial75']),('v9',r['new_baselines']['v9'])]:
            require(sum(reference['groups']['partition:'+p]['rows'] for p in ('probe_operator','probe_both'))==768,'Strict reference denominator differs')
            if pooled(c)>=pooled(reference):failures.append(language+'/no_strict_transfer_improvement/'+name)
        limit=GATE_CONTRACT['strict_frozen125_primary_assertion_errors'][language]
        require(pooled(frozen['primary'])==limit,'Frozen125106/102 reference changed')
        if pooled(c)>=limit:failures.append(language+'/no_strict_frozen125_floor_improvement')
        for name,reference in probe_refs+[('initial75',dict(dev=r['initial75_dev']))]:
            for key in ('binary_errors','joint_errors','introduced_v9'):
                if c['dev'][key]>reference['dev'][key]:failures.append(language+'/original_dev/'+name+'/'+key)
        for key,limit in GATE_CONTRACT['fixed_original_dev_ceilings'][language].items():
            if c['dev'][key]>limit:failures.append(language+'/fixed_original_dev_ceiling/'+key)
        panel=c['full_panel'];require(panel['rows']==20 and panel['bank_aliases']==16 and panel['supplemental_rows']==4,'Full40 panel denominator differs')
        for key in ('binary_errors','joint_errors'):
            if panel[key]!=0:failures.append(language+'/full_panel/'+key)
    return dict(escalation_allowed=not failures,failures=failures,primary_candidate=PRIMARY,primary_control=CONTROL,qualified=False)


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
            and audit.get('device') == 'cpu'
            and type(audit.get('token_rows_replayed')) is int and type(audit.get('head_rows_replayed')) is int
            and audit.get('token_rows_replayed') == audit.get('head_rows_replayed') == 3080,
            'Incomplete new-feature audit')
    for field, key in [('source_sha256', 'new_feature_auditor'), ('audit_plan_sha256', 'new_feature_audit_plan'),
                       ('extraction_plan_sha256', 'new_extraction_plan'), ('producer_report_sha256', 'new_feature_report')]:
        require(audit[field] == files[key]['sha256'], 'New-feature audit provenance differs: ' + field)
    for flag in ('encoder_loaded', 'native_encoder_independently_replayed', 'probe_accuracy_summaries_produced', 'training_performed', 'training_admitted'):
        require(audit.get(flag) is False, 'New-feature audit scope differs: ' + flag)
    require(audit['encoder_weights'] == V9_WEIGHTS, 'Frozen original-v9 teacher differs')
    for language in LANGUAGES:
        observation = audit['languages'][language]
        require(type(observation['rows']) is int and type(observation['native_observations_recounted']) is int
                and observation['rows'] == observation['native_observations_recounted'] == 1540
                and canonical_digest(observation['partitions']) == canonical_digest(NEW_COUNTS)
                and type(observation['seven_head_argmax_rows_differing']) is int
                and type(observation['binary_decision_differences']) is int
                and observation['seven_head_argmax_rows_differing'] == observation['binary_decision_differences'] == 0
                and type(observation['maximum_cpu_logit_error']) in (int, float)
                and np.isfinite(observation['maximum_cpu_logit_error']) and 0 <= observation['maximum_cpu_logit_error'] <= 1e-3,
                'New-feature audit language parity differs')
        require(observation.get('native_encoder_independently_replayed') is False,
                'Audit native replay scope differs')
        for key,limit in [('producer_native_maximum_recounted',1e-3),('producer_score_maximum_recounted',1e-10)]:
            value=observation[key]
            require(type(value) in (int,float) and np.isfinite(value) and 0<=value<=limit,
                    'Audited producer parity differs')
    expected = {language + '-' + filename: files[language + '_' + key]['sha256']
                for language in LANGUAGES for filename, key in [('rows.jsonl', 'new_rows'),
                    ('features.npy', 'new_features'), ('logits.npy', 'new_logits')]}
    require(audit['feature_artifacts'] == expected, 'New-feature audit does not bind actual fit inputs')
    require(audit['encoder_weights'] == {language: files[language + '_v9_weights']['sha256'] for language in LANGUAGES},
            'New-feature teacher differs')
    require(audit['full_panel_sha256'] == files['full_panel']['sha256']
            and audit['panel_alias_map_sha256'] == files['full_panel_alias_map']['sha256']
            and canonical_digest(audit['panel_alias_summary']) == canonical_digest(dict(
                full_panel_rows=40,main_bank_alias_rows=32,unique_supplemental_rows=8,
                all_panel_responses_represented=True,independent_extra_panel_examples=False)),
            'Audited full40 panel differs')
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
        require(all(canonical_digest(record['expected_values'].get(k)) == canonical_digest(v) for k, v in statuses.items()), 'Historical completion requirements omitted')
        require(set(record['artifact_bindings'].values()) == expected_bindings[role], 'Historical artifact coverage incomplete')
        for location, value in record['expected_values'].items():
            require(canonical_digest(pointer(document, location)) == canonical_digest(value), 'Historical evidence status differs: ' + role)
        for location, key in record['artifact_bindings'].items():
            require(pointer(document, location) == files[key]['sha256'], 'Historical artifact binding differs: ' + role)
    return audit


def verify_baseline122(fit_plan):
    """Bind the exact already-exposed baselines and independent root recount."""
    files=fit_plan['files']; path=lambda k:Path(files[k]['path'])
    root=read_json(path('baseline122_verification'))
    require(sha(path('baseline122_verification')) == BASELINE122_ROOT_SHA,
            'Independent122 evidence identity differs')
    require(root['verification']=='PASS' and root['rows_recounted']==6160
            and root['panel_responses_recounted']==80 and root['initial75_head_rows_independently_replayed']==3080
            and root['all_group_metrics_exact'] is True and root['all_panel_metrics_exact'] is True
            and root['qualified'] is False and root['training_performed'] is False,
            'Independent122 recount incomplete')
    report=read_json(path('baseline122_report')); plan=read_json(path('baseline122_plan'))
    require(root['producer_report_sha256']==files['baseline122_report']['sha256']
            and report['source_sha256']==files['baseline122_source']['sha256']
            and report['plan_sha256']==files['baseline122_plan']['sha256']
            and report['format']=='roguard-balanced-baseline-profile-1' and report['complete'] is True,
            'Completed122 provenance differs')
    for flag in ('encoder_loaded','training_performed','selection_performed','held_out_test_opened','training_admitted','qualified'):
        require(report[flag] is False,'Baseline122 scope differs: '+flag)
    require(report['eligibility']['strict_improvement_study_eligible'] is True,
            'No registered baseline headroom')
    require(plan['files']['feature_audit121']['sha256']==files['new_feature_audit']['sha256'],
            'Baseline122 feature audit differs')
    expected={f'{l}-{r}-{kind}' for l in LANGUAGES for r in ('original-v9','unchanged-initial75')
              for kind in ('logits.npy','results.jsonl')} | {
                  'original-v9-full40-panel.jsonl','unchanged-initial75-full40-panel.jsonl','registered-plan.json','source.py'}
    require(set(report['artifacts'])==expected,'Baseline122 artifact coverage incomplete')
    for filename,digest in report['artifacts'].items():
        require(files['baseline122_artifact:'+filename]['sha256']==digest,
                'Baseline122 artifact binding differs')
    for language,counts in [('ro',(343,357)),('uk',(274,296))]:
        require(files[language+'_initial_weights']['sha256']==INITIAL75[language],
                'Initial75 reference identity differs')
        parity_record=root['head_parity'][language]
        require(parity_record['rows']==1540 and parity_record['seven_head_argmax_differences']==0
                and parity_record['binary_decision_differences']==0 and 0<=parity_record['max_cpu_logit_error']<=1e-3,
                'Independent122 CPU parity differs')
        for reference,expected_errors in zip(('original-v9','unchanged-initial75'),counts):
            group=report['languages'][language][reference]['primary_transfer']['all']
            require(group['rows']==768 and group['assertion_errors']==expected_errors,
                    'Fixed122 reference or transfer population changed')
    return report


def verify_frozen125126(evaluation,fit_plan):
    files=fit_plan['files'];path=lambda k:Path(files[k]['path'])
    fit=read_json(path('frozen125_fit_report'));prior=read_json(path('frozen126_report'))
    require(fit['complete'] is True and fit['qualified'] is False and fit['plan_sha256']==sha(path('frozen125_plan')) and fit['source_sha256']==sha(path('frozen125_source')),'Frozen125 lineage differs')
    require(prior['verification']=='PASS' and prior['decision']=='REJECT_ESCALATION' and prior['qualified'] is False and prior['fit_report_sha256']==sha(path('frozen125_fit_report')) and prior['source_sha256']==sha(path('frozen126_source')) and prior['evaluation_plan_sha256']==sha(path('frozen126_plan')),'Frozen126 lineage/rejection differs')
    for language in LANGUAGES:
        primary=prior['results'][language]['arms']['new-mask_wrong_assertion']
        require(sum(primary['groups']['partition:'+p]['assertion_errors'] for p in ('probe_operator','probe_both'))==GATE_CONTRACT['strict_frozen125_primary_assertion_errors'][language],'Frozen125 primary transfer count differs')
        for oldarm in ('new-mask_wrong_assertion','control-full_teacher'):
            selected=fit['languages'][language]['arms'][oldarm]['selected']
            binding=evaluation['files']['frozen125:'+language+':'+oldarm+':selected-dev']
            require(binding['sha256']==selected['dev_logits_sha256'],'Frozen125 selectedDEV binding differs')
            binding=evaluation['files']['frozen126:'+language+':'+oldarm+':logits']
            require(binding['sha256']==prior['results'][language]['arms'][oldarm]['logits_sha256'],'Frozen126 comparator logits binding differs')
        epochs=fit['languages'][language]['arms'][CONTROL]['epochs']
        require(len(epochs)==13,'Frozen125 control epoch count differs')
        for i,record in enumerate(epochs):require(record['epoch']==i and files[language+'_frozen125_epoch_'+str(i)+'_dev']['sha256']==record['dev_logits_sha256'],'Frozen125 control trajectory binding differs')
    return fit,prior


def panel_indices(rows, panel, mapping, language):
    """All20 responses per language, including16 non-fitting main-bank aliases."""
    require(len(panel)==len(mapping['aliases'])==40,'Full panel source denominator differs')
    lookup={r['case_id']:i for i,r in enumerate(rows)}
    require(len(lookup)==len(rows),'Duplicate canonical identities')
    indices=[];records=[]
    for position,(original,alias) in enumerate(zip(panel,mapping['aliases'])):
        require(type(alias['panel_index']) is int and alias['panel_index']==position
                and alias['panel_example_id']==original['example_id'], 'Panel alias order differs')
        if original['language']!=language:continue
        index=lookup[alias['canonical_case_id']];row=rows[index]
        require(row['language']==language and alias['canonical_partition']==row['partition']
                and row['partition'] in ('probe_operator','probe_both','counterfoil'),
                'Panel alias includes fitting/invalid population')
        require(canonical_digest(row['fields'])==canonical_digest(alias['proposed_fields'])
                and canonical_digest(row['gold'])==canonical_digest(alias['proposed_gold'])
                and canonical_digest(row['gold'])==canonical_digest(original['proposed_binary'])
                and alias['labels_checked'] is True,'Panel alias labels differ')
        require(type(alias['canonical_index']) is int and alias['canonical_index']>=0,
                'Panel canonical index type differs')
        require(alias['alias_kind'] in ('EXACT_ROLE_PRESERVING_NORMALIZED_BANK_ALIAS',
                                      'UNIQUE_SUPPLEMENTAL_NONFITTING_CONTROL'), 'Unknown panel alias kind')
        bank=alias['alias_kind']=='EXACT_ROLE_PRESERVING_NORMALIZED_BANK_ALIAS'
        require(bank==(row['partition']!='counterfoil'),'Panel alias kind differs')
        indices.append(index);records.append(dict(panel_index=position,panel_example_id=original['example_id'],
            canonical_case_id=row['case_id'],canonical_language_index=index,
            partition=row['partition'],bank_alias=bank))
    require(len(indices)==len(set(indices))==20 and Counter(rows[i]['partition'] for i in indices)=={
        'probe_operator':8,'probe_both':8,'counterfoil':4},'Full panel response denominator differs')
    return indices,records


def panel_metric(logits,rows,spec,scorer,references,indices):
    require(len(indices)==20,'Full panel metric denominator differs')
    refs={name:(fields[indices],binary[indices]) for name,(fields,binary) in references.items()}
    result=metrics(logits[indices],[rows[i] for i in indices],spec,scorer,refs)
    result.update(bank_aliases=sum(rows[i]['partition']!='counterfoil' for i in indices),
                  supplemental_rows=sum(rows[i]['partition']=='counterfoil' for i in indices))
    return result


def compare_baseline_counts(groups,panel,retained,retained_panel):
    """Recount fixed122 comparator errors; CE was not a122 acceptance metric."""
    for partition in NEW_COUNTS:
        current=groups['partition:'+partition];original=retained['partition'][partition]
        for key in ('rows','binary_errors','joint_errors','assertion_errors'):
            require(type(original[key]) is int and current[key]==original[key],
                    'Fixed122 partition recount differs: '+partition+'/'+key)
        require(current['field_errors']==[original['field_errors'][name] for name in FIELDS],
                'Fixed122 field recount differs')
    for key in ('rows','binary_errors','joint_errors','assertion_errors','bank_aliases'):
        require(panel[key]==retained_panel[key],'Fixed122 full panel recount differs')
    require(panel['field_errors']==[retained_panel['field_errors'][name] for name in FIELDS],
            'Fixed122 panel field recount differs')


def load_initial_bound(path,expected_sha256,language):
    from safetensors.numpy import load
    raw=Path(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==expected_sha256,'Initial75 changed before projection')
    tensors=load(raw)
    return {name:tensors['heads.'+language+'.'+name] for name in ('first.weight','first.bias','last.weight','last.bias')}


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
    require(evaluation.get('format') == 'roguard-semantic-mask-head-evaluation-1'
            and evaluation.get('registered') is True and evaluation.get('held_out_test_opened') is False,
            'Evaluation not separately registered')
    require(sha(registration['path']) == registration['sha256'], 'Evaluation registration changed')
    require(evaluation.get('fit_producer_exited') is True and type(evaluation.get('fit_producer_exit_code')) is int and evaluation['fit_producer_exit_code']==0,'Completed producer lifecycle required')
    require(evaluation['source_sha256'] == sha(__file__), 'Executing evaluator differs')
    require(evaluation['replay_tolerance'] == 1e-3, 'Replay tolerance changed')
    require(not Path(output).exists(), 'Refusing overwrite')
    fit_binding = evaluation['fit_plan']
    require(sha(fit_binding['path']) == fit_binding['sha256'], 'Fit registration changed')
    fit_plan = read_json(fit_binding['path'])
    require(fit_plan.get('format') == 'roguard-new-fit-semantic-mask-study-1'
            and fit_plan.get('registered') is True and fit_plan.get('synthetic_pilot_admitted') is True,
            'Missing separate synthetic-pilot admission')
    require((fit_plan['epochs'], fit_plan['seed'], fit_plan['batch_size'], fit_plan['repeat']) == (12, 1011, 64, 32), 'Schedule changed')
    require(canonical_digest(fit_plan['gate_contract'])==canonical_digest(GATE_CONTRACT)
            and fit_plan['new_probe_accuracy_scored_before_registration'] is True
            and fit_plan['new_trained_probe_scoring_before_selection_freeze'] is False,
            'Prospective gates or prior baseline exposure differ')
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
    baseline122 = verify_baseline122(fit_plan)
    frozen125,frozen126=verify_frozen125126(evaluation,fit_plan)
    prospective = evaluation['prospective_design']
    verify_prospective_binding(evaluation, fit_plan)
    design = read_json(prospective['path'])
    require(canonical_digest(design['gate_contract'])==canonical_digest(GATE_CONTRACT),
            'Prospective primary gate differs')
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
    require(report.get('format')=='roguard-new-fit-semantic-mask-study-result-1'
            and canonical_digest(report['gate_contract'])==canonical_digest(GATE_CONTRACT),
            'Retained study identity/gate differs')
    require(set(report['languages']) == set(frozen) == set(LANGUAGES), 'Language inventory incomplete')
    verify_fit_inventory(fit,report)
    verify_fit_source(report,fit_plan,evaluation,fit)
    require(read_json(fit / 'protocol.json') == fit_plan, 'Retained fit protocol changed')
    snapshot = {report_path: evaluation['fit_report_sha256'], frozen_path: evaluation['selection_sha256'],
                Path(fit_binding['path']): fit_binding['sha256'], Path(prospective['path']): prospective['sha256']}
    path = lambda key: Path(fit_plan['files'][key]['path'])
    panel,alias_map=read_rows(path('full_panel')),read_json(path('full_panel_alias_map'))
    require(alias_map['full_panel_sha256']==fit_plan['files']['full_panel']['sha256'],
            'Panel source mapping binding differs')
    source_bytes=Path(__file__).read_bytes()
    require(hashlib.sha256(source_bytes).hexdigest()==evaluation['source_sha256'],'Evaluator source changed')
    plan_bytes=Path(registration['path']).read_bytes()
    require(hashlib.sha256(plan_bytes).hexdigest()==registration['sha256'],'Evaluation plan changed')
    snapshot.update({Path(record['path']):record['sha256'] for record in ledger.values()})
    snapshot[Path(__file__)]=evaluation['source_sha256']
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
        require(expected_fit == sorted(fit_plan['admitted_fit_case_ids'][language]) and len(expected_fit) == 288, 'Fit admission membership differs')
        pairs = match_occurrences(old, new)
        pairs_path = fit / (language + '-paired-occurrences.json')
        require(canonical_digest(read_json(pairs_path)) == canonical_digest(pairs) and len(pairs) == 9216, 'Paired occurrence order differs')
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
        result = dict(arms={}, initial75_dev=metrics(initial, dev_rows, spec, _score, refs),frozen125_references={})
        for name,oldarm in [('primary','new-mask_wrong_assertion'),('full_control','control-full_teacher')]:
            binding=evaluation['files']['frozen125:'+language+':'+oldarm+':selected-dev']
            retained=np.load(binding['path'],allow_pickle=False)
            require(retained.dtype==np.float32 and retained.shape==(len(dev_rows),19),'Frozen125 selectedDEV shape differs')
            recounted=metrics(retained,dev_rows,spec,_score,refs)
            require(canonical_digest(recounted)==canonical_digest(frozen126['results'][language]['arms'][oldarm]['dev']),'Frozen125 originalDEV metric recount differs')
            result['frozen125_references'][name]=dict(dev=recounted)
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
                reproduction_error=None
                if arm==CONTROL:
                    prior=np.load(path(language+'_frozen125_epoch_'+str(epoch)+'_dev'),allow_pickle=False)
                    reproduction_error=verify_control_reproduction(z,prior,declared['frozen125_control_reproduction'],spec,_score)
                m = metrics(z, dev_rows, spec, _score, refs)
                compare_retained(m, declared, epoch)
                require(declared['updates'] == epoch * ((data_proof['rows'] + 63) // 64)
                        and declared['presentations'] == epoch * data_proof['rows'], 'Training schedule counters differ')
                checked.append((metric_rank(m, epoch), declared, m, checkpoint))
                replay_records.append(dict(language=language, arm=arm, epoch=epoch, maximum_cpu_logit_error=error,
                                           checkpoint_sha256=declared['checkpoint_sha256'], rows=len(dev_rows),frozen125_control_reproduction_max_error=reproduction_error,
                                           frozen125_checkpoint_bytes_identical=(declared['checkpoint_sha256']==frozen125['languages'][language]['arms'][CONTROL]['epochs'][epoch]['checkpoint_sha256']) if arm==CONTROL else None,
                                           frozen125_dev_bytes_identical=(declared['dev_logits_sha256']==frozen125['languages'][language]['arms'][CONTROL]['epochs'][epoch]['dev_logits_sha256']) if arm==CONTROL else None))
            chosen = min(checked, key=lambda item: item[0])
            if arm==CONTROL:
                require(chosen[1]['epoch']==frozen125['languages'][language]['arms'][CONTROL]['selected']['epoch'],'Reproduced control selected different125 epoch')
            require(chosen[1] == record['selected'] == frozen[language][arm], 'Selected checkpoint differs from independent ORIGINALDEV reselection')
            selections[language][arm] = dict(epoch=chosen[1]['epoch'], checkpoint=str(chosen[3]),
                                            checkpoint_sha256=chosen[1]['checkpoint_sha256'])
            result['arms'][arm] = dict(dev=chosen[2], data_proof=data_proof)
        panel_ix,panel_records=panel_indices(new,panel,alias_map,language)
        cached[language] = (new, nx, nz, spec, v9spec, panel_ix, panel_records)
    # Mandatory all-four barrier. Nothing above projects any new feature row.
    verify_fit_inventory(fit,report)
    verify_selection_barrier(selections, replay_records)
    require(sha(frozen_path) == evaluation['selection_sha256'] and sha(report_path) == evaluation['fit_report_sha256'], 'Selection changed before new scoring')
    check_ledger(ledger)
    check_origins(fit_plan['project_root'], ledger)
    def score_new_stage():
        out = Path(output)
        out.mkdir(parents=True)
        for name,raw,expected in [('source.py',source_bytes,evaluation['source_sha256']),
                                  ('registered-plan.json',plan_bytes,registration['sha256'])]:
            with (out/name).open('xb') as retained:retained.write(raw)
            require(sha(out/name)==expected,'Retained evaluator snapshot differs')
            snapshot[out/name]=expected
        save_json(out / 'verified-selection-before-new-scoring.json', dict(
            fit_report_sha256=evaluation['fit_report_sha256'], selection_sha256=evaluation['selection_sha256'],
            selected=selections, original_dev_checkpoints_replayed=52, original_dev_rows_replayed=229762))
        selection_proof = out / 'verified-selection-before-new-scoring.json'
        snapshot[selection_proof] = sha(selection_proof)
        for language in LANGUAGES:
            rows, features, teacher, spec, v9spec, panel_ix, panel_records = cached[language]
            initial_replay = replay_head(load_initial_bound(path(language+'_initial_weights'),
                fit_plan['files'][language+'_initial_weights']['sha256'],language),features)
            initial_logits=np.load(path('baseline122_artifact:'+language+'-unchanged-initial75-logits.npy'),allow_pickle=False)
            teacher122=np.load(path('baseline122_artifact:'+language+'-original-v9-logits.npy'),allow_pickle=False)
            require(initial_logits.dtype==np.float32 and initial_logits.shape==(1540,19),
                    'Fixed122 initial baseline shape differs')
            baseline_error=parity(initial_replay,initial_logits,spec,_score)
            require(teacher122.dtype==teacher.dtype and teacher122.shape==teacher.shape
                    and teacher122.tobytes()==teacher.tobytes(),'Fixed122 original-v9 logits differ')
            refs = {'v9': interpretations(teacher, v9spec, _score), 'initial75': interpretations(initial_logits, spec, _score)}
            language_out = out / language
            language_out.mkdir()
            save_json(language_out / 'row-order.json', [dict(case_id=r['case_id'], partition=r['partition'],
                                                           source_row_sha256=r['source_row_sha256']) for r in rows])
            snapshot[language_out / 'row-order.json'] = sha(language_out / 'row-order.json')
            save_json(language_out/'panel-order.json',panel_records)
            snapshot[language_out/'panel-order.json']=sha(language_out/'panel-order.json')
            baseline_groups = {}
            for name, z, current_spec in [('v9', teacher, v9spec), ('initial75', initial_logits, spec)]:
                target = language_out / (name + '-all-new.npy')
                retained = retain_grouped_output(target, z, rows, current_spec, _score, refs)
                snapshot[target] = retained['logits_sha256']
                saved=np.load(target,allow_pickle=False)
                retained['full_panel']=panel_metric(saved,rows,current_spec,_score,refs,panel_ix)
                reference_name='original-v9' if name=='v9' else 'unchanged-initial75'
                compare_baseline_counts(retained['groups'],retained['full_panel'],
                    baseline122['languages'][language][reference_name],baseline122['panel'][reference_name][language])
                baseline_groups[name] = retained
            results[language]['new_baselines'] = baseline_groups
            results[language]['baseline122_initial75_replay_max_error']=baseline_error
            for refname,oldarm in [('primary','new-mask_wrong_assertion'),('full_control','control-full_teacher')]:
                binding=evaluation['files']['frozen126:'+language+':'+oldarm+':logits']
                z=np.load(binding['path'],allow_pickle=False)
                require(z.dtype==np.float32 and z.shape==(1540,19),'Frozen126 comparator logits shape differs')
                target=language_out/('frozen125-'+refname+'-all-new.npy')
                retained=retain_grouped_output(target,z,rows,spec,_score,refs);snapshot[target]=retained['logits_sha256']
                retained['full_panel']=panel_metric(np.load(target,allow_pickle=False),rows,spec,_score,refs,panel_ix)
                previous=frozen126['results'][language]['arms'][oldarm]
                require(canonical_digest(retained['groups'])==canonical_digest(previous['groups']) and canonical_digest(retained['full_panel'])==canonical_digest(previous['full_panel']),'Frozen126 comparator group/panel recount differs')
                results[language]['frozen125_references'][refname].update(retained)

            for arm in ARMS:
                chosen = selections[language][arm]
                z = project_bound_checkpoint(chosen['checkpoint'], chosen['checkpoint_sha256'], features)
                target = language_out / (arm + '-all-new.npy')
                retained = retain_grouped_output(target, z, rows, spec, _score, refs)
                snapshot[target] = retained['logits_sha256']
                retained['full_panel']=panel_metric(np.load(target,allow_pickle=False),rows,spec,_score,refs,panel_ix)
                results[language]['arms'][arm].update(retained)
                if arm==CONTROL:
                    old_control=np.load(language_out/'frozen125-primary-all-new.npy',allow_pickle=False)
                    seven=lambda value:np.column_stack([part.argmax(1) for part in np.split(value,[3,6,10,13,15,17],axis=1)])
                    results[language]['control_probe_reproduction_diagnostic']=dict(rows=len(rows),maximum_logit_difference=float(np.max(np.abs(z-old_control))),
                        logits_bytes_identical=z.tobytes()==old_control.tobytes(),seven_argmax_rows_differing=int(np.any(seven(z)!=seven(old_control),axis=1).sum()),
                        served_decision_differences=int(np.sum(interpretations(z,spec,_score)[1]!=interpretations(old_control,spec,_score)[1])),
                        acceptance_gate_changed=False)
        return out
    out = guarded_new_scoring(snapshot, selections, replay_records, score_new_stage)
    gate = primary_gate(results)
    require(all(sha(p) == expected for p, expected in snapshot.items()), 'Fit/source changed during independent evaluation')
    check_ledger(ledger)
    check_origins(fit_plan['project_root'], ledger)
    final = dict(format='roguard-semantic-mask-head-evaluation-1', verification='PASS',
        source_sha256=sha(__file__), evaluation_plan_sha256=registration['sha256'], fit_report_sha256=evaluation['fit_report_sha256'],
        selection_sha256=evaluation['selection_sha256'], selection_proof_sha256=sha(out / 'verified-selection-before-new-scoring.json'),
        new_feature_audit_sha256=sha(path('new_feature_audit')), new_feature_order={l: new_audit['languages'][l]['ordered_inputs_sha256'] for l in LANGUAGES},
        replay_records=replay_records, results=results, gate=gate, secondary_deltas=secondary_deltas(results),
        gelu_contract=gelu, baseline122_report_sha256=fit_plan['files']['baseline122_report']['sha256'],
        baseline122_verification_sha256=fit_plan['files']['baseline122_verification']['sha256'],
        panel_scope='All40 responses retained separately:32 main-bank probe aliases+8 unique supplements;3080 unique inputs, never3120 independent examples.',
        gates=GATE_CONTRACT, frozen126_report_sha256=fit_plan['files']['frozen126_report']['sha256'],
        control_reproduction_scope='All26 control oldDEV checkpoints, exact7argmax/served decisions, max1e-3 logits and same125 selected epochs.',
        exposed_diagnostics=True, secondary_promotion_permitted=False,
        decision='ALLOW_RESEARCH_ESCALATION' if gate['escalation_allowed'] else 'REJECT_ESCALATION',
        encoder_loaded=False, training_performed=False, held_out_test_opened=False, independently_annotated=False, qualified=False,
        limitations='Independent retained CPU head replay, metric reselection and data/mask proof; not optimizer-step replay, independent semantic gold or release qualification.')
    finalize_report(out,final,snapshot,ledger,fit_plan['project_root'],lambda:verify_fit_inventory(fit,report))
    return final


def finalize_report(out,final,snapshot,ledger,project_root,fit_inventory_check=lambda:None):
    path=out/'report.json'
    try:
        fit_inventory_check()
        save_json(path,final)
        fit_inventory_check()
        require(all(sha(p)==expected for p,expected in snapshot.items()),'Bound artifact changed after final report')
        check_ledger(ledger);check_origins(project_root,ledger)
        require(canonical_digest(read_json(path))==canonical_digest(final),'Final report changed at completion')
        require(all(sha(p)==expected for p,expected in snapshot.items()),'Final artifact snapshot differs')
        fit_inventory_check()
    except BaseException as error:
        if path.exists():path.replace(out/'rejected-final-report.json')
        failed=dict(final,verification='FAIL',decision='INTEGRITY_FAILURE',error_type=type(error).__name__,error=str(error),qualified=False)
        try:save_json(path,failed)
        except BaseException:
            if path.exists():path.replace(out/'unverified-failure-report.json')
        raise


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
