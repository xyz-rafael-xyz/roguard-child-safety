"""Count-matched input views for the prospective wording comparison."""
from roguard.assurance import digest
from roguard.trajectory_data import file_sha, read_jsonl


def addition_metadata(runtime):
    return [{'case_id': r['case_id'], 'language': r['language'], 'partition': 'train',
             'group': 'assertion_variation_77', 'gold': r['gold'], 'fields': r['fields'],
             'token_sha256': r['token_sha256'], 'pair_id': r['pair_id'],
             'parent_feature_index': r['parent']['feature_index'],
             'parent_case_id': r['parent']['case_id'], 'parent_token_sha256': r['parent']['token_sha256'],
             'factors': r['factors']} for r in runtime]


def paired_rows(original, additions):
    rows = list(original); indices = []; seen = {row['case_id'] for row in original}
    for added in additions:
        i = added['parent_feature_index']
        if type(i) is not int or not 0 <= i < len(original):
            raise ValueError('Invalid parent feature index')
        parent = original[i]
        if (parent['partition'] != 'train' or added['partition'] != 'train'
                or added['parent_case_id'] != parent['case_id']
                or added['parent_token_sha256'] != parent['token_sha256']
                or any(added[k] != parent[k] for k in ('language', 'gold', 'fields'))):
            raise ValueError('Added example does not match an original TRAIN parent')
        identity = 'matched79:'+added['pair_id']
        if identity in seen: raise ValueError('Duplicate paired identity')
        seen.add(identity); indices.append(i)
        rows.append({'case_id': identity, 'language': added['language'], 'partition': 'train',
                     'group': 'assertion_variation_77', 'gold': added['gold'], 'fields': added['fields'],
                     'pair_id': added['pair_id'], 'parent_feature_index': i,
                     'treatment_token_sha256': added['token_sha256'],
                     'control_token_sha256': parent['token_sha256']})
    return rows, indices


def arm_arrays(old_x, old_z, added_x, added_z, indices, arm):
    import numpy as np
    if arm not in ('matched-control', 'wording-treatment'):
        raise ValueError('Unknown registered arm')
    if (old_x.ndim != 2 or old_z.ndim != 2 or len(old_x) != len(old_z)
            or added_x.shape != (len(indices), old_x.shape[1])
            or added_z.shape != (len(indices), old_z.shape[1])
            or any(type(i) is not int or not 0 <= i < len(old_x) for i in indices)
            or any(v.dtype != np.float32 or not np.isfinite(v).all() for v in (old_x, old_z, added_x, added_z))):
        raise ValueError('Malformed matched feature arrays')
    tail_x, tail_z = (old_x[indices], old_z[indices]) if arm == 'matched-control' else (added_x, added_z)
    return np.concatenate((old_x, tail_x)), np.concatenate((old_z, tail_z))


def schedule(rows, epochs=12, batch_size=64):
    count = sum(r['partition'] == 'train' for r in rows)
    if not count or any(r['partition'] not in ('train', 'dev') for r in rows):
        raise ValueError('Invalid training partitions')
    return {'training_rows': count, 'development_rows': len(rows)-count,
            'updates_per_arm': epochs*((count+batch_size-1)//batch_size),
            'presentations_per_arm': epochs*count, 'paired_rows_sha256': digest(rows)}


def validate_bindings(args, extraction, plan, *, ready):
    import json
    import numpy as np
    expected = [{'name': name, 'learning_rate': .0001, 'teacher_kl': .2,
                 'policy_ce': {'ro': .25, 'uk': 1.}, 'dropout': {'ro': .1, 'uk': 0.}}
                for name in ('matched-control', 'wording-treatment')]
    if (plan['registered_candidates'] != expected or plan['epochs'] != 12 or plan['seed'] != 1011
            or plan['batch_size'] != 64 or plan['device'] != 'cpu' or plan['policy_loss_epsilon'] != 1e-6):
        raise ValueError('Registered matched configuration differs')
    for base, mapping in ((args.root, plan['public']), (args.private_root, plan['private'])):
        for name, expected_hash in mapping.items():
            if file_sha(base/name) != expected_hash: raise ValueError('Bound source changed: '+name)
    runtime = read_jsonl(args.private_root/'assertion-variation-77/curated/runtime-inputs.jsonl')
    schedules = {}
    if extraction.get('complete') is not True or set(extraction['languages']) != {'ro', 'uk'}:
        raise ValueError('Original features are incomplete')
    for lang in ('ro', 'uk'):
        original = read_jsonl(args.features/(lang+'-rows.jsonl')); expected_lang = plan['languages'][lang]
        for suffix, expected_hash in extraction['languages'][lang]['artifacts'].items():
            if file_sha(args.features/(lang+'-'+suffix)) != expected_hash:
                raise ValueError('Original feature artifact differs')
        x = np.load(args.features/(lang+'-features.npy'), mmap_mode='r', allow_pickle=False)
        z = np.load(args.features/(lang+'-logits.npy'), mmap_mode='r', allow_pickle=False)
        if (x.shape != (len(original), 2560) or z.shape != (len(original), 19)
                or x.dtype != np.float32 or z.dtype != np.float32 or not np.isfinite(x).all() or not np.isfinite(z).all()):
            raise ValueError('Malformed original feature arrays')
        additions = addition_metadata([r for r in runtime if r['language'] == lang])
        rows, _ = paired_rows(original, additions); schedules[lang] = schedule(rows)
        if (schedules[lang]['training_rows'] != expected_lang['training_rows']
                or schedules[lang]['development_rows'] != expected_lang['development_rows']
                or len(additions) != expected_lang['additional_rows']
                or schedules[lang]['updates_per_arm'] != expected_lang['updates_per_arm']
                or schedules[lang]['presentations_per_arm'] != expected_lang['presentations_per_arm']):
            raise ValueError('Registered schedule differs')
        if (file_sha(args.candidate/'actual'/(lang+'-selected.safetensors')) != expected_lang['v9_weights_sha256']
                or file_sha(args.feature_study/'v9-binding'/(lang+'-review.json')) != expected_lang['v9_spec_sha256']):
            raise ValueError('Original initialization or policy specification differs')
    result = {'public': plan['public'], 'private': plan['private'], 'schedules': schedules,
              'feature_report_sha256': file_sha(args.feature_study/'report.json'), 'future_features_verified': False}
    if not ready: return result
    study = args.root/'research/v10-evidence/assertion-variation-features-78'
    data = args.private_root/'assertion-variation-features-78'
    proof_path = args.private_root/plan['feature_verification']
    proof = json.loads(proof_path.read_text()); report = json.loads((study/'report.json').read_text())
    if (proof.get('verification') != 'PASS' or proof.get('encoder_loaded') is not False
            or proof.get('source_sha256') != file_sha(args.root/'eval/verify_assertion_features_v10.py')
            or proof.get('report_sha256') != file_sha(study/'report.json')
            or proof.get('protocol_sha256') != file_sha(study/'protocol.json')
            or proof.get('plan_sha256') != file_sha(args.root/'research/v10-evidence/protocol-amendment-68.json')
            or proof.get('runtime_inputs_sha256') != file_sha(args.private_root/'assertion-variation-77/curated/runtime-inputs.jsonl')
            or report.get('complete') is not True):
        raise ValueError('Completed feature audit does not bind current78 outputs')
    actual_artifacts = {}
    for lang in ('ro', 'uk'):
        for name, checksum in report['languages'][lang]['artifacts'].items():
            if file_sha(data/name) != checksum or proof['artifacts'].get(name) != checksum:
                raise ValueError('Audited future feature artifact differs')
            actual_artifacts[name] = checksum
        saved = read_jsonl(data/(lang+'-rows.jsonl'))
        if saved != addition_metadata([r for r in runtime if r['language'] == lang]):
            raise ValueError('Actual future feature order or metadata differs')
    if actual_artifacts != proof['artifacts']:
        raise ValueError('Feature audit inventory differs')
    result.update(future_features_verified=True, added_feature_report_sha256=file_sha(study/'report.json'),
                  added_feature_protocol_sha256=file_sha(study/'protocol.json'),
                  added_feature_verification_sha256=file_sha(proof_path), added_feature_artifacts=actual_artifacts)
    return result
