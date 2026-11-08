"""Bind completed native references to the paired role-diversity attention fit."""
import json

from roguard.trajectory_data import file_sha, read_jsonl
from training.matched_role_diversity_v10 import matched_population


OPTIMIZATION = dict(epochs=3, batch_size=8, seed=1073, learning_rate=1e-5,
    head_learning_rate=1e-4, weight_decay=.01, gradient_norm_clip=1., teacher_kl=.2,
    policy_ce={'ro': .25, 'uk': 1.}, head_dropout={'ro': .1, 'uk': 0.})
POPULATIONS = {'ro': {'base_train': 13174, 'added_train': 1584, 'train': 14758, 'dev': 4202},
               'uk': {'base_train': 15755, 'added_train': 1584, 'train': 17339, 'dev': 4635}}
AUDIT_PLAN = 'research/v10-evidence/role-diversity-runtime-audit-103/plan.json'
AUDIT_PROOF = 'role-diversity-runtime102-completion-verification-for103.json'
AUDIT_QUEUE = 'role-diversity-runtime-audit-sequence-103-plan.json'
AUDIT_SEQUENCE = 'research/v10-evidence/role-diversity-runtime-audit-sequence-103/report.json'


def read(path):
    return json.loads(path.read_text())


def require_audit(proof, report, sequence):
    """Reject unfinished dependencies before importing numerical libraries."""
    from eval.verify_role_diversity_runtime_v10 import require_complete
    require_complete(report)
    if (proof.get('verification') != 'PASS' or proof.get('artifact_count') != 16
            or proof.get('cpu_head_rows') != 6336 or proof.get('development_rows_compared') != 8837
            or proof.get('ordered_progress_events') != 154 or proof.get('device') != 'cpu'
            or any(proof.get(k) is not False for k in ('encoder_loaded', 'training_performed',
                                                       'training_admitted', 'held_out_test_opened', 'qualified'))
            or sequence.get('execution_complete') is not True or sequence.get('phase') != 'finished'
            or not sequence.get('completed_steps')
            or any(s.get('exit_code') != 0 for s in sequence['completed_steps'])):
        raise ValueError('Completed reference audit103 is required before paired fitting')


def validate_initial_weights(original, initial, language):
    import numpy as np
    allowed = {f'base.model.layers.{i}.self_attn.{p}.lora_{part}'
               for i in range(24, 42) for p in ('q_proj', 'o_proj') for part in ('a', 'b')}
    allowed |= {f'heads.{language}.{layer}.{part}' for layer in ('first', 'last') for part in ('weight', 'bias')}
    if set(original) != set(initial) or not allowed <= set(initial):
        raise ValueError('Selected95 tensor inventory differs')
    for name, values in initial.items():
        if values.shape != original[name].shape or values.dtype != np.float32 or not np.isfinite(values).all():
            raise ValueError('Selected95 tensor shape, dtype or values differ')
        if name not in allowed and not np.array_equal(values, original[name]):
            raise ValueError('Selected95 changes a frozen tensor')
    return sorted(allowed)


def completed_inputs(root, private):
    """Load only TRAIN/DEV, requiring both completed reference audits first."""
    evidence = root/'research/v10-evidence'
    reference = evidence/'role-diversity-runtime-102'
    proof_path = private/AUDIT_PROOF
    proof, report, sequence = read(proof_path), read(reference/'report.json'), read(root/AUDIT_SEQUENCE)
    require_audit(proof, report, sequence)
    from scripts.wait_completed_sequence_exit_v10 import require_exit
    from training.role_diversity_runtime_v10 import check_bindings, load_registered, validate_array
    plan_path = root/AUDIT_PLAN; audit_plan = read(plan_path)
    queue_path = private/AUDIT_QUEUE; queue = read(queue_path)
    queue_record = read(evidence/'role-diversity-runtime-audit-103/queue.json')
    if (proof['source_sha256'] != file_sha(root/'eval/verify_role_diversity_runtime_v10.py')
            or proof['audit_plan_sha256'] != file_sha(plan_path)
            or proof['report_sha256'] != file_sha(reference/'report.json')
            or proof['protocol_sha256'] != file_sha(reference/'protocol.json')
            or proof['sequence_sha256'] != file_sha(root/audit_plan['sequence'])
            or proof['log_sha256'] != file_sha(private/audit_plan['log'])
            or sequence['plan_sha256'] != file_sha(queue_path)
            or sequence['source_hashes'] != queue['source_hashes']
            or [s['name'] for s in sequence['completed_steps']] != [s['name'] for s in queue['steps']]
            or queue_record['queue_plan_sha256'] != file_sha(queue_path)):
        raise ValueError('Reference audit binding differs')
    require_exit(sequence, queue_record['queue_pid'])
    check_bindings(root, private, audit_plan)
    for name, sha in queue['source_hashes'].items():
        if file_sha(root/name) != sha:
            raise ValueError('Reference audit source changed: '+name)
    for name, sha in proof['artifacts'].items():
        if file_sha(private/'role-diversity-runtime-102'/name) != sha:
            raise ValueError('Audited reference artifact changed: '+name)
    if len(proof['artifacts']) != 16:
        raise ValueError('Audited reference inventory differs')
    extraction_plan = root/audit_plan['extraction_plan']
    if proof['extraction_plan_sha256'] != file_sha(extraction_plan):
        raise ValueError('Audited extraction plan differs')
    _, new = load_registered(root, private, extraction_plan, proof['extraction_plan_sha256'])
    from training.runtime_attention_continuation_v10 import runtime_inputs
    from training.large_review_attention_v10 import validate_rows, training_schedule
    import numpy as np
    from safetensors.numpy import load_file
    baseline = runtime_inputs(root, private)
    recovered = read_jsonl(private/'large-review-input-recovery-72/actual/inputs.jsonl')
    metadata = {lang: read_jsonl(private/'runtime-review-baseline-88'/(lang+'-rows.jsonl'))
                for lang in ('ro', 'uk')}
    base = validate_rows(recovered, metadata)
    languages, matching, schedules, initial_inventory = {}, {}, {}, {}
    for lang in ('ro', 'uk'):
        inputs = new[lang]
        if [r for r in base[lang] if r['partition'] == 'dev'] != inputs['dev']:
            raise ValueError('Paired development inputs/order differ from audited selected95 references')
        result = matched_population(base[lang], inputs['train'], lang, seed=OPTIMIZATION['seed'])
        summary = result['summary']; counts = POPULATIONS[lang]
        if (summary['base_train_rows'] != counts['base_train'] or summary['added_train_slots'] != counts['added_train']
                or summary['train_rows_per_arm'] != counts['train'] or summary['development_rows'] != counts['dev']):
            raise ValueError('Paired population differs from the prospective design')
        matching[lang] = summary
        schedules[lang] = training_schedule(result['control'], OPTIMIZATION['epochs'], OPTIMIZATION['batch_size'])
        initial_inventory[lang] = validate_initial_weights(load_file(inputs['weights']['v9']),
                                                         load_file(inputs['weights']['selected95']), lang)
        arrays = {}
        for name, path, shape in (
                ('base_teacher', private/'runtime-review-baseline-88'/(lang+'-logits.npy'), (len(base[lang]), 19)),
                ('new_teacher', private/'role-diversity-runtime-102'/(lang+'-v9-logits.npy'), (1584, 19)),
                ('new_initial_features', private/'role-diversity-runtime-102'/(lang+'-selected95-features.npy'), (1584, 2560)),
                ('new_initial_logits', private/'role-diversity-runtime-102'/(lang+'-selected95-logits.npy'), (1584, 19)),
                ('initial_dev_logits', private/'role-diversity-runtime-102'/(lang+'-selected95-dev-logits.npy'), (counts['dev'], 19))):
            value = np.load(path, mmap_mode='r', allow_pickle=False); validate_array(value, shape); arrays[name] = value
        languages[lang] = dict(**result, **arrays, base=base[lang], new=inputs['train'],
                                weights=inputs['weights'], specs=inputs['specs'])
    bindings = dict(baseline=baseline, audit103_sha256=file_sha(proof_path),
        audit103_plan_sha256=file_sha(plan_path), audit103_queue_sha256=file_sha(queue_path),
        audit103_sequence_sha256=file_sha(root/AUDIT_SEQUENCE), reference102_artifacts=proof['artifacts'],
        extraction_plan_sha256=file_sha(extraction_plan), matching=matching, schedules=schedules,
        populations=POPULATIONS, trainable_inventory=initial_inventory)
    return languages, bindings


def aligned_teacher(inputs, arm):
    """Map each occurrence to its own unchanged original-v9 teacher output."""
    import numpy as np
    if arm not in ('control', 'treatment'):
        raise ValueError('Unknown matched arm')
    references = inputs['references'][arm]
    if len(references) != len(inputs[arm]):
        raise ValueError('Teacher occurrence count differs')
    pieces = []
    for source, index in references:
        if source not in ('base', 'new') or type(index) is not int or index < 0:
            raise ValueError('Invalid teacher source reference')
        array = inputs['base_teacher'] if source == 'base' else inputs['new_teacher']
        if index >= len(array):
            raise ValueError('Teacher reference out of bounds')
        pieces.append(array[index])
    values = np.stack(pieces)
    if values.shape != (len(inputs[arm]), 19) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError('Aligned teacher outputs differ')
    return values
