"""Freeze completed runtime references under a preregistered attention protocol."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from roguard.trajectory_data import file_sha

OPTIMIZATION = dict(epochs=3, batch_size=8, seed=1073, learning_rate=1e-5,
    head_learning_rate=1e-4, weight_decay=.01, gradient_norm_clip=1., teacher_kl=.2,
    policy_ce={'ro': .25, 'uk': 1.}, head_dropout={'ro': .1, 'uk': 0.})
POPULATIONS = {'ro': {'train': 13174, 'dev': 4202}, 'uk': {'train': 15755, 'dev': 4635}}


def read(path):
    return json.loads(Path(path).read_text())


def input_args(root, private_root):
    e = root/'research/v10-evidence'
    return SimpleNamespace(root=root, private_root=private_root,
        input_study=e/'large-review-input-recovery-72', inputs=private_root/'large-review-input-recovery-72/actual',
        feature_study=e/'crossed-review-features-62', features=private_root/'crossed-review-features-62',
        initial_study=e/'dropout-readout-75', initial_private=private_root/'dropout-readout-75',
        candidate=root/'research/v9-stabilization/reported-tail-fit-196')


def paths(root):
    e = root/'research/v10-evidence'
    return (e/'protocol-amendment-75.json', e/'runtime-attention-registration-91/binding.json',
            e/'runtime-attention-preflight-91/report.json')


def population(rows):
    counts = {lang: {split: sum(r['partition'] == split for r in values) for split in ('train', 'dev')}
              for lang, values in rows.items()}
    if counts != POPULATIONS or any(sum(counts[lang].values()) != len(values) for lang, values in rows.items()):
        raise ValueError('Registered attention populations differ')
    return counts, {lang: dict(updates_per_epoch=(value['train']+7)//8,
        total_updates=3*((value['train']+7)//8), total_presentations=3*value['train']) for lang, value in counts.items()}


def static_inputs(a):
    registration = read(a.registration)
    if registration['source_sha256'] != file_sha(__file__) or registration['optimization'] != OPTIMIZATION:
        raise ValueError('Registration source or optimization changed')
    for base, mapping in ((a.root, registration['public']), (a.private_root, registration['private'])):
        for name, expected in mapping.items():
            if file_sha(base/name) != expected:
                raise ValueError('Preregistered source or input changed: '+name)
    from training.train_large_review_attention_v10 import load_inputs
    rows, original = load_inputs(input_args(a.root, a.private_root))
    counts, schedules = population(rows)
    if (original != registration['original_input_bindings'] or counts != registration['populations']
            or schedules != registration['schedules']):
        raise ValueError('Preregistered original input bindings differ')
    return registration, rows


def compose(registration, rows, bindings, warm, registration_sha256):
    counts, schedules = population(rows)
    if (registration['optimization'] != OPTIMIZATION or counts != registration['populations']
            or schedules != registration['schedules']
            or bindings['original'] != registration['original_input_bindings']
            or bindings['runtime_helpers'] != registration['runtime_helpers']
            or warm.get('complete') is not True
            or warm.get('decision') != 'WARM_START_REFERENCE_ONLY_NO_MODEL_QUALIFICATION'):
        raise ValueError('Completed inputs do not match the preregistration')
    observations = {}
    for lang in ('ro', 'uk'):
        record = warm['languages'][lang]
        current, previous = record['runtime75'], record['runtime71']
        if (current['rows'] != counts[lang]['dev'] or previous['rows'] != counts[lang]['dev']
                or any(type(current[key]) is not int or not 0 <= current[key] <= current['rows']
                       or type(previous[key]) is not int or not 0 <= previous[key] <= previous['rows']
                       or current[key] > previous[key] for key in ('binary_errors', 'joint_argmax_errors'))):
            raise ValueError('Selected75 is worse than71 on the audited runtime references')
        observations[lang] = {key: record[key] for key in ('original75', 'runtime75', 'runtime71',
            'runtime75_vs_original75', 'runtime75_vs_runtime71', 'historical_logit_difference')}
    return dict(format='roguard-runtime-attention-completed-input-registration-1', experiment='runtime-review-attention-91',
        source_sha256=registration['trainer_source_sha256'], registration_source_sha256=registration['source_sha256'],
        preregistration_sha256=registration_sha256, optimization=registration['optimization'],
        input_bindings=bindings, populations=counts, schedules=schedules, dependency_pid=registration['dependency_pid'],
        completion_audit_sources=registration['completion_audit_sources'], warm_start_observation=observations,
        held_out_test_opened=False, qualified=False,
        scope='Freeze actual completed88/89/90 artifact hashes before optimizer updates. Retain selected75 fixed initialization '
              'only if its runtime binary and joint errors are no worse than71 in each language. This is not a qualification pass.')


def completed_plan(a):
    registration, _ = static_inputs(a)
    from training.train_runtime_review_attention_v10 import load_inputs
    rows, bindings = load_inputs(input_args(a.root, a.private_root))
    warm_path = a.root/'research/v10-evidence/runtime-warm-start-90/report.json'
    if file_sha(warm_path) != bindings['runtime']['warm_report_sha256']:
        raise ValueError('Warm-start report changed during registration')
    result = compose(registration, rows, bindings, read(warm_path), file_sha(a.registration))
    if static_inputs(a)[0] != registration:
        raise ValueError('Registration inputs changed')
    return result


def verify_completed_registration(a, *, require_preflight):
    amendment, proof, preflight = paths(a.root)
    expected = completed_plan(a)
    if read(amendment) != expected:
        raise ValueError('Completed-input registration changed')
    record = read(proof)
    if record != dict(complete=True, amendment_sha256=file_sha(amendment), preregistration_sha256=file_sha(a.registration),
                      source_sha256=file_sha(__file__), encoder_loaded=False, training_launched=False, qualified=False):
        raise ValueError('Registration binding proof differs')
    if require_preflight:
        p = read(preflight)
        if (p.get('complete') is not True or p['source_sha256'] != expected['source_sha256']
                or p['amendment_sha256'] != file_sha(amendment) or p['bindings'] != expected['input_bindings']
                or p['schedules'] != expected['schedules'] or p['populations'] != expected['populations']
                or any(p.get(k) is not False for k in ('encoder_loaded', 'training_launched', 'held_out_test_opened', 'qualified'))):
            raise ValueError('Completed-input training preflight differs')
    return file_sha(amendment)


def write_new(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        stream.write(json.dumps(value, indent=2)+'\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private_root', 'registration'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--preflight-report', type=Path)
    mode.add_argument('--verify-registration', action='store_true')
    a = p.parse_args()
    if a.preflight_report:
        registration, rows = static_inputs(a)
        counts, schedules = population(rows)
        write_new(a.preflight_report, dict(complete=True, preregistration_sha256=file_sha(a.registration),
            source_sha256=file_sha(__file__), populations=counts, schedules=schedules,
            original_input_bindings=registration['original_input_bindings'], future_runtime_inputs_opened=False,
            encoder_loaded=False, training_launched=False, qualified=False))
        print(json.dumps(dict(static_preflight=True, actual_runtime_registration=False))); return
    if a.verify_registration:
        checksum = verify_completed_registration(a, require_preflight=True)
        print(json.dumps(dict(registration_verified=True, amendment_sha256=checksum))); return
    amendment, proof, preflight = paths(a.root)
    if any(path.exists() for path in (amendment, proof, preflight,
            a.root/'research/v10-evidence/runtime-review-attention-91', a.private_root/'runtime-review-attention-91')):
        raise ValueError('Refusing to overwrite runtime attention registration or fit')
    registration = read(a.registration)
    from scripts.wait_completed_sequence_exit_v10 import require_exit
    require_exit(read(a.root/'research/v10-evidence/runtime-warm-start-sequence-90/report.json'), registration['dependency_pid'])
    expected = completed_plan(a)
    write_new(amendment, expected)
    write_new(proof, dict(complete=True, amendment_sha256=file_sha(amendment), preregistration_sha256=file_sha(a.registration),
        source_sha256=file_sha(__file__), encoder_loaded=False, training_launched=False, qualified=False))
    verify_completed_registration(a, require_preflight=False)
    print(json.dumps(dict(registration_complete=True, amendment_sha256=file_sha(amendment), encoder_loaded=False)))


if __name__ == '__main__':
    main()
