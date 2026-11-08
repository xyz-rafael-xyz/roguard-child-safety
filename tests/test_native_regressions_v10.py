"""Regression admission, complete denominators and fresh serving-path wiring."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from eval import native_regression_inputs_v10 as admission
from eval import evaluate_native_regressions_v10 as runner
from roguard.trajectory_data import file_sha, input_identity


def passing_proof():
    languages = {lang: dict(rows=count, binary_errors=0, joint_argmax_errors=0, introduced_v9=0)
                 for lang, count in (('ro', 4202), ('uk', 4635))}
    return dict(verification='PASS', decision='DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION',
        model_qualified=False, held_out_test_opened=False, encoder_loaded=False,
        checkpoint_count=16, checkpoint_dev_rows_recounted=70696, reload_rows_recounted=17674,
        initial_new_rows_compared=6336, development_pass_arms=['treatment'],
        arms={'treatment': dict(decision='DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION', languages=languages)})


def test_zero_development_admission_is_arm_specific():
    admission.require_development_pass(passing_proof(), 'treatment')
    with pytest.raises(ValueError, match='Requested arm'):
        admission.require_development_pass(passing_proof(), 'control')


@pytest.mark.parametrize('key', ['binary_errors', 'joint_argmax_errors', 'introduced_v9'])
@pytest.mark.parametrize('language', ['ro', 'uk'])
def test_one_error_blocks_regression_encoder(key, language):
    proof = passing_proof(); proof['arms']['treatment']['languages'][language][key] = 1
    with pytest.raises(ValueError, match='Nonzero'):
        admission.require_development_pass(proof, 'treatment')


@pytest.mark.parametrize('key,value', [('verification', 'FAIL'), ('decision', 'REJECT'),
    ('model_qualified', True), ('held_out_test_opened', True), ('encoder_loaded', True),
    ('checkpoint_count', 15), ('checkpoint_dev_rows_recounted', 70695),
    ('reload_rows_recounted', 17673), ('initial_new_rows_compared', 6335)])
def test_incomplete_or_invalid_audit_blocks_admission(key, value):
    proof = passing_proof(); proof[key] = value
    with pytest.raises(ValueError): admission.require_development_pass(proof, 'treatment')


def test_missing_audit_stops_before_process_or_output(tmp_path, monkeypatch):
    monkeypatch.setattr(admission.subprocess, 'run', lambda *a, **k: pytest.fail('Must not start a subprocess'))
    with pytest.raises(ValueError, match='audit107 is not complete'):
        admission.revalidate_admission(tmp_path, tmp_path, 'treatment', 1)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('mutation', ['none', 'current_artifacts', 'retained_proof'])
def test_admission_reaudits_current_artifacts(tmp_path, monkeypatch, mutation):
    proof = passing_proof(); path = tmp_path/admission.PROOF
    path.write_text(json.dumps(proof))
    prospective = tmp_path/admission.PROSPECTIVE; prospective.parent.mkdir(parents=True)
    prospective.write_text('{}'); monkeypatch.setattr(admission, 'PROSPECTIVE_SHA', file_sha(prospective))
    def run(command, *, cwd, check):
        assert cwd == tmp_path and check
        assert command[1] == 'scripts/audit_registered_matched_role_v10.py'
        assert command[command.index('--producer-pid')+1] == '123'
        fresh = deepcopy(proof)
        if mutation == 'current_artifacts': fresh['arms']['treatment']['languages']['uk']['binary_errors'] = 1
        if mutation == 'retained_proof': path.write_text(json.dumps(proof)+'\n')
        Path(command[command.index('--output')+1]).write_text(json.dumps(fresh))
    monkeypatch.setattr(admission.subprocess, 'run', run)
    if mutation == 'none':
        assert admission.revalidate_admission(tmp_path, tmp_path, 'treatment', 123) == (proof, file_sha(path))
    else:
        with pytest.raises(ValueError, match='differ|changed'):
            admission.revalidate_admission(tmp_path, tmp_path, 'treatment', 123)


def make_case(identifier, gold, lang='ro', world=True):
    point = dict(focus=0, review=gold)
    if world: point['world'] = runner.world_from_logits([0.]*19)
    return dict(format='roguard-trajectory-v10-research1', case_id=identifier, language=lang,
        turns=[dict(role='minor', text='Example')], split='challenge', family=identifier,
        author='fixture', generator='fixture', synthetic=True, checkpoints=[point])


def make_input(case, unsupported=False):
    payload = {k: case[k] for k in ('language', 'turns')}
    identity = input_identity(payload)
    return dict(population='fixture', checkpoint_id=case['case_id']+':0', language=case['language'],
        focus=0, input_digest=identity, inference_input_digest=identity, inference=payload,
        unsupported=unsupported, evidence=[])


def test_repeated_inputs_and_abstentions_stay_in_denominator():
    cases = [make_case('a', 0), make_case('b', 0), make_case('c', 1, world=False)]
    inputs = [make_input(case, i == 2) for i, case in enumerate(cases)]
    calls = []
    def predict(payload):
        calls.append(payload)
        assert set(payload) == {'language', 'turns'}
        return dict(predicted=0, score=.1), [0.]*19
    records, count = runner.predict_records(inputs, predict, 'fixture-model')
    assert len(calls) == count == 1 and len(records) == 3
    assert records[-1]['predicted'] is None and records[-1]['field_logits'] is None
    report = runner.assess(cases, records, records)
    assert report['totals'] == dict(rows=3, candidate_errors=1, v9_errors=1, fixed_v9=0,
        introduced_v9=0, annotated_worlds=2, joint_argmax_errors=0, unannotated_worlds=1)
    assert report['candidate']['groups']['ro']['strict_accuracy'] == 2/3


def test_field_abstention_and_single_regression_are_errors():
    cases = [make_case('a', 0), make_case('b', 1, 'uk')]
    inputs = [make_input(c) for c in cases]
    def predict(payload): return dict(predicted=int(payload['language'] == 'uk'), score=.5), [0.]*19
    baseline, _ = runner.predict_records(inputs, predict, 'fixture')
    candidate = deepcopy(baseline)
    candidate[1].update(predicted=None, score=None, field_logits=None)
    candidate[1].pop('interpretation_set')
    report = runner.assess(cases, baseline, candidate)
    assert report['totals']['candidate_errors'] == report['totals']['introduced_v9'] == 1
    assert report['totals']['joint_argmax_errors'] == 1
    assert report['groups']['uk']['candidate_errors'] == 1


def test_mismatched_memory_or_missing_prediction_rejected():
    cases = [make_case('a', 0)]
    rows, _ = runner.predict_records([make_input(cases[0])], lambda p: (dict(predicted=0, score=.5), [0.]*19), 'x')
    changed = deepcopy(rows); changed[0]['inference_input_digest'] = 'different'
    with pytest.raises(ValueError, match='memory views'): runner.assess(cases, rows, changed)
    with pytest.raises(ValueError, match='Missing predictions'): runner.assess(cases, rows, [])


@pytest.mark.parametrize('raw,logits', [(dict(predicted=False, score=.2), [0.]*19),
    (dict(predicted=1, score=.5), None), (dict(predicted=None, score=None), [0.]*19),
    (dict(predicted=1, score=.5), [float('nan')]*19), (dict(predicted=1, score=.5), [0.]*18),
    (dict(predicted=1, score=float('nan')), [0.]*19), (dict(predicted=1, score=1.1), [0.]*19)])
def test_malformed_serving_output_rejected(raw, logits):
    with pytest.raises(ValueError):
        runner.predict_records([make_input(make_case('a', 0))], lambda p: (raw, logits), 'x')


def test_loader_binds_population_and_keeps_unsupported_rows(tmp_path, monkeypatch):
    from roguard import trajectory_model, trajectory_neural
    cases = [make_case('a', 0), make_case('b', 1, 'uk')]
    path = tmp_path/'cases.jsonl'; path.write_text(''.join(json.dumps(c)+'\n' for c in cases))
    monkeypatch.setattr(admission, 'POPULATIONS', {'fixture': ('cases.jsonl', file_sha(path), 2)})
    monkeypatch.setattr(trajectory_model, 'TrajectoryModel', lambda *a, **kw: 'selector')
    seen = []
    def memory(payload, selector, budget, *, preserve_complete):
        assert selector == 'selector' and budget == 1200 and preserve_complete
        assert set(payload) == {'language', 'turns'}
        seen.append(payload)
        return dict(mandatory_overflow=payload['language'] == 'uk', spans=[]), payload
    monkeypatch.setattr(trajectory_neural, 'prepare_memory', memory)
    loaded, inputs = admission.load_inputs(tmp_path, tmp_path/'selector')
    assert loaded == {'fixture': cases} and len(seen) == len(inputs) == 2
    assert [r['unsupported'] for r in inputs] == [False, True]
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='population changed'):
        admission.load_inputs(tmp_path, tmp_path/'selector')


def test_public_engine_used_and_released_on_error(monkeypatch):
    np = pytest.importorskip('numpy')
    # Lightweight fake MLX interface: prove call wiring and cleanup, not neural accuracy.
    import sys
    from types import SimpleNamespace
    from roguard import gemma4_review
    fake_nn = SimpleNamespace(Module=object)
    monkeypatch.setitem(sys.modules, 'mlx', SimpleNamespace(nn=fake_nn))
    monkeypatch.setitem(sys.modules, 'mlx.nn', fake_nn)
    class MX:
        def eval(self, *a): pass
        def clear_cache(self): pass
    original_head = lambda h: np.zeros((1, 19), np.float32)
    engines = []
    class Engine:
        def __init__(self, *args):
            assert args == ('base', 'adapter', 'sha')
            self.mx = MX(); self.model = SimpleNamespace(heads={'ro': original_head}); engines.append(self)
    monkeypatch.setattr(gemma4_review, '_Engine', Engine)
    class Component:
        _args = ('base', 'adapter', 'sha'); language = 'ro'; identity = 'identity'
        _spec = dict(mode='binary'); _engine = None
        def predict(self, payload):
            self._engine.model.heads['ro'](None)
            return dict(predicted=1, score=.5)
    component = Component()
    records, calls = runner.infer_model(component, [make_input(make_case('a', 1))], lambda v: None)
    assert len(records) == calls == 1 and component._engine is None
    assert engines[-1].model.heads['ro'] is original_head
    def fail(payload): raise RuntimeError('injected inference failure')
    component.predict = fail
    with pytest.raises(RuntimeError, match='injected'):
        runner.infer_model(component, [make_input(make_case('a', 1))], lambda v: None)
    assert component._engine is None and engines[-1].model.heads['ro'] is original_head


@pytest.mark.parametrize('candidate_fails', [False, True])
def test_complete_cli_records_results_without_release_admission(tmp_path, monkeypatch, candidate_fails):
    import sys
    from types import SimpleNamespace
    from eval import run_trajectory_v10
    from roguard import gemma4_review
    from training.register_matched_role_attention_v10 import OUTPUTS
    root = tmp_path/'repo'; private = tmp_path/'private'; study = root/'new-study'
    actual_root = Path(runner.__file__).resolve().parents[1]
    for name in ('eval/native_regression_inputs_v10.py', 'eval/evaluate_native_regressions_v10.py',
                 'eval/run_trajectory_v10.py', 'src/roguard/trajectory_model.py',
                 'src/roguard/trajectory_memory.py', 'src/roguard/trajectory_neural.py',
                 'src/roguard/trajectory_data.py', 'src/roguard/trajectory_evaluation.py'):
        target = root/name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((actual_root/name).read_bytes())
    selector = root/'selector'; selector.write_bytes(b'fixture')
    cases = [make_case('a', 0), make_case('b', 1, 'uk', world=False)]
    inputs = [make_input(c) for c in cases]
    case_path = root/'cases.jsonl'; case_path.write_text(''.join(json.dumps(c)+'\n' for c in cases))
    monkeypatch.setattr(runner, 'POPULATIONS', {'fixture': ('cases.jsonl', file_sha(case_path), 2)})
    monkeypatch.setattr(runner, 'PUBLISHED_RUNTIME', {})
    monkeypatch.setattr(runner, 'SELECTOR_SHA', file_sha(selector))
    monkeypatch.setattr(runner, 'verify_runtime', lambda root: None)
    monkeypatch.setattr(runner, 'load_inputs', lambda *a: ({'fixture': cases}, inputs))
    proof = passing_proof()
    for lang in ('ro', 'uk'):
        path = private/OUTPUTS['treatment']['private']/(lang+'-selected.safetensors')
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b'fixture-weights')
        proof['arms']['treatment']['languages'][lang]['artifacts'] = {path.name: file_sha(path)}
        spec = root/OUTPUTS['treatment']['study']/(lang+'-review.json')
        spec.parent.mkdir(parents=True, exist_ok=True); spec.write_text('{}')
    monkeypatch.setattr(runner, 'revalidate_admission', lambda *a: (proof, 'admission-fixture'))
    class Frozen:
        def __init__(self, base, candidate, staging):
            staging.mkdir()
            self.models = {lang: SimpleNamespace(identity='baseline-'+lang) for lang in ('ro', 'uk')}
    monkeypatch.setattr(run_trajectory_v10, 'LocalFrozenV9', Frozen)
    monkeypatch.setattr(gemma4_review, 'Gemma4ReviewModel',
        lambda *a, **k: SimpleNamespace(identity='candidate'))
    seen = []
    def inference(component, rows, progress):
        seen.append((component.identity, rows[0]['language']))
        def predict(payload):
            value = int(payload['language'] == 'uk')
            if candidate_fails and component.identity == 'candidate': value = 1-value
            return dict(predicted=value, score=.9 if value else .1), [0.]*19
        return runner.predict_records(rows, predict, component.identity, progress=progress)
    monkeypatch.setattr(runner, 'infer_model', inference)
    argv = ['runner', '--root', str(root), '--private-root', str(private), '--base', str(root/'base'),
        '--selector', str(selector), '--study', str(study), '--arm', 'treatment', '--producer-pid', '123']
    monkeypatch.setattr(sys, 'argv', argv)
    runner.main()
    report = json.loads((study/'report.json').read_text())
    assert seen == [('baseline-ro', 'ro'), ('candidate', 'ro'), ('baseline-uk', 'uk'), ('candidate', 'uk')]
    assert report['complete'] and not report['stable_release_ready'] and not report['reviewers_admitted']
    assert not report['held_out_test_opened'] and not report['independent_validation']
    assert report['decision'] == ('REJECT' if candidate_fails else 'EXPOSED_REGRESSION_PASS_ONLY')
    assert report['populations']['fixture']['totals']['unannotated_worlds'] == 1
    for kind in ('baseline', 'candidate'):
        assert file_sha(study/(kind+'.jsonl')) == report['prediction_sha256'][kind]
    with pytest.raises(ValueError, match='overwrite'): runner.main()
