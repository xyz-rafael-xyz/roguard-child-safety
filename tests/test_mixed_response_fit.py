import json

import pytest

pytest.importorskip('numpy')

from roguard.trajectory_model import RESPONSE_LABELS
from training import train_mixed_response_attention_v10 as trainer


SAMPLING = {'kind': 'four-stratum-cyclic-v1', 'examples_per_stratum': 2,
            'batches_per_pass': 3072, 'presentations_per_pass': 24576}


def registered_shape():
    rows = [{'case_id': 'dev-excluded', 'partition': 'dev'}]
    for origin in ('historical', 'generated'):
        for language in ('ro', 'uk'):
            for gold in RESPONSE_LABELS:
                n = (98 if gold == 'appropriate' else 50) if origin == 'historical' else (3072 if gold == 'appropriate' else 768)
                rows.extend({'case_id': f'{origin}:{language}:{gold}:{i}',
                             'origin': origin, 'language': language,
                             'gold': gold, 'partition': 'train'} for i in range(n))
    return rows, list(range(1, len(rows)))


@pytest.mark.parametrize('changed', ['kind', 'examples_per_stratum', 'batches_per_pass',
                                   'presentations_per_pass', 'extra'])
def test_sampler_contract_cannot_silently_change(changed):
    plan = {**SAMPLING, changed: 'invalid'}
    with pytest.raises(ValueError, match='configuration differs'):
        trainer.prepare_sampler([], [], {'batch_size': 8}, plan)


@pytest.mark.parametrize('batch_size', [8, 16])
def test_trainer_preflight_counts_real_schedule_and_never_loads_encoder(tmp_path, monkeypatch, batch_size):
    rows, indices = registered_shape()
    monkeypatch.setattr(trainer, 'load_inputs', lambda args: (
        rows, None, indices, [0], [], {}, {}, {'fixture': 'metadata-only'}))

    def forbidden(*args, **kwargs):
        pytest.fail('Preflight attempted encoder construction')

    monkeypatch.setattr(trainer, '_Engine', forbidden)
    monkeypatch.setattr(trainer, 'LocalFrozenV9', forbidden)
    amendment = tmp_path / 'plan.json'
    amendment.write_text(json.dumps({'optimization': {'batch_size': batch_size, 'epochs': 6},
                                    'sampling': SAMPLING, 'fresh_response_head': True}))
    output = tmp_path / 'preflight.json'
    argv = ['mixed-trainer', '--amendment', str(amendment), '--preflight-report', str(output)]
    for name in ('base', 'candidate', 'data', 'parent-study', 'parent-private',
                 'contrast-study', 'contrast-private', 'context-study', 'generated-study',
                 'generated-private', 'feature-study', 'feature-private', 'study', 'private'):
        argv += ['--' + name, str(tmp_path / name)]
    monkeypatch.setattr('sys.argv', argv)
    if batch_size != 8:
        with pytest.raises(ValueError, match='configuration differs'):
            trainer.main()
        assert not output.exists()
        return
    trainer.main()
    report = json.loads(output.read_text())
    assert report['train_rows'] == 12884
    assert report['updates'] == 18432
    assert report['presentations'] == 147456
    assert report['sampling'] == SAMPLING
    assert report['model_loaded'] is False and report['qualified'] is False
    assert not (tmp_path / 'study').exists()
    assert not (tmp_path / 'private').exists()
