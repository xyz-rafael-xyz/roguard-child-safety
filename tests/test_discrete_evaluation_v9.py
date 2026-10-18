import gzip
import hashlib
import json
import sys

import pytest


@pytest.mark.parametrize('corrupt_gold', [False, True])
def test_research_evaluation_binds_labels_and_keeps_groups(tmp_path, monkeypatch, corrupt_gold):
    from roguard.evidence import ONTOLOGY
    import roguard.discrete_fields as runtime
    from training.evaluate_discrete_fields_v9 import main
    parent = tmp_path / 'scope-repair-66/ro'
    data = tmp_path / 'scope-repair-66/encoders/ro'
    study = tmp_path / 'discrete-policy-69'
    for path in (parent, data, study):
        path.mkdir(parents=True)
    (parent / 'selected-heads.json.gz').write_bytes(b'fixture identity')
    digest = hashlib.sha256(b'fixture identity').hexdigest()
    (parent / 'selection.json').write_text(json.dumps(dict(artifact_sha256=digest)))
    (data / 'review.json.gz').write_bytes(b'fixture native identity')
    from pathlib import Path
    (study / 'source-freeze.json').write_text(json.dumps({'src/roguard/discrete_fields.py':
        hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest()}))
    rows, predictions = [], []
    for source in ('formal', 'v7_training_frame_projection'):
        for label in ('0', '1'):
            values = dict(subject='SELF', time='CURRENT', assertion='AFFIRMED', source='DIRECT', relation='NEW', concern=label)
            identity = str(len(rows))
            rows.append(dict(case_id=identity, language='ro', source=source, focus=0,
                turns=[dict(role='minor', text='Exemplu fictiv ' + identity)], attributes=values))
            predictions.append(dict(case_id=identity, gold=int(label),
                field_probabilities={f: [float(v == item) for item in ONTOLOGY[f]] for f, v in values.items()}))
    if corrupt_gold:
        predictions[0]['gold'] = 1
    for split in ('dev', 'cal'):
        (data / (split + '.jsonl.gz')).write_bytes(gzip.compress(''.join(json.dumps(v)+'\n' for v in rows).encode()))
        (parent / (split + '-predictions.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in predictions))
    monkeypatch.setattr(sys, 'argv', ['evaluate', 'research', '--root', str(tmp_path), '--language', 'ro'])
    if corrupt_gold:
        with pytest.raises(ValueError, match='Policy labels'):
            main()
        assert not (study / 'ro/report.json').exists()
    else:
        main()
        report = json.loads((study / 'ro/report.json').read_text())
        assert report['complete'] and report['accuracy_gates_pass']
        assert set(report['splits']) == {'dev', 'cal'}
        assert set(report['splits']['cal']['counts']) == {'formal', 'conversational'}
