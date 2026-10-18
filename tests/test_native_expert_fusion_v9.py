import gzip
import json
import pytest


@pytest.mark.parametrize('bad_language', [False, True])
def test_expert_feature_alignment_and_native_marginals(tmp_path, bad_language):
    from roguard.evidence import ONTOLOGY
    from training.native_expert_fusion_v9 import load_split
    attrs = dict(subject='SELF', source='DIRECT', time='CURRENT', assertion='AFFIRMED', relation='NEW', concern='1')
    row = dict(case_id='example', language='uk', source='formal', attributes=attrs)
    direct = dict(case_id='example', language='uk', gold=1, score=.8)
    field = dict(case_id='example', language='ro' if bad_language else 'uk', gold=1,
        field_probabilities={f: [float(v == label) for label in ONTOLOGY[f]] for f, v in attrs.items()})
    native = tmp_path / 'native-full-22-uk'
    fields = tmp_path / 'scope-repair-66/uk'
    data = tmp_path / 'scope-repair-66/encoders/uk'
    for path in (native, fields, data):
        path.mkdir(parents=True)
    encoded = gzip.compress((json.dumps(row)+'\n').encode())
    for path in (native, data):
        (path / 'dev.jsonl.gz').write_bytes(encoded)
    (native / 'dev-predictions.jsonl').write_text(json.dumps(direct)+'\n')
    (fields / 'dev-predictions.jsonl').write_text(json.dumps(field)+'\n')
    if bad_language:
        with pytest.raises(ValueError, match='languages'):
            load_split(tmp_path, 'uk', 'dev')
    else:
        rows, vectors, labels, hashes = load_split(tmp_path, 'uk', 'dev')
        assert labels == [1] and vectors[0][0] == .8
        assert len(vectors[0]) == 2 + 3*sum(map(len, ONTOLOGY.values()))
        assert len(hashes) == 3
