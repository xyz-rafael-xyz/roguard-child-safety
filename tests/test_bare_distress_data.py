from pathlib import Path
from training.bare_distress_data_v9 import generate
from training.needs_repair_data_v9 import generate as needs
from training.train_native_review_v9 import texts
from training.review_fusion_v9 import read
from roguard.evidence import ONTOLOGY


def test_authored_inputs_exclude_existing_development_calibration_and_stress():
    rows=generate();keys=[texts(r) for r in rows]
    assert len(keys)==len(set(keys))==864
    for row in rows:
        assert set(row['attributes'])==set(ONTOLOGY)
        assert all(v in ONTOLOGY[k] for k,v in row['attributes'].items())
    protected=needs('dev')
    for split in ('dev','cal'):
        protected+=read(Path('research/v9-stabilization/native-full-22-uk')/(split+'.jsonl.gz'))
    assert not set(keys)&{texts(r) for r in protected}
