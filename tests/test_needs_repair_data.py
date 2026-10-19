from training.needs_repair_data_v9 import generate
from training.train_native_review_v9 import texts
from roguard.evidence import ONTOLOGY


def test_new_content_families_do_not_overlap_stress_development():
    sets=[]
    for split in ('train','dev'):
        rows=generate(split);keys=[texts(r) for r in rows]
        assert len(keys)==len(set(keys))
        assert all(set(r['attributes'])==set(ONTOLOGY) for r in rows)
        assert all(all(v in ONTOLOGY[k] for k,v in r['attributes'].items()) for r in rows)
        sets.append(set(keys))
    assert not sets[0]&sets[1]
    assert len(sets[0])==864 and len(sets[1])==216
