from collections import Counter
from roguard.evidence import ONTOLOGY, disposition
from training.distress_repair_data_v9 import generate
from training.train_native_review_v9 import texts


def test_authored_contrasts_are_unique_and_cover_both_decisions():
    rows=generate();seen={};counts=Counter()
    assert len({r['case_id'] for r in rows})==len(rows)
    for row in rows:
        assert row['synthetic'] is True and row['language']=='uk'
        assert row['focus']==len(row['turns'])-1
        assert set(row['attributes'])==set(ONTOLOGY)
        assert all(v in ONTOLOGY[k] for k,v in row['attributes'].items())
        key=texts(row)
        assert key not in seen
        seen[key]=row['attributes']
        decision=disposition({k:{'label':v} for k,v in row['attributes'].items()})[0]
        counts[decision]+=1
    assert counts['CONTINUE']>0 and sum(v for k,v in counts.items() if k!='CONTINUE')>0
