import pytest
from eval.analyze_large_review_errors_v10 import analyze


def test_unannotated_world_stays_unannotated_and_missing_results_fail():
    cases=[{'case_id':'c','family':'f','language':'ro','checkpoints':[{'focus':0,'review':0}]}]
    old=[{'checkpoint_id':'c:0','input_digest':'a','inference_input_digest':'a','predicted':0,'score':.1}]
    new=[{**old[0],'predicted':1,'score':.9}]
    records,groups=analyze(cases,old,new)
    assert records[0]['world'] is None and records[0]['outcome']=='introduced'
    assert groups=={'introduced:ro:0:UNANNOTATED:UNANNOTATED:UNANNOTATED:UNANNOTATED':1}
    for bad in ([],new+new):
        with pytest.raises(ValueError,match='Incomplete or duplicate'):analyze(cases,old,bad)
    with pytest.raises(ValueError,match='inputs differ'):
        analyze(cases,old,[{**new[0],'inference_input_digest':'different'}])
