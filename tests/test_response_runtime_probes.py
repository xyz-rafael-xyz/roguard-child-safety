import pytest
from eval.verify_response_runtime_v10 import select_probes,payload
from roguard.assurance import digest


def rows():
    return [{'case_id':str(i),'partition':'train' if i<2 else 'dev','language':'ro','gold':'appropriate','context':'','draft':'Un mesaj '+str(i)} for i in range(4)]


def identities(data):return [{'case_id':r['case_id'],'partition':r['partition'],'input_sha256':digest(payload(r))} for r in data]


def test_probes_match_complete_bound_inputs_and_never_open_held_out_test():
    data=rows();assert select_probes(data,identities(data))==[0,1,2,3]
    with pytest.raises(ValueError):select_probes(data,identities(data)[:-1])
    bound=identities(data);data[0]['draft']='Changed'
    with pytest.raises(ValueError,match='identity'):select_probes(data,bound)
    data=rows();data[0]['partition']='test'
    with pytest.raises(ValueError,match='TRAIN/DEV'):select_probes(data,identities(data))
