"""Train a separately represented semantic probe without modifying review scores."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time

from roguard.evidence import ONTOLOGY
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl, input_identity
from training.semantic_sparse_v10 import DIMENSION, vector, probabilities


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('additional_training','development_study','study'):
        p.add_argument('--'+key.replace('_','-'),dest=key,type=Path,required=True)
    p.add_argument('--amendment',type=Path,default=Path('research/v10-evidence/protocol-amendment-13.json'))
    p.add_argument('--feature-mode',choices=('plain','quote-scope'),default='plain')
    a=p.parse_args()
    if a.study.exists():raise ValueError('Refusing to overwrite semantic probe')
    roots=[Path('research/v9-stabilization')/name for name in
           ('shared-tail-corpus-164','grounding-tail-corpus-174','referent-tail-corpus-176','reported-tail-corpus-195')]
    examples=[];inputs={}
    for root in roots:
        path=root/'inputs.jsonl.gz';inputs[str(path)]=file_sha(path)
        examples.extend(json.loads(line) for line in gzip.open(path,'rt',encoding='utf-8'))
    added_report=json.loads((a.additional_training/'report.json').read_text())
    if file_sha(a.additional_training/'train.jsonl')!=added_report['training_data_sha256']:
        raise ValueError('Additional training differs')
    for row in read_jsonl(a.additional_training/'train.jsonl'):
        if row['partition']!='train':raise ValueError('Only training additions allowed')
        examples.append({'row':row,'partition':'train','group':row['source']})
    dev_report=json.loads((a.development_study/'report.json').read_text())
    if not dev_report['complete'] or file_sha(a.development_study/'inputs.jsonl')!=dev_report['inputs_sha256']:
        raise ValueError('Eligible trajectory development differs')
    additions=read_jsonl(a.development_study/'inputs.jsonl')
    if any(r['partition']!='dev' for r in additions):raise ValueError('Only development additions allowed')
    examples.extend(additions)
    a.study.mkdir(parents=True)
    amendment=a.amendment
    configuration=json.loads(amendment.read_text())['registered_fit']
    protocol={'format':'roguard-semantic-sparse-study-1','source_sha256':file_sha(Path(__file__)),
        'feature_source_sha256':file_sha(Path('training/semantic_sparse_v10.py')),
        'amendment_sha256':file_sha(amendment),'inputs':inputs,
        'additional_training_sha256':file_sha(a.additional_training/'train.jsonl'),
        'development_report_sha256':file_sha(a.development_study/'report.json'),
        'feature_mode':a.feature_mode,
        'calibration_or_test_loaded':False,'binary_review_model_changed':False,'stable_release_ready':False}
    (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    (a.study/'feature-source.py').write_bytes(Path('training/semantic_sparse_v10.py').read_bytes())
    import numpy as np
    from scipy.sparse import csr_matrix
    from sklearn.linear_model import LogisticRegression
    from array import array
    data=array('d');indices=array('i');indptr=array('i',[0]);metadata=[];started=time.monotonic()
    seen={}
    for entry in examples:
        row=entry['row'];payload={'language':row['language'],'turns':row['turns'][:row['focus']+1]}
        identity=input_identity(payload)
        if identity in seen and seen[identity]!=entry['partition']:
            raise ValueError('Exact train/development input leakage')
        seen[identity]=entry['partition']
        features=vector(payload,a.feature_mode)
        for index,value in sorted(features.items()):indices.append(index);data.append(value)
        indptr.append(len(data))
        metadata.append({'case_id':row['case_id'],'language':row['language'],'partition':entry['partition'],
                         'group':entry['group'],'input_digest':identity,'world':row['attributes']})
    x=csr_matrix((np.frombuffer(data,dtype=np.float64),np.frombuffer(indices,dtype=np.int32),
                  np.frombuffer(indptr,dtype=np.int32)),shape=(len(examples),DIMENSION))
    train=np.array([i for i,r in enumerate(metadata) if r['partition']=='train'])
    dev=np.array([i for i,r in enumerate(metadata) if r['partition']=='dev'])
    if len(train)+len(dev)!=len(metadata):raise ValueError('Unexpected partition')
    heads={};predictions=[dict(metadata[i],probabilities={},prediction={}) for i in dev]
    export_error=0
    for field in ONTOLOGY:
        labels=[metadata[i]['world'][field] for i in train]
        model=LogisticRegression(C=configuration['regularization_C'],max_iter=configuration['max_iterations'],
            solver=configuration['solver'],class_weight=configuration['class_weight'],random_state=configuration['seed'])
        model.fit(x[train],labels)
        if np.any(model.n_iter_>=configuration['max_iterations']):raise ValueError('Semantic classifier did not converge')
        head={'classes':model.classes_.tolist(),'weights':model.coef_.tolist(),'bias':model.intercept_.tolist()};heads[field]=head
        values=model.predict_proba(x[dev])
        for offset,row in enumerate(predictions):
            probs=dict(zip(head['classes'],values[offset].tolist()))
            row['probabilities'][field]=probs;row['prediction'][field]=max(probs,key=probs.get)
        for index in (0,len(dev)//2,len(dev)-1):
            sparse=x[dev[index]];portable=probabilities(head,dict(zip(sparse.indices,sparse.data)))
            export_error=max(export_error,max(abs(portable[label]-values[index,j]) for j,label in enumerate(head['classes'])))
        print(json.dumps({'field':field,'iterations':model.n_iter_.tolist(),'seconds':time.monotonic()-started}),flush=True)
    if export_error>1e-10:raise ValueError('Portable semantic export differs')
    artifact={'format':'roguard-semantic-sparse-v10-research1','dimension':DIMENSION,
        'feature_contract':'focus-history-word12-char345-blake2b-l2-v1' if a.feature_mode=='plain' else 'quotation-frame-word12-char345-blake2b-channel-l2-v1',
        'feature_mode':a.feature_mode,'heads':heads,
        'qualified':False,'binary_review_model_changed':False,'permits_external_action':False}
    path=a.study/'semantic.json.gz';path.write_bytes(gzip.compress(json.dumps(artifact,sort_keys=True,separators=(',',':'),allow_nan=False).encode(),mtime=0))
    write_jsonl(a.study/'development-predictions.jsonl',predictions)
    groups={}
    for lang in ('ro','uk'):
        selected=[r for r in predictions if r['language']==lang]
        groups[lang]={'rows':len(selected),'field_errors':{f:sum(r['world'][f]!=r['prediction'][f] for r in selected) for f in ONTOLOGY},
                      'joint_argmax_errors':sum(r['world']!=r['prediction'] for r in selected)}
    report={'complete':True,'training_rows':len(train),'development_rows':len(dev),'groups':groups,
        'artifact_sha256':file_sha(path),'artifact_bytes':path.stat().st_size,'portable_export_max_error':export_error,
        'seconds':time.monotonic()-started,'stable_release_ready':False,'calibration_or_test_loaded':False}
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
