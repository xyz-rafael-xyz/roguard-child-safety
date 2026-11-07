"""Append bound compositional TRAIN/DEV features to the original v9 feature store."""
import argparse
import json
from pathlib import Path
import shutil
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.trajectory_data import file_sha,read_jsonl,write_jsonl
from training.extract_semantic_augmentation_v10 import CachedEngine
from training.native_fields_v9 import targets


def validate_new(rows):
    seen=set();families={}
    for row in rows:
        if row['partition'] not in ('train','dev'):raise ValueError('Only TRAIN/DEV extraction is allowed')
        if row['focus']!=len(row['turns'])-1:raise ValueError('Expected focal-last input')
        key=digest({'language':row['language'],'turns':row['turns']})
        if key in seen:raise ValueError('Duplicate new payload')
        seen.add(key)
        if families.setdefault(row['family'],row['partition'])!=row['partition']:raise ValueError('Family leakage')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('base','candidate','data_study','data_private','features','feature_study','study','private','amendment'):
        p.add_argument('--'+name.replace('_','-'),dest=name,type=Path,required=True)
    p.add_argument('--resume',action='store_true');p.add_argument('--preflight-report',type=Path)
    a=p.parse_args();plan=json.loads(a.amendment.read_text())
    data=json.loads((a.data_study/'report.json').read_text());parent=json.loads((a.feature_study/'report.json').read_text())
    if not data['complete'] or not parent['complete']:raise ValueError('Incomplete bound source')
    if data['format']!='roguard-compositional-review-1':raise ValueError('Wrong generated corpus')
    for split in ('train','dev','test'):
        if file_sha(a.data_private/(split+'.jsonl'))!=data['artifacts'][split]:raise ValueError('Generated source changed')
    rows=read_jsonl(a.data_private/'train.jsonl')+read_jsonl(a.data_private/'dev.jsonl');validate_new(rows)
    if len(rows)!=plan['new_train_rows']+plan['new_dev_rows']:raise ValueError('Registered count differs')
    for language,record in parent['languages'].items():
        for suffix,expected in record['artifacts'].items():
            if file_sha(a.features/(language+'-'+suffix))!=expected:raise ValueError('Parent feature binding differs')
    protocol={'format':'roguard-large-review-features-1','source_sha256':file_sha(__file__),
              'amendment_sha256':file_sha(a.amendment),'data_report_sha256':file_sha(a.data_study/'report.json'),
              'parent_report_sha256':file_sha(a.feature_study/'report.json'),
              'helper_sources':{path:file_sha(path) for path in ('training/extract_semantic_augmentation_v10.py',
                  'src/roguard/trajectory_neural.py','src/roguard/gemma4_review.py','src/roguard/gemma4_contract.py')},
              'new_partitions':['train','dev'],'test_opened':False,'stable_release_ready':False}
    if a.preflight_report:
        if a.preflight_report.exists():raise ValueError('Preflight exists')
        a.preflight_report.parent.mkdir(parents=True,exist_ok=True)
        a.preflight_report.write_text(json.dumps({**protocol,'new_rows':len(rows),'model_loaded':False},indent=2)+'\n');return
    if a.resume:
        if json.loads((a.study/'protocol.json').read_text())!=protocol:raise ValueError('Resume binding differs')
        if (a.study/'report.json').exists() and json.loads((a.study/'report.json').read_text()).get('complete'):raise ValueError('Extraction already complete')
    else:
        if a.study.exists() or a.private.exists():raise ValueError('Refusing to overwrite extraction')
        a.study.mkdir(parents=True);a.private.mkdir(parents=True)
        (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
        (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import fcntl
    lock=(a.private/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import gc
    import numpy as np
    import mlx.nn as nn
    from roguard.gemma4_contract import classifier_tokens
    binding=a.study/'v9-binding'
    if binding.exists():
        number=1
        while (a.study/('v9-binding-resume-'+str(number))).exists():number+=1
        binding=a.study/('v9-binding-resume-'+str(number))
    frozen=LocalFrozenV9(a.base,a.candidate,binding);started=time.monotonic()
    report={'complete':False,'languages':{},'test_opened':False,'stable_release_ready':False}
    for language in ('ro','uk'):
        component=frozen.models[language];engine=CachedEngine(*component._args);mx=engine.mx
        class Capture(nn.Module):
            def __init__(self,inner):super().__init__();self.inner=inner
            def __call__(self,h):
                z=self.inner(h);mx.eval(h,z);self.h=np.asarray(h).copy();self.z=np.asarray(z).copy();return z
        tap=Capture(engine.model.heads[language]);engine.model.heads[language]=tap
        added=[row for row in rows if row['language']==language]
        old=read_jsonl(a.features/(language+'-rows.jsonl'));metadata=list(old);seen={row['token_sha256'] for row in old}
        arrays=[];folder=a.private/(language+'-chunks');folder.mkdir(exist_ok=True)
        for offset in range(0,len(added),32):
            if shutil.disk_usage(a.private).free<8*1024**3:raise RuntimeError('Disk reserve reached')
            batch=added[offset:offset+32];path=folder/(str(offset)+'.npy');meta=folder/(str(offset)+'.json')
            row_hash=digest(batch)
            if meta.exists():
                record=json.loads(meta.read_text())
                if record['row_sha256']!=row_hash or file_sha(path)!=record['array_sha256']:raise ValueError('Retained feature chunk differs')
                values=np.load(path,allow_pickle=False);new_metadata=record['rows']
            else:
                if path.exists():raise ValueError('Uncommitted feature chunk retained')
                values=[];new_metadata=[]
                for row in batch:
                    payload={'language':language,'turns':row['turns']}
                    if len(payload['turns'])>12 or engine.scores([payload],component._spec)[0] is None:raise ValueError('Input exceeds frozen envelope')
                    values.append(np.concatenate((tap.h[0],tap.z[0])))
                    new_metadata.append({'case_id':row['case_id'],'language':language,'partition':row['partition'],
                         'group':row['source'],'gold':row['review'],'fields':targets([row])[0],
                         'family':row['family'],'token_sha256':digest(classifier_tokens(engine.tokenizer,payload))})
                    mx.clear_cache()
                values=np.asarray(values,dtype=np.float32);np.save(path,values,allow_pickle=False)
                meta.write_text(json.dumps({'row_sha256':row_hash,'array_sha256':file_sha(path),'rows':new_metadata},indent=2)+'\n')
            if values.shape!=(len(batch),2579) or not np.isfinite(values).all():raise ValueError('Malformed feature chunk')
            if len(new_metadata)!=len(batch):raise ValueError('Incomplete chunk metadata')
            for row in new_metadata:
                if row['token_sha256'] in seen:raise ValueError('Token input duplicates parent or new population')
                seen.add(row['token_sha256'])
            arrays.append(values);metadata.extend(new_metadata)
            progress={'language':language,'completed':offset+len(batch),'total':len(added),'seconds':time.monotonic()-started}
            temp=a.study/'progress.tmp';temp.write_text(json.dumps(progress,indent=2)+'\n');temp.replace(a.study/'progress.json')
            print(json.dumps(progress),flush=True)
        new=np.concatenate(arrays)
        for suffix,new_values in (('features.npy',new[:,:2560]),('logits.npy',new[:,2560:])):
            values=np.concatenate((np.load(a.features/(language+'-'+suffix),allow_pickle=False),new_values))
            np.save(a.private/(language+'-'+suffix),values,allow_pickle=False)
        write_jsonl(a.private/(language+'-rows.jsonl'),metadata)
        report['languages'][language]={'rows':len(metadata),'dimensions':2560,
            'partitions':{split:sum(row['partition']==split for row in metadata) for split in ('train','dev')},
            'artifacts':{suffix:file_sha(a.private/(language+'-'+suffix)) for suffix in ('features.npy','logits.npy','rows.jsonl')}}
        (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        del engine,tap,new,values,arrays;gc.collect();mx.clear_cache()
    report.update(complete=True,seconds=time.monotonic()-started)
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
