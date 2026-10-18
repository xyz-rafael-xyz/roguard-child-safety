"""Extract fixed native CLS/focal representations without changing the encoder."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--language',choices=('ro','uk'),required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--stage',choices=('fit','cal'),required=True)
    a=p.parse_args()
    import numpy as np
    import torch
    from roguard.native_review import NativeReviewModel,_TorchEngine,prepare_input
    from roguard.native_fields import _native_states
    from roguard.native_assets import PADDING_MULTIPLE
    torch.set_num_threads(4)
    root=Path('research/v9-stabilization');parent=root/('native-full-22-'+a.language)
    study=root/'native-fields-43'/a.language
    if a.stage=='cal' and not (study/'selection.json').exists():
        raise ValueError('Head selection must precede calibration feature extraction')
    artifact=parent/'review.json.gz'
    wrapper=NativeReviewModel(artifact,parent/'selected-model.safetensors',a.base,
        root/'native-full-22-base-manifest.json',expected_artifact_sha256=sha(artifact),device='mps')
    engine=_TorchEngine(*wrapper._engine_args)
    if not engine.tokenizer.is_fast:raise ValueError('Fast tokenizer sequence identity is required')
    study.mkdir(parents=True,exist_ok=True)
    for split in (('train','dev') if a.stage=='fit' else ('cal',)):
        path=parent/(split+'.jsonl.gz');rows=[json.loads(l) for l in gzip.decompress(path.read_bytes()).splitlines()]
        pairs=[prepare_input(dict(language=r['language'],turns=r['turns']),r['focus'])[0] for r in rows]
        encoded=engine.tokenizer([p[0] for p in pairs],[p[1] for p in pairs],padding=False,truncation=False)
        if max(map(len,encoded['input_ids']))>512:raise ValueError('No truncation permitted')
        features=[];scores=[];start=time.monotonic()
        with torch.inference_mode():
            for offset in range(0,len(rows),8):
                ix=list(range(offset,min(offset+8,len(rows))))
                batch=engine.tokenizer.pad([{k:v[i] for k,v in encoded.items()} for i in ix],
                    return_tensors='pt',pad_to_multiple_of=PADDING_MULTIPLE)
                mask=np.zeros(tuple(batch['input_ids'].shape),dtype=np.float32)
                for j,i in enumerate(ix):
                    ids=encoded.sequence_ids(i)
                    mask[j,:len(ids)]=[v==1 for v in ids]
                if (mask.sum(axis=1)==0).any():raise ValueError('Focal token span is empty')
                inputs={k:v.to('mps') for k,v in batch.items()}
                hidden,logits=_native_states(engine,inputs,include_logits=True)
                focal=torch.from_numpy(mask).to('mps')
                pooled=(hidden*focal[:,:,None]).sum(dim=1)/focal.sum(dim=1)[:,None]
                features.append(torch.cat((hidden[:,0,:],pooled),dim=1).cpu().numpy())
                scores.extend(torch.softmax(logits.float(),dim=-1)[:,1].cpu().tolist())
                del logits,hidden,pooled,inputs,focal
                if (offset//8+1)%25==0:torch.mps.empty_cache()
                if (offset//8+1)%100==0:print(split,min(offset+8,len(rows)),'of',len(rows),flush=True)
        values=np.concatenate(features)
        if values.shape!=(len(rows),1536) or not np.isfinite(values).all():raise ValueError('Invalid native features')
        record=dict(rows=len(rows),dimensions=1536,seconds=time.monotonic()-start,source_sha256=sha(path),
            native_artifact_sha256=sha(artifact),checkpoint_sha256=sha(parent/'selected-model.safetensors'),
            extractor_sha256=sha(__file__),no_encoder_update=True,no_truncation=True)
        if split!='train':
            reference_path=parent/(split+'-predictions.jsonl')
            reference=[json.loads(l) for l in reference_path.read_text().splitlines()]
            if [r['case_id'] for r in rows]!=[r['case_id'] for r in reference]:raise ValueError('Reference rows differ')
            error=max(abs(s-r['score']) for s,r in zip(scores,reference))
            changes=sum(int(s>=wrapper.threshold)!=r['predicted'] for s,r in zip(scores,reference))
            record.update(native_score_max_error=error,native_decision_changes=changes,reference_sha256=sha(reference_path))
            if error>1e-4 or changes:raise ValueError('Feature extraction changed native inference')
        target=study/(split+'-features.npz')
        if target.exists():raise ValueError('Feature extraction already exists')
        np.savez_compressed(target,features=values,native_scores=np.array(scores),case_ids=np.array([r['case_id'] for r in rows]))
        record['features_sha256']=sha(target)
        (study/(split+'-features.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
