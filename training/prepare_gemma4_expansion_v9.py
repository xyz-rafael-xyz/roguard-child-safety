"""Expand only existing formal and indirect-distress training coverage."""
import argparse
from collections import defaultdict,Counter
from datetime import datetime,timezone
import gzip,hashlib,json
from pathlib import Path
import shutil
from training.gemma4_prompt_v9 import classifier_tokens
from training.prepare_instruction_sft_v8 import target


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    from transformers import AutoTokenizer
    root=Path('research/v9-stabilization');parent=root/'gemma4-readout-141';out=root/'gemma4-expansion-144'
    out.mkdir(exist_ok=False);sources=set()
    def read(path):
        sources.add(path);raw=path.read_bytes()
        if str(path).endswith('.gz'):raw=gzip.decompress(raw)
        values=[json.loads(l) for l in raw.decode().splitlines()]
        return values
    baseline=read(parent/'inputs.jsonl.gz')
    tokenizer=AutoTokenizer.from_pretrained(a.base,local_files_only=True,trust_remote_code=False)
    label_ids=[tokenizer.encode(str(i),add_special_tokens=False) for i in (0,1)]
    if any(len(v)!=1 for v in label_ids):raise ValueError('Expected one-token labels')
    def tokens(row):
        row=dict(row,focus=row.get('focus',len(row['turns'])-1))
        ids=classifier_tokens(tokenizer,row,[v[0] for v in label_ids])
        if not ids or len(ids)>1024:raise ValueError('No empty or truncated input')
        identity=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
        return identity,len(ids)
    prior_entries={r['token_sha256']:r for r in baseline}
    prior_keys=set(prior_entries);new=[];seen=set();audit={}
    for lang in ('ro','uk'):
        native=root/'native-scope-operators-131'/lang;protected=[]
        for split in ('dev','cal','scope_dev','exposed')+(('stress',) if lang=='uk' else ()):
            protected+=read(native/(split+'.jsonl.gz'))
        protected+=read(root/'paired-scope-117/data'/(lang+'-test.jsonl'))
        forbidden={tokens(r)[0] for r in protected}
        candidates=[r for r in read(native/'train.jsonl.gz') if r['source']=='controlled_composition_v8']
        candidates+=read(root/'indirect-distress-135'/(lang+'-train.jsonl.gz'))
        added=[];retained=0
        for row in candidates:
            key,length=tokens(row)
            if key in forbidden:raise ValueError('Training input overlaps protected evaluation')
            if key in prior_keys:
                if target(row)!=target(prior_entries[key]['row']):raise ValueError('Contradictory retained training label')
                retained+=1;continue
            if key in seen:raise ValueError('Duplicate expansion input')
            seen.add(key)
            entry=dict(partition='train',group=row['source'],row=row,token_sha256=key,token_count=length)
            new.append(entry);added.append(row)
        audit[lang]=dict(candidates=len(candidates),already_in_141=retained,new_rows=len(added),
            new_by_source=dict(Counter(r['source'] for r in added)),protected_token_overlap=0,
            existing_feature_token_overlap=0,reserved_rows_excluded=300)
    new.sort(key=lambda r:(r['token_count'],r['row']['language'],r['row']['case_id']))
    references=[]
    for lang,needed in (('ro',48),('uk',52)):
        groups=defaultdict(list)
        for i,row in enumerate(baseline):
            if row['partition']=='train' and row['row']['language']==lang:groups[row['token_count']].append(i)
        selected=[]
        for length,indices in sorted(groups.items(),reverse=True):
            for start in range(0,len(indices)-3,4):
                if len(selected)<needed:selected.extend(indices[start:start+4])
        if len(selected)!=needed:raise ValueError('Insufficient length-matched reference rows')
        references.extend(selected)
    raw=''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in new).encode()
    (out/'inputs.jsonl.gz').write_bytes(gzip.compress(raw,mtime=0))
    for name in ('base-manifest.json','software.json'):shutil.copyfile(parent/name,out/name)
    (out/'overlap-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    protocol=dict(study=144,created_utc=datetime.now(timezone.utc).isoformat(),new_rows=len(new),
        sources=['controlled_composition_v8','indirect_distress_v9'],existing_training_rows=3607,
        combined_training_rows=3607+len(new),unchanged_development_rows=5421,
        reference_indices=references,reference_use='100existingtrainingrepresentations only;48RO52UK,exactlength groups of4',
        batch_selection='Compare4then1;requireall100reference relative feature errors<=1e-4;choose4onlyifalsofaster;otherwise1mustpass',
        compute_dtype='float32',maximum_tokens=max(r['token_count'] for r in new),chunk_size=200,
        input_order='Sorted by exact token length for efficient unpadded batches;training later shuffles',
        only_training_inferred=True,reserved_access='Exact token exclusion only',
        synthetic_only=True,independently_annotated=False,stable_release_ready=False,
        release_gates_unchanged=True,batching_deployment_qualified=False)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    sources.update(Path(s) for s in ('training/prepare_gemma4_expansion_v9.py','training/extract_gemma4_expansion_v9.py',
        'training/gemma4_batch_features_v9.py','training/gemma4_readout_v9.py','training/gemma4_prompt_v9.py',
        'training/evaluate_gemma_review_v9.py','training/prepare_instruction_sft_v8.py','training/probe_instruction_v8.py','src/roguard/evidence.py'))
    sources.add(parent/'source-freeze.json');sources.add(parent/'actual/report.json');sources.update(out.glob('*'))
    (out/'source-freeze.json').write_text(json.dumps({str(q):digest(q) for q in sorted(sources)},indent=2)+'\n')
    print(json.dumps(dict(audit=audit,protocol=protocol),indent=2))


if __name__=='__main__':main()
