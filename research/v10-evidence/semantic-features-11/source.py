"""Extract frozen v9 semantic representations from verified training/dev caches.

No prior regression, calibration or final-test inputs are loaded. Large features
stay in a caller-supplied private directory; the public study retains hashes.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.gemma4_review import _Engine
from roguard.trajectory_data import file_sha, write_jsonl
from training.composite_tail_cache_v9 import CompositeReader
from training.native_fields_v9 import targets
from training.prepare_instruction_sft_v8 import target
from training.shared_tail_cache_v9 import batch_states, to_mlx


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('base','candidate','study','private'):
        p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args()
    if a.study.exists() or a.private.exists():raise SystemExit('Refusing to overwrite extracted representations')
    a.study.mkdir(parents=True);a.private.mkdir(parents=True)
    roots=[Path('research/v9-stabilization')/name for name in
           ('shared-tail-corpus-164','grounding-tail-corpus-174','referent-tail-corpus-176','reported-tail-corpus-195')]
    reader=CompositeReader(roots)
    if any(r['partition'] not in ('train','dev') for r in reader.entries):
        raise ValueError('Only training and development caches are allowed')
    frozen=LocalFrozenV9(a.base,a.candidate,a.study/'v9-binding')
    protocol={'format':'roguard-v10-semantic-representations-1',
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'components':{k:v.identity for k,v in frozen.models.items()},
              'cache_inputs':{str(p/'inputs.jsonl.gz'):file_sha(p/'inputs.jsonl.gz') for p in roots},
              'cache_parent_binding':digest(reader.parents),
              'source_sha256':file_sha(Path(__file__)),
              'representation':'Released v9 adapted final-token hidden state; float32; no encoder updates',
              'forward_batching':'Equal-length cache groups of at most eight; installed single-input parity checked later',
              'allowed_partitions':['train','dev'],'calibration_or_test_loaded':False,
              'stable_release_ready':False}
    (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    import numpy as np
    from roguard.gemma4_attention import replay
    started=time.monotonic();report={'languages':{},'complete':False}
    for lang in ('ro','uk'):
        component=frozen.models[lang];engine=_Engine(*component._args);mx=engine.mx
        indices=[i for i,r in enumerate(reader.entries) if r['row']['language']==lang]
        entries=[reader.entries[i] for i in indices]
        dimensions=engine.model.base.args.hidden_size
        features=np.lib.format.open_memmap(a.private/(lang+'-features.npy'),mode='w+',dtype=np.float32,shape=(len(indices),dimensions))
        logits=np.empty((len(indices),19),dtype=np.float32)
        chunks=[indices[i:i+64] for i in range(0,len(indices),64)]
        with ThreadPoolExecutor(max_workers=1) as pool:
            future=pool.submit(reader.read_many,chunks[0])
            offset=0
            for block,chosen in enumerate(chunks):
                states=future.result()
                if block+1<len(chunks):future=pool.submit(reader.read_many,chunks[block+1])
                grouped=defaultdict(list)
                for i,state in enumerate(states):grouped[state['length']].append(i)
                for group in grouped.values():
                    for pos in range(0,len(group),8):
                        local=group[pos:pos+8]
                        state=batch_states([to_mlx(states[i]) for i in local])
                        hidden=replay(engine.model.base,state)[:,0,:]
                        values=engine.model.heads[lang](hidden);mx.eval(hidden,values)
                        h=np.asarray(hidden);v=np.asarray(values)
                        if not np.isfinite(h).all() or not np.isfinite(v).all():raise ValueError('Nonfinite representation')
                        features[[offset+i for i in local]]=h;logits[[offset+i for i in local]]=v
                        del state,hidden,values,h,v
                offset+=len(chosen);del states;mx.clear_cache()
                if block%10==0 or offset==len(indices):
                    progress={'language':lang,'done':offset,'total':len(indices),'seconds':time.monotonic()-started}
                    (a.study/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
                    print(json.dumps(progress),flush=True)
        features.flush();del features
        np.save(a.private/(lang+'-logits.npy'),logits,allow_pickle=False)
        rows=[]
        for entry in entries:
            row=entry['row']
            rows.append({'case_id':row['case_id'],'language':lang,'partition':entry['partition'],
                         'group':entry['group'],'gold':target(row),'fields':targets([row])[0],
                         'token_sha256':entry['token_sha256']})
        write_jsonl(a.private/(lang+'-rows.jsonl'),rows)
        record={'rows':len(rows),'dimensions':dimensions,
                'partitions':{s:sum(r['partition']==s for r in rows) for s in ('train','dev')},
                'artifacts':{suffix:file_sha(a.private/(lang+'-'+suffix)) for suffix in ('features.npy','logits.npy','rows.jsonl')}}
        report['languages'][lang]=record
        (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        del engine;component._engine=None
        import gc
        gc.collect()
    report.update(complete=True,seconds=time.monotonic()-started,calibration_or_test_loaded=False,stable_release_ready=False)
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
