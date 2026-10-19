"""Gemma 4 direct-answer retry after correcting the wrapper thinking default."""
import argparse
from copy import deepcopy
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import time
from training.prepare_instruction_sft_v8 import messages
from training.gemma4_prompt_v9 import classifier_tokens
from training.gemma4_projection_v9 import last_logits
from training.verify_candidate_pipeline_v9 import counts


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    study=Path('research/v9-stabilization/gemma4-direct-140')
    for name,expected in json.loads((study/'source-freeze.json').read_text()).items():
        if digest(name)!=expected:raise ValueError('Source freeze differs: '+name)
    manifest=json.loads((study/'base-manifest.json').read_text())
    if a.base.name!=manifest['revision']:raise ValueError('Wrong base revision')
    for name,entry in manifest['files'].items():
        if digest(a.base/name)!=entry['sha256']:raise ValueError('Base file differs: '+name)
    from importlib.metadata import version,distribution
    software=json.loads((study/'software.json').read_text())
    for name,expected in software['versions'].items():
        if version(name)!=expected:raise ValueError('Runtime version differs: '+name)
    package=Path(distribution('mlx-lm').locate_file('mlx_lm'))
    for name,expected in software['implementation_sha256'].items():
        if digest(package/name)!=expected:raise ValueError('MLX implementation differs: '+name)
    output=study/'actual';output.mkdir(exist_ok=False)
    import mlx.core as mx
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    mx.set_memory_limit(10*1024**3);mx.set_cache_limit(256*1024**2)
    started=time.monotonic()
    model,tok=load(str(a.base),tokenizer_config={'trust_remote_code':False})
    model.set_dtype(mx.float32)
    print('LOADED',time.monotonic()-started,flush=True)
    ids=[tok.encode(str(i),add_special_tokens=False) for i in (0,1)]
    if any(len(v)!=1 for v in ids):raise ValueError('Class labels must each be one token')
    ids=[v[0] for v in ids]
    rows=[json.loads(l) for l in (study/'screen.jsonl').read_text().splitlines()]
    tokens=[classifier_tokens(tok,r,ids) for r in rows]
    expected_tokens=json.loads((study/'preflight.json').read_text())['token_inputs_sha256']
    for row,encoded in zip(rows,tokens):
        identity=hashlib.sha256(json.dumps(encoded,separators=(',',':')).encode()).hexdigest()
        if identity!=expected_tokens[row['case_id']]:raise ValueError('Prepared/runtime prompt tokens differ')
    if any(not v or len(v)>1024 for v in tokens):raise ValueError('No empty/truncated input permitted')
    common=0
    for columns in zip(*tokens):
        if len(set(columns))!=1:break
        common+=1
    common=min(common,min(map(len,tokens))-1)
    cache=make_prompt_cache(model)
    if common:mx.eval(model.language_model.model(mx.array([tokens[0][:common]]),cache=cache))
    values=[];maximum=0.0
    for i,(row,encoded) in enumerate(zip(rows,tokens)):
        tick=time.monotonic()
        logits=last_logits(model,mx.array([encoded[common:]]),cache=deepcopy(cache))[0,-1].astype(mx.float32)
        pair=mx.softmax(logits[mx.array(ids)]);top=mx.argmax(logits);mx.eval(pair,top)
        score=float(pair[1].item());top=int(top.item())
        if i<2:
            full=model(mx.array([encoded]))[0,-1].astype(mx.float32)
            other=mx.softmax(full[mx.array(ids)]);full_top=mx.argmax(full);mx.eval(other,full_top)
            error=abs(score-float(other[1].item()));maximum=max(maximum,error)
            if error>.001 or int(full_top.item())!=top:raise ValueError('Cached/full model parity differs')
            del full,other,full_top
        valid=top in ids;predicted=int(score>=.5) if valid else None
        result=dict(case_id=row['case_id'],language=row['language'],scope=row['scope'],gold=row['review'],
            predicted=predicted,score=score,top_token_id=top,valid_label_token=valid,seconds=time.monotonic()-tick)
        values.append(result)
        (output/'predictions.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
        print(i+1,json.dumps(result),flush=True)
        del logits,pair
        mx.clear_cache()
    metrics={lang:counts([v for v in values if v['language']==lang]) for lang in ('ro','uk')}
    correct={lang:sum(v['gold']==v['predicted'] for v in values if v['language']==lang) for lang in ('ro','uk')}
    report=dict(complete=True,finished_utc=datetime.now(timezone.utc).isoformat(),metrics=metrics,correct_per_language=correct,
        screen_warrants_expansion=all(v>=18 for v in correct.values()),screen_promotion_rule='At least18of20correct per language; not a release gate',
        source_freeze_sha256=digest(study/'source-freeze.json'),base_manifest_sha256=digest(study/'base-manifest.json'),
        screen_sha256=digest(study/'screen.jsonl'),rows=len(values),maximum_cache_full_score_error=maximum,
        shared_prefix_tokens=common,maximum_input_tokens=max(map(len,tokens)),peak_mlx_bytes=mx.get_peak_memory(),
        seconds=time.monotonic()-started,compute_dtype='float32',thinking_enabled=False,fixed_threshold=.5,invalid_top_tokens_count_as_errors=True,
        synthetic_only=True,independently_annotated=False,fresh_test_read=False,stable_release_ready=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print('FINISHED',json.dumps(report),flush=True)


if __name__=='__main__':main()
