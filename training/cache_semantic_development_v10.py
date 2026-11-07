"""Prepare additional development states, excluding all known model-token overlap."""
import argparse
import json
from pathlib import Path
import time

from eval.run_trajectory_v10 import LocalFrozenV9
from roguard.assurance import digest
from roguard.gemma4_attention import capture_with_prefix, replay
from roguard.gemma4_contract import classifier_tokens, PREFIX_TOKENS
from roguard.gemma4_review import _Engine
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl, prefix, input_identity
from roguard.trajectory_model import TrajectoryModel
from roguard.trajectory_neural import prepare_memory
from training.cache_semantic_tail_v10 import AddedReader
from training.composite_tail_cache_v9 import CompositeReader
from training.shared_tail_cache_v9 import write_state
from training.prepare_instruction_sft_v8 import target


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('base', 'candidate', 'added_study', 'added_private', 'reference_study',
                'reference_private', 'selector', 'study', 'private'):
        p.add_argument('--'+key.replace('_', '-'), dest=key, type=Path, required=True)
    p.add_argument('--cache-root',type=Path,default=Path('research/v9-stabilization'))
    a = p.parse_args()
    if a.study.exists() or a.private.exists():
        raise ValueError('Refusing to overwrite development states')
    old = CompositeReader([a.cache_root/name for name in
        ('shared-tail-corpus-164','grounding-tail-corpus-174','referent-tail-corpus-176','reported-tail-corpus-195')])
    added = AddedReader(a.added_study, a.added_private)
    known = {r['token_sha256']:r for r in old.entries+added.entries}
    reference = json.loads((a.reference_study/'report.json').read_text())
    if not reference['complete']:
        raise ValueError('Reference inference is incomplete')
    for lang, artifacts in reference['features'].items():
        for suffix, sha in artifacts.items():
            if file_sha(a.reference_private/(lang+'-'+suffix)) != sha:
                raise ValueError('Reference inference binding differs')
    cases_path = Path('research/v10-evidence/data/dev.jsonl')
    if file_sha(cases_path) != json.loads((a.reference_study/'protocol.json').read_text())['inputs']['dev']:
        raise ValueError('Development data differs from reference inference')
    cases = read_jsonl(cases_path); selector = TrajectoryModel(a.selector)
    if any(r['split'] != 'dev' for r in cases): raise ValueError('Only development is allowed')
    a.study.mkdir(parents=True); a.private.mkdir(parents=True)
    frozen = LocalFrozenV9(a.base, a.candidate, a.study/'v9-binding')
    protocol = {'format':'roguard-semantic-development-tail-cache-1',
        'source_sha256':file_sha(Path(__file__)), 'development_sha256':file_sha(cases_path),
        'amendment_sha256':file_sha(Path('research/v10-evidence/protocol-amendment-12.json')),
        'reference_report_sha256':file_sha(a.reference_study/'report.json'),
        'training_cache_report_sha256':file_sha(a.added_study/'report.json'),
        'selector_sha256':selector.identity, 'old_cache_parents':old.parents,
        'allowed_partition':'dev', 'stable_release_ready':False}
    (a.study/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    (a.study/'source.py').write_bytes(Path(__file__).read_bytes())
    binding = file_sha(a.study/'protocol.json')
    import numpy as np
    entries = []; states = []; exclusions = []; max_error = 0; started = time.monotonic()
    for lang in ('ro','uk'):
        mappings = read_jsonl(a.reference_private/(lang+'-rows.jsonl'))
        lookup = {r['checkpoint_id']:i for i,r in enumerate(mappings) if r['population']=='dev'}
        h_reference = np.load(a.reference_private/(lang+'-features.npy'),mmap_mode='r')
        z_reference = np.load(a.reference_private/(lang+'-original-logits.npy'),mmap_mode='r')
        component = frozen.models[lang]; engine = _Engine(*component._args); mx = engine.mx
        cache = engine.make_cache(engine.model.base)
        mx.eval(engine.model.base.model(mx.array([PREFIX_TOKENS]),cache=cache))
        for case in (r for r in cases if r['language']==lang):
            for point in case['checkpoints']:
                identity = case['case_id']+':'+str(point['focus'])
                payload = prefix(case,point['focus'])
                view,inference = prepare_memory(payload,selector,1200,preserve_complete=True)
                if view['mandatory_overflow'] or len(inference['turns'])>12 or point.get('world') is None:
                    raise ValueError('Unsupported development checkpoint')
                row = {'case_id':identity,**inference,'focus':len(inference['turns'])-1,
                    'attributes':point['world'],'review':point['review'],'partition':'dev',
                    'source':'trajectory_development_v10','synthetic_only':True,'independently_annotated':False}
                if target(row)!=point['review']: raise ValueError('Development policy annotation differs')
                tokens = classifier_tokens(engine.tokenizer,inference); token_sha = digest(tokens)
                if token_sha in known:
                    prior = known[token_sha]
                    if prior['row']['attributes']!=row['attributes'] or target(prior['row'])!=target(row):
                        raise ValueError('Conflicting labels for identical model-token input')
                    exclusions.append({'checkpoint_id':identity,'reason':'existing_'+prior['partition']+'_model_input'})
                    continue
                if not len(PREFIX_TOKENS)<len(tokens)<=1024 or tokens[:len(PREFIX_TOKENS)]!=PREFIX_TOKENS:
                    raise ValueError('Unsupported development token envelope')
                position = lookup[identity]; mapping = mappings[position]
                if mapping['input_digest']!=input_identity(payload) or mapping['inference_input_digest']!=input_identity(inference):
                    raise ValueError('Development reference input differs')
                state = capture_with_prefix(engine.model.base,mx.array([tokens[len(PREFIX_TOKENS):]]),cache,start=24)
                h = replay(engine.model.base,state)[:,0,:]; mx.eval(h)
                error = float(np.max(np.abs(np.asarray(h)[0]-h_reference[position]))); max_error=max(max_error,error)
                if error>1e-5: raise ValueError('Development cache replay differs')
                index = len(entries); path = a.private/('state-'+str(index).zfill(6)+'.npz')
                write_state(path,state,{'token_sha256':token_sha,'source_freeze_sha256':binding})
                entry = {'row':row,'partition':'dev','group':'trajectory-development',
                    'token_sha256':token_sha,'token_count':len(tokens),
                    'original_input_digest':input_identity(payload),'reference_logits':z_reference[position].tolist()}
                entries.append(entry);known[token_sha]=entry
                states.append({'index':index,'sha256':file_sha(path),'bytes':path.stat().st_size})
                del state,h;mx.clear_cache()
        del engine,cache,h_reference,z_reference
        import gc
        gc.collect()
    write_jsonl(a.study/'inputs.jsonl',entries)
    report = {'complete':True,'source_freeze_sha256':binding,'inputs_sha256':file_sha(a.study/'inputs.jsonl'),
        'states':states,'exclusions':exclusions,'eligible_development_inputs':len(entries),
        'maximum_feature_replay_error':max_error,'seconds':time.monotonic()-started,'stable_release_ready':False}
    (a.study/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('states','exclusions')},indent=2))


if __name__=='__main__':main()
