"""Prepare a training-only semantic augmentation with protected-input exclusion."""
import gzip
import hashlib
import json
from pathlib import Path

from roguard.evidence import ONTOLOGY
from roguard.native_review import prepare_input
from training.scope_repair_data_v9 import generate
from training.verify_reserved_native_v9 import PROBES


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):
    raw=Path(path).read_bytes()
    if str(path).endswith('.gz'):raw=gzip.decompress(raw)
    return [json.loads(line) for line in raw.splitlines()]
def key(row):
    return prepare_input(dict(language=row['language'],turns=row['turns']),row.get('focus',len(row['turns'])-1))[0]


def main():
    root=Path('research/v9-stabilization');probes=[]
    for filename,expected in PROBES.values():
        path=root.parent/'v8-stabilization'/filename
        if sha(path)!=expected:raise ValueError('Reserved overlap-audit file changed')
        probes.extend(read(path))
    for language in ('ro','uk'):
        parent=root/('native-full-22-'+language);output=root/'scope-repair-66/encoders'/language
        output.mkdir(parents=True,exist_ok=False)
        protected={key(row) for split in ('dev','cal') for row in read(parent/(split+'.jsonl.gz'))}
        protected.update(key(row) for row in probes if row['language']==language)
        generated=generate(language);seen={};accepted=[];excluded=[]
        original=read(root/'multitask-review-61'/language/'train.jsonl.gz')
        import random
        random.Random(966).shuffle(original)
        original=original[:6144]
        for row in original+generated:
            if set(row['attributes'])!=set(ONTOLOGY) or any(row['attributes'][f] not in ONTOLOGY[f] for f in ONTOLOGY):
                raise ValueError('Invalid synthetic attributes')
            identity=key(row)
            if identity in protected:
                excluded.append(dict(case_id=row['case_id'],reason='protected_input'));continue
            if identity in seen:
                if seen[identity]!=row['attributes']:raise ValueError('Contradictory semantic training labels')
                excluded.append(dict(case_id=row['case_id'],reason='duplicate'));continue
            seen[identity]=row['attributes'];accepted.append(row)
        for split in ('dev','cal'):
            (output/(split+'.jsonl.gz')).write_bytes((parent/(split+'.jsonl.gz')).read_bytes())
        (output/'train.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in accepted).encode(),mtime=0))
        record=dict(language=language,original_rows=len(original),generated_candidates=len(generated),
            retained_rows=len(accepted),retained_generated=sum(r['source']=='scope_repair_paraphrases_v9' for r in accepted),
            excluded=excluded,generator_sha256=sha(Path('training/scope_repair_data_v9.py')),
            data_sha256={split:sha(output/(split+'.jsonl.gz')) for split in ('train','dev','cal')},
            reserved_access='Normalized-input overlap audit only; no reserved model predictions or training labels used.',
            synthetic_only=True,independently_reviewed=False,stable_release_ready=False)
        (output/'data-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k!='excluded'}),flush=True)


if __name__=='__main__':main()
