"""Prepare study141 using training-only examples and existing development data."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil

from training.gemma4_prompt_v9 import classifier_tokens
from training.prepare_instruction_sft_v8 import target


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    raw = Path(path).read_bytes()
    if str(path).endswith('.gz'):
        raw = gzip.decompress(raw)
    rows = [json.loads(line) for line in raw.decode().splitlines()]
    return [dict(row, focus=row.get('focus', len(row['turns'])-1)) for row in rows]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    output = root/'gemma4-readout-141'
    output.mkdir(exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    ids = [tokenizer.encode(str(i), add_special_tokens=False) for i in (0, 1)]
    if any(len(v) != 1 for v in ids):
        raise ValueError('Expected one-token labels')
    ids = [v[0] for v in ids]
    sources = set()
    def load(path):
        sources.add(path)
        return read(path)
    def tokens(row):
        result = classifier_tokens(tokenizer, row, ids)
        if not result or len(result) > 1024:
            raise ValueError('No empty or truncated prompts permitted')
        return tuple(result)
    def identity(encoded):
        return hashlib.sha256(json.dumps(encoded, separators=(',', ':')).encode()).hexdigest()
    inputs = []
    audit = {}
    rng = random.Random(1141)
    for lang in ('ro', 'uk'):
        parent = root/'native-scope-operators-131'/lang
        protected = []
        dev = []
        for split in ('dev', 'cal', 'scope_dev', 'exposed') + (('stress',) if lang == 'uk' else ()):
            rows = load(parent/(split+'.jsonl.gz'))
            protected.extend(rows)
            if split in ('dev', 'scope_dev', 'stress'):
                for row in rows:
                    group = ('conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal') if split == 'dev' else split
                    dev.append(dict(partition='dev', group=group, row=row))
        # Access is for exact-token exclusion only. Reserved examples are never
        # written to the feature input artifact or passed to the neural model.
        fresh = load(root/'paired-scope-117'/'data'/(lang+'-test.jsonl'))
        protected_keys = {tokens(r) for r in protected+fresh}
        history = load(parent/'training-history.jsonl.gz')
        if protected_keys.intersection(tokens(r) for r in history):
            raise ValueError('Protected prompt overlaps inherited training')
        candidates = load(parent/'train.jsonl.gz')
        for name in ('indirect-distress-135', 'freeform-scope-136'):
            candidates += load(root/name/(lang+'-train.jsonl.gz'))
        unique = {}
        for row in candidates:
            if row['language'] != lang:
                raise ValueError('Language differs')
            key = tokens(row)
            if key in protected_keys:
                raise ValueError('Training prompt overlaps evaluation')
            if key in unique and target(unique[key]) != target(row):
                raise ValueError('Contradictory training duplicate')
            unique.setdefault(key, row)
        groups = defaultdict(list)
        for row in unique.values():
            groups[(row['source'], target(row))].append(row)
        chosen = []
        for key, rows in sorted(groups.items()):
            rng.shuffle(rows)
            chosen.extend(rows[:50])
        rng.shuffle(chosen)
        inputs.extend(dict(partition='train', group=r['source'], row=r) for r in chosen)
        inputs.extend(dev)
        audit[lang] = dict(candidate_rows=len(candidates), unique_training_rows=len(unique),
            selected_rows=len(chosen), development_rows=len(dev), protected_rows=len(protected),
            reserved_rows_excluded=len(fresh), inherited_protected_token_overlap=0,
            training_protected_token_overlap=0,
            selected_by_source_and_label={str(k):min(50, len(v)) for k,v in sorted(groups.items())})
    for entry in inputs:
        encoded = tokens(entry['row'])
        entry['token_sha256'] = identity(encoded)
        entry['token_count'] = len(encoded)
    raw = ''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in inputs).encode()
    (output/'inputs.jsonl.gz').write_bytes(gzip.compress(raw, mtime=0))
    for name in ('base-manifest.json', 'software.json'):
        shutil.copyfile(root/'gemma4-direct-140'/name, output/name)
    from importlib.metadata import version
    software = json.loads((output/'software.json').read_text())
    software['versions'].update({name:version(name) for name in ('numpy','scikit-learn','scipy','joblib','threadpoolctl','cloudpickle','narwhals')})
    (output/'software.json').write_text(json.dumps(software, indent=2)+'\n')
    (output/'overlap-audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    protocol = dict(study=141, created_utc=datetime.now(timezone.utc).isoformat(), seed=1141,
        architecture='Frozen Gemma4 final contextual vector plus learned binary logistic readout per language',
        motivation='Test contextual representations separately from instruction following and output token choice',
        training_sampling='At most50 per source and label, deterministic training-only sampling',
        normalization='Per-row L2 norm multiplied by sqrt(hidden dimension); no evaluation-fitted statistics',
        candidates_C=[.001,.01,.1,1.,10.], class_weight='balanced', max_iter=2000,
        selection='Per-language development only: at least99% recall in each group, minimize worst false-review rate, sum rates, misses, then smaller C',
        foundation_model_frozen=True, compute_dtype='float32', max_input_tokens=max(x['token_count'] for x in inputs),
        malformed_generation_policy='Not a generative predictor; readout always returns a binary classifier score. Study139/140 results remain unchanged.',
        prior_calibration_and_exposed_inference='Deferred to a separate verifier if all development groups pass',
        reserved_inference=False, reserved_access='Token overlap exclusion only, no representation extraction',
        synthetic_only=True, independently_annotated=False, stable_release_ready=False,
        release_gates_unchanged=True, feature_chunk_size=200,
        representation_parity='First two inputs: cached vs full final features relative max error<=0.0001')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    sources.update(Path(p) for p in (
        'training/prepare_gemma4_readout_v9.py', 'training/train_gemma4_readout_v9.py',
        'training/gemma4_readout_v9.py', 'training/gemma4_prompt_v9.py',
        'training/evaluate_gemma_review_v9.py', 'training/prepare_instruction_sft_v8.py',
        'training/probe_instruction_v8.py', 'training/budgeted_native_frontier_v9.py',
        'training/entailment_frontier_v9.py', 'src/roguard/evidence.py'))
    sources.update(output.glob('*'))
    (output/'source-freeze.json').write_text(json.dumps({str(p):digest(p) for p in sorted(sources)}, indent=2)+'\n')
    print(json.dumps(dict(audit=audit, protocol=protocol), indent=2))


if __name__ == '__main__':
    main()
