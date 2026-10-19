"""Prepare separate calibration/regression inference for qualified UK development."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from training.gemma4_prompt_v9 import classifier_tokens
from training.diagnose_contextual_complementarity_v9 import read_rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    root = Path('research/v9-stabilization')
    study = root/'gemma4-qualified-scope-148'
    parent = root/'gemma4-readout-141'
    chosen = json.loads((root/'expanded-scope-gate-147/actual/report.json').read_text())
    if not chosen['complete'] or not chosen['languages']['uk']['selected']['frontier']['counts_pass']:
        raise ValueError('Ukrainian development must qualify before this stage')
    study.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    labels = [tokenizer.encode(str(i), add_special_tokens=False) for i in (0, 1)]
    if any(len(v) != 1 for v in labels):
        raise ValueError('Unexpected label tokenization')
    native = root/'native-scope-operators-131/uk'
    entries = []
    sources = set()
    for population in ('cal', 'exposed'):
        path = native/(population+'.jsonl.gz')
        sources.add(path)
        for row in read_rows(path):
            # Canonical prompt helper supplies a default focus for old regressions.
            normalized = dict(row, focus=row.get('focus', len(row['turns'])-1))
            ids = classifier_tokens(tokenizer, normalized, [v[0] for v in labels])
            if not ids or len(ids) > 1024:
                raise ValueError('No empty or truncated input')
            group = ('conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal') if population == 'cal' else 'exposed'
            entries.append(dict(partition=population, group=group, row=normalized,
                                token_sha256=hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest(), token_count=len(ids)))
    training_hashes = set()
    for name in ('gemma4-readout-141', 'gemma4-expansion-144'):
        path = root/name/'inputs.jsonl.gz'
        sources.add(path)
        training_hashes.update(r['token_sha256'] for r in read_rows(path) if r['partition'] == 'train')
    if training_hashes & {r['token_sha256'] for r in entries}:
        raise ValueError('Calibration/regression overlaps contextual training')
    entries.sort(key=lambda r: (r['token_count'], r['row']['case_id']))
    (study/'inputs.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in entries).encode(), mtime=0))
    for name in ('base-manifest.json', 'software.json'):
        shutil.copyfile(parent/name, study/name)
    references = json.loads((root/'gemma4-expansion-144/protocol.json').read_text())['reference_indices']
    protocol = dict(study=148, created_utc=datetime.now(timezone.utc).isoformat(), language='uk',
                    inference_rows=len(entries), populations=['cal', 'exposed'], reference_indices=references,
                    reference_use='100 previously extracted TRAIN inputs only', chunk_size=200,
                    maximum_tokens=max(r['token_count'] for r in entries), compute_dtype='float32',
                    batch_selection='Compare4then1; all reference relative errors <=1e-4; choose4 only if faster',
                    candidate='Frozen study147 Ukrainian scope gate with study146 blend and study143 semantic head',
                    threshold=chosen['languages']['uk']['selected']['frontier']['threshold'],
                    full_development_score_tolerance=1e-5, training_token_overlap=0,
                    fresh_test_inferred=False, fresh_test_in_inputs=False, release_gates_unchanged=True,
                    synthetic_only=True, independently_annotated=False, stable_release_ready=False)
    (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    sources.update(study.glob('*'))
    sources.update(Path(p) for p in ('training/prepare_gemma4_qualified_scope_v9.py', 'training/extract_gemma4_qualified_scope_v9.py',
        'training/score_gemma4_qualified_scope_v9.py', 'training/gemma4_scope_pipeline_v9.py',
        'training/gemma4_structured_v9.py', 'training/gemma4_readout_v9.py', 'training/gemma4_batch_features_v9.py',
        'training/native_fields_v9.py', 'training/diagnose_contextual_complementarity_v9.py',
        'training/score_scope_gate_v9.py', 'training/quote_scope_v9.py', 'training/gemma4_prompt_v9.py',
        'training/evaluate_gemma_review_v9.py', 'training/prepare_instruction_sft_v8.py',
        'training/probe_instruction_v8.py', 'src/roguard/evidence.py'))
    for name in ('gemma4-readout-141', 'gemma4-structured-143', 'contextual-complementarity-146', 'expanded-scope-gate-147'):
        sources.add(root/name/'source-freeze.json')
        sources.add(root/name/'actual/report.json')
    sources.update((root/'gemma4-structured-143/actual'/p) for p in ('uk-readout.json', 'uk-dev-predictions.jsonl'))
    sources.update((root/'expanded-scope-gate-147/actual'/p) for p in ('uk-scope-gate.json.gz', 'uk-dev-predictions.jsonl'))
    sources.update(native/(p+'-predictions.jsonl') for p in ('dev', 'scope_dev', 'stress', 'cal', 'exposed'))
    sources.add(native/'development.json')
    (study/'source-freeze.json').write_text(json.dumps({str(p): digest(p) for p in sorted(sources)}, indent=2)+'\n')
    print(json.dumps(protocol, indent=2))


if __name__ == '__main__':
    main()
