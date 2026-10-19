"""Freeze prior qualification only after both adapted languages pass development."""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest
from training.gemma4_prompt_v9 import classifier_tokens
from training.prepare_instruction_sft_v8 import target
from training.shared_tail_qualification_v9 import candidate_bindings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=Path('research/v9-stabilization/grounding-tail-fit-177'))
    parser.add_argument('--study', type=Path, default=Path('research/v9-stabilization/grounding-tail-prior-178'))
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    candidate = args.candidate
    cache = root/'shared-tail-corpus-164'
    specs, sources = candidate_bindings(candidate)
    training = json.loads((candidate/'protocol.json').read_text())
    caches = [Path(p) for p in training['caches']]
    manifest = json.loads((cache/'base-manifest.json').read_text())
    if args.base.name != manifest['revision']:
        raise ValueError('Wrong base revision')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base, local_files_only=True, trust_remote_code=False)
    labels = [tokenizer.encode(str(i), add_special_tokens=False) for i in (0, 1)]
    if any(len(v) != 1 for v in labels):
        raise ValueError('Unexpected label tokenization')
    entries = []
    for language in ('ro', 'uk'):
        for partition in ('cal', 'exposed'):
            path = root/'native-scope-operators-131'/language/(partition+'.jsonl.gz')
            sources[str(path)] = digest(path)
            for row in read_rows(path):
                normalized = dict(row, focus=row.get('focus', len(row['turns'])-1))
                tokens = classifier_tokens(tokenizer, normalized, [v[0] for v in labels])
                if not tokens or len(tokens) > 1024:
                    raise ValueError('No empty or truncated inputs')
                gold = int(row['review']) if 'review' in row else target(row)
                group = ('conversational' if row.get('source') == 'v7_training_frame_projection' else 'formal') if partition == 'cal' else 'exposed'
                entries.append(dict(row=normalized, group=group, gold=gold, partition=partition,
                    token_sha256=hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest(), token_count=len(tokens)))
    old = [row for corpus in caches for row in read_rows(corpus/'inputs.jsonl.gz')]
    if {r['token_sha256'] for r in old} & {r['token_sha256'] for r in entries}:
        raise ValueError('Prior input overlaps the training/development corpus')
    if len({r['row']['case_id'] for r in entries}) != len(entries):
        raise ValueError('Duplicate prior case identity')
    entries.sort(key=lambda r: (r['row']['language'], r['token_count'], r['row']['case_id']))
    study = args.study
    study.mkdir(exist_ok=False)
    (study/'inputs.jsonl.gz').write_bytes(gzip.compress(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in entries).encode(), mtime=0))
    protocol = dict(study=int(study.name.rsplit('-', 1)[1]), created_utc=datetime.now(timezone.utc).isoformat(), rows=len(entries),
        populations=['cal', 'exposed'], batch_size=1, development_batch_size=8, boundary=24, compute_dtype='float32',
        thresholds={lang: spec['threshold'] for lang, spec in specs.items()},
        full_development_replay=True, development_score_tolerance=1e-5,
        full_forward_training_reference_relative_tolerance=1e-4,
        portable_head_logit_tolerance=1e-3,
        qualification='Frozen adapter and output mode; no threshold selection or fitting',
        batch_contract='Development batch8 per corpus matches177; prior batch1 matches installed single-example execution',
        reference_use='Same16 training references as163, evaluated with selected language adapters',
        calibration_used_for_selection=False, fresh_test_in_inputs=False, fresh_test_inferred=False,
        synthetic_only=True, independently_annotated=False, installed_execution_verified=False,
        stable_release_ready=False)
    (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    paths = [study/'inputs.jsonl.gz', study/'protocol.json']
    paths += [cache/name for name in ('source-freeze.json', 'inputs.jsonl.gz', 'base-manifest.json', 'software.json', 'common-prefix.json', 'protocol.json', 'actual/report.json')]
    paths += [Path('training')/name for name in ('prepare_grounding_tail_qualification_v9.py', 'qualify_grounding_tail_v9.py', 'composite_tail_cache_v9.py',
        'shared_tail_qualification_v9.py', 'shared_tail_training_v9.py', 'train_shared_tail_v9.py',
        'gemma4_shared_tail_v9.py', 'gemma4_shared_prefix_v9.py', 'shared_tail_cache_v9.py',
        'gemma4_structured_v9.py', 'native_fields_v9.py', 'gemma4_prompt_v9.py',
        'prepare_instruction_sft_v8.py', 'evaluate_gemma_review_v9.py', 'probe_instruction_v8.py',
        'fit_scope_entailment_v9.py', 'diagnose_contextual_complementarity_v9.py', 'extract_gemma4_qualified_scope_v9.py')]
    paths += [Path('src/roguard/evidence.py')]
    paths += [corpus/name for corpus in caches for name in ('source-freeze.json', 'inputs.jsonl.gz', 'actual/report.json')]
    sources.update({str(path): digest(path) for path in paths})
    (study/'source-freeze.json').write_text(json.dumps(sources, indent=2)+'\n')
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == '__main__':
    main()
