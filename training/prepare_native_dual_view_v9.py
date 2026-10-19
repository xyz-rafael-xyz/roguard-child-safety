"""Freeze and audit a new two-view input definition before native training."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

from training.native_quote_view_v9 import PREPROCESSING, texts
from training.prepare_instruction_sft_v8 import target
from training.train_bare_distress_v9 import BASES, digest, read


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ro-base', type=Path, required=True)
    parser.add_argument('--uk-base', type=Path, required=True)
    args = parser.parse_args()
    from transformers import AutoTokenizer
    from roguard.native_assets import reject_native_overrides
    root = Path('research/v9-stabilization')
    output = root/'native-dual-view-150'
    output.mkdir(exist_ok=False)
    manifest_path = root/'native-full-22-base-manifest.json'
    manifest = json.loads(manifest_path.read_text())['models']
    sources = {manifest_path}
    audits = {}
    training_counts = {}
    for language in ('ro', 'uk'):
        base = getattr(args, language+'_base')
        if base.name != BASES[language][1]:
            raise ValueError('Wrong pretrained revision')
        reject_native_overrides(base, manifest[language]['files'])
        for filename, entry in manifest[language]['files'].items():
            if digest(base/filename) != entry['sha256']:
                raise ValueError('Base file differs: ' + filename)
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
        def tracked(path):
            sources.add(path)
            return read(path) if path.suffix == '.gz' else [json.loads(line) for line in path.read_text().splitlines()]
        parent = root/'native-scope-operators-131'/language
        train = tracked(parent/'train.jsonl.gz')
        for name in ('indirect-distress-135', 'freeform-scope-136'):
            train += tracked(root/name/(language+'-train.jsonl.gz'))
        protected = []
        for split in ('dev', 'cal', 'scope_dev', 'exposed')+(('stress',) if language == 'uk' else ()):
            protected += tracked(parent/(split+'.jsonl.gz'))
        reserved = tracked(root/'paired-scope-117/data'/(language+'-test.jsonl'))
        protected += reserved
        def encode(rows):
            pairs = [texts(row) for row in rows]
            tokens = tokenizer([p[0] for p in pairs], [p[1] for p in pairs], padding=False, truncation=False)['input_ids']
            if any(not values or len(values) > 512 for values in tokens):
                raise ValueError('No empty or truncated native inputs')
            return tokens
        forbidden = {tuple(ids) for ids in encode(protected)}
        seen = {}
        duplicates = []
        lengths = []
        for row, ids in zip(train, encode(train)):
            key = tuple(ids)
            gold = target(row)
            if key in forbidden:
                raise ValueError('Two-view training input overlaps protected evaluation')
            if key in seen:
                if seen[key] != gold:
                    raise ValueError('Contradictory two-view training labels')
                duplicates.append(row['case_id'])
            seen[key] = gold
            lengths.append(len(ids))
        training_counts[language] = len(seen)
        audits[language] = dict(candidates=len(train), unique_training_inputs=len(seen),
            protected_token_overlap=0, excluded_same_label_duplicates=duplicates,
            maximum_training_tokens=max(lengths), reserved_exclusion_rows=len(reserved),
            reserved_inference=False, complete_content_preserved=True)
    protocol = dict(study=150, created_utc=datetime.now(timezone.utc).isoformat(),
        initialization='Original pinned pretrained encoders; new binary head; no previous task checkpoint',
        preprocessing=PREPROCESSING, first_view='Complete conversation through focus, all quote content retained',
        second_view='Focal text with closed quote spans abstracted; attribution and negation remain',
        training_rows=training_counts, epochs=2, evaluation_epochs=[0, .5, 1, 1.5, 2],
        seed=1150, full_encoder_adaptation=True, top_learning_rate=2e-5, layer_decay=.9,
        batch_size=8, optimizer='AdamW', weight_decay=.01, warmup_fraction=.05,
        objective='Binary CE plus0.2 paired margin softplus for opposed training pairs',
        selection='Unchanged per-group99recall; minimize worst false-review rate, sum rates, then misses',
        calibration_used_for_selection=False, reserved_access='Token overlap exclusion only',
        release_gates_unchanged=True, synthetic_only=True, independently_annotated=False,
        stable_release_ready=False, new_runtime_required_before_installed_qualification=True)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    (output/'preflight.json').write_text(json.dumps(audits, indent=2)+'\n')
    packages = ('torch', 'transformers', 'numpy', 'safetensors', 'tokenizers')
    (output/'software.json').write_text(json.dumps({name: version(name) for name in packages}, indent=2)+'\n')
    sources.update(output.glob('*'))
    sources.update(Path(name) for name in ('training/prepare_native_dual_view_v9.py',
        'training/train_native_dual_view_v9.py', 'training/native_quote_view_v9.py',
        'training/quote_scope_v9.py', 'training/train_bare_distress_v9.py',
        'training/native_full_optimizer_v9.py', 'training/budgeted_native_frontier_v9.py',
        'training/entailment_frontier_v9.py', 'training/verify_candidate_pipeline_v9.py',
        'training/prepare_instruction_sft_v8.py', 'src/roguard/context.py', 'src/roguard/semantic.py',
        'src/roguard/native_assets.py', 'src/roguard/evidence.py'))
    (output/'source-freeze.json').write_text(json.dumps({str(path): digest(path) for path in sorted(sources)}, indent=2)+'\n')
    print(json.dumps(dict(protocol=protocol, audit=audits), indent=2))


if __name__ == '__main__':
    main()
