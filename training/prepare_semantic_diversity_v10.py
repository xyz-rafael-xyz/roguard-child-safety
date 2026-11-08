"""Materialize a registered authored TRAIN corpus; no model fitting or admission."""
import argparse
import json
from pathlib import Path

from roguard.assurance import digest
from roguard.gemma4_contract import PREFIX_TOKENS, classifier_tokens
from roguard.trajectory_data import file_sha, write_jsonl
from training.semantic_diversity_v10 import build, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'private_root', 'base', 'plan', 'study', 'output'):
        p.add_argument('--'+name.replace('_', '-'), dest=name, type=Path, required=True)
    a = p.parse_args()
    if a.study.exists() or a.output.exists():
        raise ValueError('Refusing to overwrite generation')
    plan = json.loads(a.plan.read_text())
    if (plan.get('format') != 'roguard-semantic-diversity-plan-1'
            or plan.get('partition') != 'train' or plan.get('training_admitted') is not False):
        raise ValueError('Only registered TRAIN preparation is allowed')
    def check_bindings():
        for base, values in ((a.root, plan['public']), (a.private_root, plan['private'])):
            for name, checksum in values.items():
                if file_sha(base/name) != checksum:
                    raise ValueError('Registered source/input changed: '+name)
    check_bindings()
    if a.base.name != plan['tokenizer_revision']:
        raise ValueError('Tokenizer snapshot differs')
    for name, checksum in plan['tokenizer_files'].items():
        if file_sha(a.base/name) != checksum:
            raise ValueError('Tokenizer artifact differs')
    manifest = json.loads((a.private_root/plan['manifest']).read_text())
    protected = [json.loads(line) for line in (a.private_root/plan['protected_inputs']).open()]
    rows = build(manifest, protected)
    if len(rows) != plan['rows'] or len(manifest['pairs']) != plan['families']:
        raise ValueError('Registered population differs')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
    protected_tokens = {r['token_sha256'] for r in protected}
    seen = set()
    for row in rows:
        tokens = classifier_tokens(tokenizer, row['payload'])
        if tokens[:len(PREFIX_TOKENS)] != PREFIX_TOKENS or not len(PREFIX_TOKENS) < len(tokens) <= plan['maximum_tokens']:
            raise ValueError('Token contract differs; no truncation is permitted')
        key = digest(tokens)
        if key in seen or key in protected_tokens:
            raise ValueError('Duplicate or protected runtime tokens')
        seen.add(key)
        row.update(tokens=tokens, token_sha256=key, token_count=len(tokens))
    check_bindings()
    a.output.mkdir(parents=True); a.study.mkdir(parents=True)
    write_jsonl(a.output/'rows.jsonl', rows)
    summary = summarize(rows)
    report = dict(format='roguard-semantic-diversity-preparation-1', complete=True,
                  source_sha256=file_sha(__file__), plan_sha256=file_sha(a.plan), **summary,
                  rows_sha256=file_sha(a.output/'rows.jsonl'),
                  minimum_tokens=min(r['token_count'] for r in rows),
                  maximum_tokens=max(r['token_count'] for r in rows),
                  exact_existing_core_substring_matches=0, exact_existing_focus_matches=0,
                  exact_existing_token_matches=0, duplicate_tokens=0,
                  protected_rows=len(protected), tokenizer_loaded=True, encoder_loaded=False,
                  training_performed=False, training_admitted=False, independently_annotated=False,
                  held_out_test_opened=False, qualified=False,
                  limitations=['Correlated author-written synthetic variants, not independent meanings.',
                               'Core concern and operator semantics require review; structural checks do not validate gold.',
                               'Only SELF/DIRECT and FICTIONAL/FICTION with CURRENT time are covered.',
                               'Designed after exposed DEV diagnostics; no independent evaluation claim.',
                               'A separately registered fitting experiment and unchanged regression gates remain required.'])
    (a.study/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('complete','rows','families','distinct_cores','minimum_tokens','maximum_tokens','training_admitted')}))


if __name__ == '__main__':
    main()
