"""Check retained TRAIN drafts with the local tokenizer; never load an encoder."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Refuse overwrite'
    assert a.base.name == '475b9088d29754a3379866cf5aeb6b41acd313c2'
    from transformers import AutoTokenizer
    from roguard.gemma4_response import response_tokens, PROMPT as runtime_prompt
    from training.train_neural_response_v10 import PROMPT
    from training.train_contrast_response_v10 import payload
    from training.probe_assertion_query_v10 import tokens_for
    assert PROMPT == runtime_prompt
    source = a.private / 'response-diversity-pilot-69/train-draft.jsonl'
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows) == 128
    tokenizer = AutoTokenizer.from_pretrained(a.base, local_files_only=True)
    groups, records = defaultdict(list), []
    for row in rows:
        item = payload(row)
        trained, _ = tokens_for(tokenizer, PROMPT, item)
        served = response_tokens(tokenizer, item)
        assert trained == served and 0 < len(trained) <= 1024
        assert item['turns'] == [{'role': 'assistant', 'text': row['draft']}]
        groups[row['language']].append(len(trained))
        records.append({'case_id': row['case_id'], 'tokens': len(trained),
                        'token_sha256': hashlib.sha256(json.dumps(trained, separators=(',', ':')).encode()).hexdigest()})
    report = {
        'complete': True, 'checks_pass': True,
        'scope': 'All128 retained draft inputs fit the1024-token envelope and training/runtime query tokens agree exactly. No inference or label validation.',
        'source_sha256': sha(Path(__file__)), 'draft_sha256': sha(source),
        'base_model': 'mlx-community/gemma-4-e4b-it-4bit', 'base_revision': a.base.name,
        'prompt_sha256': hashlib.sha256(PROMPT.encode()).hexdigest(),
        'helpers': {name: sha(a.root / name) for name in (
            'src/roguard/gemma4_response.py', 'training/train_neural_response_v10.py',
            'training/train_contrast_response_v10.py', 'training/probe_assertion_query_v10.py')},
        'tokenizer_files': {name: sha(a.base / name) for name in (
            'tokenizer.json', 'tokenizer_config.json', 'chat_template.jinja') if (a.base / name).is_file()},
        'groups': {lang: {'rows': len(v), 'minimum_tokens': min(v), 'maximum_tokens': max(v)} for lang, v in groups.items()},
        'input_tokens_sha256': hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
        'model_weights_loaded': False, 'candidate_predictions_obtained': False,
        'training_ready': False, 'used_for_training': False,
        'reserved_test_opened': False, 'stable_release_ready': False,
    }
    with a.output.open('x') as stream:
        stream.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
