"""Check token-input overlap without neural inference or displaying messages."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile

from roguard.evidence import disposition
from roguard.native_assets import reject_native_overrides
from roguard.native_review import prepare_input
from training.audit_reserved_overlap_v9 import PROBES, read


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def token_identity(value, language, kind):
    # The shared interpreter prompt contains no declared-language field.
    # Including it here could hide a cross-language duplicate of actual tokens.
    identity = value if kind == 'interpreter' else [language,value]
    return sha(json.dumps(identity,sort_keys=True,separators=(',',':')).encode())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',choices=('ro','uk','interpreter'),required=True)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--history',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.kind == 'interpreter':
        from roguard.interpreter_assets import FILES, BASE_REVISION
        from roguard.interpreter_schema import chat,aligned_tokens,FIELDS
        files,revision = FILES,BASE_REVISION
    else:
        manifest = json.loads(Path('research/v9-stabilization/native-full-22-base-manifest.json').read_text())['models'][a.kind]
        files,revision = manifest['files'],manifest['revision']
    if a.base.name != revision:
        raise ValueError('Tokenizer base revision differs')
    reject_native_overrides(a.base,files)
    with tempfile.TemporaryDirectory(prefix='roguard-token-audit-') as temporary:
        snapshot = Path(temporary); hashes = {}
        for name,expected in files.items():
            if name.startswith('model.') or name.endswith('.bin'):
                continue
            raw = (a.base/name).read_bytes()
            if len(raw) != expected['bytes'] or sha(raw) != expected['sha256']:
                raise ValueError('Tokenizer/configuration identity differs')
            (snapshot/name).write_bytes(raw); hashes[name] = sha(raw)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(snapshot,local_files_only=True,trust_remote_code=False)
        def key(row):
            focus = row.get('focus',len(row['turns'])-1)
            if a.kind == 'interpreter':
                tokens,positions = aligned_tokens(tokenizer,chat(row['turns'][:focus+1]),{field:0 for field in FIELDS})
                value = tokens[:positions[0]]
            else:
                pair,_,_ = prepare_input(dict(language=row['language'],turns=row['turns']),focus)
                value = dict(tokenizer(*pair,padding=False,truncation=False))
            return token_identity(value,row['language'],a.kind)
        identities = defaultdict(set); history_rows = 0; languages = set()
        for row in read(a.history):
            if a.kind != 'interpreter' and row['language'] != a.kind:
                raise ValueError('Training history language differs')
            gold = int(disposition({field:{'label':value} for field,value in row['attributes'].items()})[0]!='CONTINUE')
            identities[key(row)].add(gold); history_rows += 1; languages.add(row['language'])
        if not history_rows:raise ValueError('Empty training history')
        result = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),kind=a.kind,base_revision=revision,
            audit_source_sha256=sha(Path(__file__).read_bytes()),
            token_identity_scope='Shared interpreter excludes the declared language, which is absent from its prompt. Native models have separate language scopes.',
            tokenizer_file_sha256=hashes,history_sha256=sha(a.history.read_bytes()),history_rows=history_rows,
            unique_training_token_inputs=len(identities),conflicting_training_binary_targets=sum(len(v)>1 for v in identities.values()),
            probes={},neural_model_loaded=False,reserved_predictions_produced=False,analyst_blind=False,
            message_text_displayed_or_saved=False,stable_release_ready=False,
            scope='Exact tokenizer input identity only, not semantic-family or pretraining independence.')
        combined = set()
        for name,(filename,expected) in PROBES.items():
            path = Path(filename)
            if sha(path.read_bytes()) != expected:raise ValueError('Reserved probe changed')
            keys = [key(row) for row in read(path) if row['language'] in languages]
            combined.update(keys)
            result['probes'][name] = dict(sha256=expected,rows=len(keys),unique_token_inputs=len(set(keys)),
                                         exact_training_token_input_overlap=sum(k in identities for k in keys))
        result['combined_unique_reserved_token_inputs'] = len(combined)
    with a.output.open('x') as stream:
        json.dump(result,stream,indent=2);stream.write('\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
