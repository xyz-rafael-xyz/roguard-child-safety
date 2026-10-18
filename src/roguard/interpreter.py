"""Opt-in offline six-field interpreter for the explicitly selected v9 pilot.

Requires Apple Silicon and the ``mlx`` extra. Nothing is downloaded, no default
model is selected, and outputs never authorize contact or other external action.
"""
from copy import deepcopy
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import tempfile
import threading

from .context import validate as validate_conversation
from .evidence import ONTOLOGY, disposition
from .interpreter_archive import validate as validate_archive, MAX_BYTES
from .interpreter_assets import BASE_MODEL, BASE_REVISION, FILES
from .interpreter_schema import FIELDS, POLICY, aligned_tokens, chat

_INFERENCE_LOCK = threading.RLock()
CONTRACT_DIGEST = hashlib.sha256(json.dumps(dict(fields=FIELDS, ontology=ONTOLOGY,
    policy=POLICY,
    serialization='selected-prefix; canonical role,text keys; NFC text; prior_turns then focal_message; ensure_ascii=False',
    decoder='conditional-greedy-full-vocabulary-validity-float32-v1'),
    sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _read_archive(path, expected):
    if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
        raise ValueError('Expected a lowercase archive SHA-256 digest')
    with Path(path).open('rb') as stream:
        raw = stream.read(MAX_BYTES+1)
    if len(raw) > MAX_BYTES or _sha(raw) != expected:
        raise ValueError('Interpreter archive size or hash differs')
    try:
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as source:
            value = source.read(MAX_BYTES*3+1)
        if len(value) > MAX_BYTES*3:
            raise ValueError('Oversized interpreter archive')
        spec = validate_archive(json.loads(value))
        if spec['base_model'] != BASE_MODEL or spec['base_revision'] != BASE_REVISION:
            raise ValueError('Interpreter runtime requires the pinned 4B base')
        if any(v['dtype'] != 'float32' for v in spec['tensors'].values()):
            raise ValueError('Interpreter runtime requires float32 adapter weights')
        return spec
    except (OSError, EOFError, KeyError, TypeError, AttributeError) as error:
        raise ValueError('Invalid interpreter archive') from error


def _snapshot(base, destination):
    """Copy and hash the same bytes; loaders see only this private inventory."""
    if not Path(base).is_dir():
        raise ValueError('Expected a local Gemma base directory')
    destination.mkdir()
    for name, expected in FILES.items():
        h = hashlib.sha256()
        total = 0
        with (Path(base)/name).open('rb') as source, (destination/name).open('xb') as target:
            while True:
                chunk = source.read(1024*1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > expected['bytes']:
                    raise ValueError('Interpreter base asset exceeds pinned length: '+name)
                h.update(chunk)
                target.write(chunk)
        if total != expected['bytes'] or h.hexdigest() != expected['sha256']:
            raise ValueError('Interpreter base asset hash differs: '+name)
    config = json.loads((destination/'config.json').read_text())
    if config.get('model_file') is not None or config.get('model_type') != 'gemma3':
        raise ValueError('Unexpected interpreter model architecture')


def prepare_input(payload, focus=None):
    if not isinstance(payload, dict) or set(payload) != {'language', 'turns'}:
        raise ValueError('Expected language and turns')
    turns = payload['turns']
    if not isinstance(turns, list) or not turns:
        raise ValueError('Expected nonempty turns')
    if focus is None:
        focus = len(turns)-1
    if type(focus) is not int or not 0 <= focus < len(turns):
        raise ValueError('Invalid focal turn index')
    if focus >= 12:
        raise ValueError('Selected prefix exceeds 12 turns')
    prefix = dict(language=payload['language'], turns=turns[:focus+1])
    validate_conversation(prefix)
    prefix = deepcopy(prefix)
    if prefix['turns'][-1]['role'] != 'minor':
        raise ValueError('Focal turn must be a minor message')
    identity = _sha(json.dumps(prefix, sort_keys=True, ensure_ascii=True, separators=(',', ':')).encode())
    return prefix, focus, identity


class _MLXEngine:
    def __init__(self, spec, base):
        try:
            import base64
            import numpy as np
            from safetensors.numpy import save_file
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError as error:
            raise ImportError('Interpreter requires Apple Silicon and roguard-child-safety[mlx]') from error
        self.mx = mx
        self.make_cache = make_prompt_cache
        with tempfile.TemporaryDirectory(prefix='roguard-interpreter-') as temporary:
            root = Path(temporary)
            _snapshot(base, root/'base')
            adapter = root/'adapter'
            adapter.mkdir()
            tensors = {k: np.frombuffer(base64.b64decode(v['data'], validate=True), dtype='<f4').reshape(v['shape']).copy()
                       for k, v in spec['tensors'].items()}
            if any(not np.isfinite(t).all() for t in tensors.values()):
                raise ValueError('Nonfinite interpreter adapter weight')
            save_file(tensors, str(adapter/'adapters.safetensors'))
            del tensors
            (adapter/'adapter_config.json').write_text(json.dumps(spec['adapter_config']))
            self.model, self.tokenizer = load(str(root/'base'), adapter_path=str(adapter),
                                             tokenizer_config={'trust_remote_code': False}, lazy=False)
            self.model.set_dtype(mx.float32)
            self.model.eval()
            mx.eval(self.model.parameters())
        self.lm = getattr(self.model, 'language_model', self.model)
        if self.lm.model_type != 'gemma3_text':
            raise ValueError('Unexpected text model architecture')
        self.ids = [self.tokenizer.encode(str(i), add_special_tokens=False) for i in range(4)]
        if any(len(v) != 1 for v in self.ids):
            raise ValueError('Class tokens must be single tokens')
        self.ids = [v[0] for v in self.ids]
        # Cache only the invariant policy/template prefix. Never retain a user's
        # conversation in the shared cache or reuse one user's KV state for another.
        templates = [aligned_tokens(self.tokenizer, chat(turns), {f: 0 for f in FIELDS})
                     for turns in ([{'role': 'minor', 'text': 'A'}],
                                   [{'role': 'assistant', 'text': 'B'}, {'role': 'minor', 'text': 'C'}])]
        common = 0
        for values in zip(*(t[:p[0]] for t, p in templates)):
            if len(set(values)) != 1:
                break
            common += 1
        common = min(common, min(p[0] for _, p in templates)-1)
        self.prefix_ids = templates[0][0][:common]
        self.prefix = self.make_cache(self.model)
        if common:
            mx.eval(self.lm.model(mx.array([self.prefix_ids]), cache=self.prefix))
        self.checked = False

    def decode(self, turns):
        mx = self.mx
        tokens, positions = aligned_tokens(self.tokenizer, chat(turns), {f: 0 for f in FIELDS})
        if len(tokens) > 1024:
            return dict(abstention='INTERPRETER_CONTEXT_LIMIT')
        shared = len(self.prefix_ids) if tokens[:len(self.prefix_ids)] == self.prefix_ids else 0
        cache = deepcopy(self.prefix) if shared else self.make_cache(self.model)
        cursor = shared
        values, probabilities = {}, {}
        valid = True
        for field, position in zip(FIELDS, positions):
            hidden = self.lm.model(mx.array([tokens[cursor:position]]), cache=cache)
            hidden = hidden[:, -1:, :]
            logits = (self.lm.model.embed_tokens.as_linear(hidden) if self.lm.tie_word_embeddings else self.lm.lm_head(hidden))[0, -1].astype(mx.float32)
            allowed = self.ids[:len(ONTOLOGY[field])]
            probs = mx.softmax(logits[mx.array(allowed)])
            top = mx.argmax(logits)
            mx.eval(probs, top)
            if not self.checked:
                full = self.model(mx.array([tokens[:position]]))[0, -1].astype(mx.float32)
                check = mx.softmax(full[mx.array(allowed)])
                mx.eval(check)
                if float(mx.max(mx.abs(check-probs)).item()) > .001 or int(mx.argmax(full).item()) != int(top.item()):
                    raise ValueError('Cached and full interpreter inference differ')
            choice = int(mx.argmax(probs).item())
            tokens[position] = allowed[choice]
            cursor = position
            values[field] = ONTOLOGY[field][choice]
            probabilities[field] = [float(v) for v in probs.tolist()]
            valid = valid and int(top.item()) in allowed
        self.checked = True
        return dict(interpretation=values, conditional_field_probabilities=probabilities, valid_label_tokens=valid)


class InterpreterModel:
    """Explicit experimental adapter, with strict output validation and no I/O actions."""
    def __init__(self, archive, base, *, expected_archive_sha256):
        self._spec = _read_archive(archive, expected_archive_sha256)
        self._identity = expected_archive_sha256
        self._base = Path(base)
        self._engine = None

    @property
    def identity(self):
        return self._identity

    def predict(self, payload, *, focus=None):
        prefix, focus, identity = prepare_input(payload, focus)
        with _INFERENCE_LOCK:
            if self._engine is None:
                self._engine = _MLXEngine(self._spec, self._base)
                self._spec = None
            decoded = self._engine.decode(prefix['turns'])
        if not isinstance(decoded, dict):
            raise ValueError('Invalid interpreter result')
        if decoded.get('abstention') == 'INTERPRETER_CONTEXT_LIMIT' and set(decoded) == {'abstention'}:
            values, probabilities, valid = None, None, False
            reason = 'INTERPRETER_CONTEXT_LIMIT'
        else:
            if set(decoded) != {'interpretation', 'conditional_field_probabilities', 'valid_label_tokens'}:
                raise ValueError('Invalid interpreter result fields')
            values = decoded['interpretation']
            probabilities = decoded['conditional_field_probabilities']
            valid = decoded['valid_label_tokens']
            if type(valid) is not bool or not isinstance(values, dict) or set(values) != set(FIELDS):
                raise ValueError('Invalid interpreter schema')
            if not isinstance(probabilities, dict) or set(probabilities) != set(FIELDS):
                raise ValueError('Invalid interpreter probabilities')
            for field in FIELDS:
                p = probabilities[field]
                if values[field] not in ONTOLOGY[field] or not isinstance(p, list) or len(p) != len(ONTOLOGY[field]):
                    raise ValueError('Invalid interpreter field')
                if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p) or abs(sum(p)-1) > 1e-4:
                    raise ValueError('Invalid interpreter probabilities')
                if ONTOLOGY[field][max(range(len(p)), key=p.__getitem__)] != values[field]:
                    raise ValueError('Interpreter label differs from its probabilities')
            reason = 'MODEL_INTERPRETATION_PROPOSAL' if valid else 'INTERPRETER_INVALID_LABEL_TOKEN'
        predicted = int(disposition({k: {'label': v} for k, v in values.items()})[0] != 'CONTINUE') if valid else None
        return dict(format='roguard-interpreter-v1', language=prefix['language'], focus=focus, input_digest=identity,
                    model_identity=self.identity, contract_digest=CONTRACT_DIGEST, predicted=predicted, interpretation=values,
                    conditional_field_probabilities=probabilities, valid_label_tokens=valid,
                    probabilities_are_calibrated=False, decision='clarify' if predicted is None else 'review' if predicted else 'no_signal',
                    reason=reason, review_requested=predicted != 0, text_in_report=False,
                    establishes_safety=False, permits_external_action=False, permits_notification=False)


def main():
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Run an explicitly selected experimental interpreter on one JSON conversation from stdin.')
    parser.add_argument('--archive', required=True)
    parser.add_argument('--archive-sha256', required=True)
    parser.add_argument('--base', required=True)
    parser.add_argument('--focus', type=int)
    args = parser.parse_args()
    text = sys.stdin.read(65537)
    if len(text) > 65536:
        parser.error('Input exceeds 65536 characters')
    payload = json.loads(text)
    model = InterpreterModel(args.archive, args.base, expected_archive_sha256=args.archive_sha256)
    print(json.dumps(model.predict(payload, focus=args.focus), ensure_ascii=False))


if __name__ == '__main__':
    main()
