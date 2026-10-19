"""Dispatch declared Romanian/Ukrainian inputs to pinned local model bundles."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import threading

from .field_pipeline import _local_path
from .native_review import _digest, prepare_input


def _read_manifest(path, expected):
    with path.open('rb') as source:
        raw = source.read(65_537)
    if len(raw) > 65_536 or hashlib.sha256(raw).hexdigest() != _digest(expected):
        raise ValueError('Language bundle identity or size differs')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate language bundle key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=unique)


class LanguageReviewModel:
    """One loaded language at a time; explicit language, no inferred routing."""

    def __init__(self, path, *, expected_sha256):
        path = Path(path).resolve()
        spec = _read_manifest(path, expected_sha256)
        if (not isinstance(spec, dict) or set(spec) != {'format', 'languages'}
                or spec['format'] != 'roguard-language-review-bundle-v1'
                or not isinstance(spec['languages'], dict)
                or set(spec['languages']) != {'ro', 'uk'}):
            raise ValueError('Expected a Romanian and Ukrainian language bundle')
        entries = {}
        for language, entry in spec['languages'].items():
            if (not isinstance(entry, dict) or set(entry) != {'path', 'sha256', 'device'}
                    or entry['device'] not in ('cpu', 'mps', 'cuda')):
                raise ValueError('Invalid language model configuration')
            child = _local_path(path.parent, entry['path'])
            child_spec = _read_manifest(child, entry['sha256'])
            if (not isinstance(child_spec, dict)
                    or child_spec.get('format') == 'roguard-language-review-bundle-v1'):
                raise ValueError('Nested language dispatch is not supported')
            entries[language] = dict(path=entry['path'], sha256=entry['sha256'], device=entry['device'])
        self.identity = expected_sha256
        self._root = path.parent
        self._entries = entries
        self._active_language = None
        self._model = None
        self._lock = threading.RLock()

    def _load(self, language):
        if language != self._active_language:
            # Release the previous model before loading the next large model.
            self._model = None
            self._active_language = None
            from .model_bundle import load_bundle
            entry = self._entries[language]
            child = _local_path(self._root, entry['path'])
            self._model = load_bundle(child, expected_sha256=entry['sha256'], device=entry['device'])
            self._active_language = language
        return self._model

    def predict(self, payload, *, focus=None):
        return self.predict_many([payload], focuses=[focus])[0]

    def predict_many(self, payloads, *, focuses=None):
        if not isinstance(payloads, list) or not 1 <= len(payloads) <= 128:
            raise ValueError('Expected one to 128 conversations')
        if focuses is None:
            focuses = [None] * len(payloads)
        if not isinstance(focuses, list) or len(focuses) != len(payloads):
            raise ValueError('Focus count differs')
        # Validate and copy the complete batch before any inference or loading.
        selected = list(focuses)
        prepared = [prepare_input(p, f) for p, f in zip(payloads, selected)]
        inputs = [dict(language=p['language'], turns=deepcopy(p['turns'][:f + 1]))
                  for p, (_, f, _) in zip(payloads, prepared)]
        result = [None] * len(inputs)
        with self._lock:
            for language in dict.fromkeys(p['language'] for p in inputs):
                indices = [i for i, p in enumerate(inputs) if p['language'] == language]
                model = self._load(language)
                values = model.predict_many([inputs[i] for i in indices], focuses=[selected[i] for i in indices])
                if not isinstance(values, list) or len(values) != len(indices):
                    raise ValueError('Incomplete language model output')
                for i, value in zip(indices, values):
                    if (not isinstance(value, dict) or value.get('language') != language
                            or value.get('focus') != prepared[i][1]
                            or value.get('input_digest') != prepared[i][2]
                            or value.get('model_identity') != model.identity):
                        raise ValueError('Language model output binding differs')
                    if any(value.get(flag) is not False for flag in (
                            'text_in_report', 'establishes_safety',
                            'permits_external_action', 'permits_notification')):
                        raise ValueError('Unsupported language model claim')
                    result[i] = value
                # Do not keep a second reference alive while switching models.
                del model
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--input', type=Path, help='Conversation JSON; otherwise stdin')
    parser.add_argument('--focus', type=int)
    args = parser.parse_args()
    try:
        if args.input:
            with args.input.open('rb') as source:
                raw = source.read(1_000_001)
        else:
            raw = sys.stdin.buffer.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError('Conversation JSON is too large')
        payload = json.loads(raw)
        prepare_input(payload, args.focus)
        model = LanguageReviewModel(args.bundle, expected_sha256=args.sha256)
        print(json.dumps(model.predict(payload, focus=args.focus), ensure_ascii=False))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
