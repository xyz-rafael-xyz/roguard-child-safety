"""Run both local encoders and their explicit field mixer as one pipeline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .field_fusion import FieldReviewFusion
from .native_fields import NativeFieldsModel
from .native_review import prepare_input


def _local_path(root, value):
    if not isinstance(value, str) or not value or Path(value).is_absolute() or '..' in Path(value).parts:
        raise ValueError('Bundle assets must use relative paths inside the bundle')
    path = (root / value).resolve()
    if root not in path.parents:
        raise ValueError('Bundle asset escapes its directory')
    return path


class FieldReviewPipeline:
    def __init__(self, fusion, native_model, pretrained_model):
        if (native_model.identity != fusion._spec['native_heads_sha256']
                or pretrained_model.identity != fusion._spec['pretrained_heads_sha256']):
            raise ValueError('Pipeline components differ from selected mixer')
        self.fusion, self.native, self.pretrained = fusion, native_model, pretrained_model
        self.identity = fusion.identity

    @classmethod
    def from_bundle(cls, bundle, *, expected_sha256, device='cpu'):
        bundle = Path(bundle).resolve()
        with bundle.open('rb') as stream:
            raw = stream.read(65_537)
        if len(raw) > 65_536 or hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError('Bundle identity or size differs')
        spec = json.loads(raw)
        if (not isinstance(spec, dict) or set(spec) != {'format', 'fusion', 'components'}
                or spec['format'] != 'roguard-field-bundle-v1'
                or not isinstance(spec['fusion'], dict) or set(spec['fusion']) != {'path', 'sha256'}
                or not isinstance(spec['components'], dict) or set(spec['components']) != {'native', 'pretrained'}):
            raise ValueError('Unsupported local field bundle')
        root = bundle.parent
        fusion = FieldReviewFusion(_local_path(root, spec['fusion']['path']),
                                   expected_sha256=spec['fusion']['sha256'])
        models = []
        for name, identity in (('native', 'native_heads_sha256'), ('pretrained', 'pretrained_heads_sha256')):
            entry = spec['components'][name]
            if not isinstance(entry, dict) or set(entry) != {'heads', 'artifact', 'checkpoint', 'base', 'manifest'}:
                raise ValueError('Invalid local model component')
            paths = {key: _local_path(root, value) for key, value in entry.items()}
            models.append(NativeFieldsModel(paths['heads'], paths['artifact'], paths['checkpoint'],
                paths['base'], paths['manifest'], expected_heads_sha256=fusion._spec[identity], device=device))
        return cls(fusion, *models)

    def predict(self, payload, *, focus=None):
        return self.predict_many([payload], focuses=[focus])[0]

    def predict_many(self, payloads, *, focuses=None):
        if not isinstance(payloads, list) or not 1 <= len(payloads) <= 128:
            raise ValueError('Expected one to 128 conversations')
        if focuses is None:
            focuses = [None] * len(payloads)
        if not isinstance(focuses, list) or len(focuses) != len(payloads):
            raise ValueError('Focus count differs')
        for payload, focus in zip(payloads, focuses):
            prepare_input(payload, focus)
            if payload['language'] != self.fusion._spec['language']:
                raise ValueError('Conversation language differs from pipeline')
        left = self.native.predict_many(payloads, focuses=focuses)
        right = self.pretrained.predict_many(payloads, focuses=focuses)
        if len(left) != len(payloads) or len(right) != len(payloads):
            raise ValueError('Incomplete neural component output')
        return [self.fusion.predict(payload, a, b, focus=focus)
                for payload, a, b, focus in zip(payloads, left, right, focuses)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--sha256', required=True, help='Trusted bundle manifest SHA256')
    parser.add_argument('--input', type=Path, help='Conversation JSON; defaults to standard input')
    parser.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    args = parser.parse_args()
    try:
        if args.input:
            with args.input.open('rb') as stream:
                raw = stream.read(1_000_001)
        else:
            raw = sys.stdin.buffer.read(1_000_001)
        if len(raw) > 1_000_000:
            raise ValueError('Conversation JSON is too large')
        payload = json.loads(raw)
        model = FieldReviewPipeline.from_bundle(args.bundle, expected_sha256=args.sha256, device=args.device)
        print(json.dumps(model.predict(payload), ensure_ascii=False))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
