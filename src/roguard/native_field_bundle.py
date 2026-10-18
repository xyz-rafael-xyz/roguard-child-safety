"""Load one explicitly selected local semantic model from a checked manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .field_pipeline import _local_path
from .native_fields import NativeFieldsModel
from .native_review import prepare_input


def load_bundle(path, *, expected_sha256, device='cpu'):
    """Validate relative assets before constructing the lazy neural runtime.

    A manifest hash establishes artifact identity, not model accuracy or
    approval. This loader never downloads assets or executes remote code.
    """
    path = Path(path).resolve()
    with path.open('rb') as stream:
        raw = stream.read(65_537)
    if len(raw) > 65_536 or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError('Bundle identity or size differs')
    spec = json.loads(raw)
    if (not isinstance(spec, dict) or set(spec) != {'format', 'heads_sha256', 'assets'}
            or spec['format'] != 'roguard-native-field-bundle-v1'
            or not isinstance(spec['assets'], dict)
            or set(spec['assets']) != {'heads', 'artifact', 'checkpoint', 'base', 'manifest'}):
        raise ValueError('Unsupported native field bundle')
    paths = {key: _local_path(path.parent, value) for key, value in spec['assets'].items()}
    return NativeFieldsModel(paths['heads'], paths['artifact'], paths['checkpoint'],
                             paths['base'], paths['manifest'],
                             expected_heads_sha256=spec['heads_sha256'], device=device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--sha256', required=True, help='Trusted manifest SHA256')
    parser.add_argument('--input', type=Path, help='Conversation JSON; defaults to stdin')
    parser.add_argument('--focus', type=int, help='Zero-based focal turn; defaults to last')
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
        prepare_input(payload, args.focus)
        model = load_bundle(args.bundle, expected_sha256=args.sha256, device=args.device)
        print(json.dumps(model.predict(payload, focus=args.focus), ensure_ascii=False))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
