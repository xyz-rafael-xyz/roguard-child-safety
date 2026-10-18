"""One local interface for explicitly selected native model bundles."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from .field_pipeline import FieldReviewPipeline, _local_path
from .native_field_bundle import load_bundle as load_fields
from .native_review import NativeReviewModel, prepare_input


def load_bundle(path, *, expected_sha256, device='cpu'):
    """Load a hash-pinned bundle without downloading or selecting a model."""
    path=Path(path).resolve()
    with path.open('rb') as stream: raw=stream.read(65_537)
    if len(raw)>65_536 or hashlib.sha256(raw).hexdigest()!=expected_sha256:
        raise ValueError('Bundle identity or size differs')
    spec=json.loads(raw)
    if not isinstance(spec,dict): raise ValueError('Expected a model bundle object')
    if spec.get('format')=='roguard-native-field-bundle-v1':
        return load_fields(path,expected_sha256=expected_sha256,device=device)
    if spec.get('format')=='roguard-field-bundle-v1':
        return FieldReviewPipeline.from_bundle(path,expected_sha256=expected_sha256,device=device)
    if (set(spec)!={'format','artifact_sha256','assets'}
            or spec['format']!='roguard-native-review-bundle-v1'
            or not isinstance(spec['assets'],dict)
            or set(spec['assets'])!={'artifact','checkpoint','base','manifest'}):
        raise ValueError('Unsupported model bundle')
    assets={key:_local_path(path.parent,value) for key,value in spec['assets'].items()}
    return NativeReviewModel(assets['artifact'],assets['checkpoint'],assets['base'],assets['manifest'],
        expected_artifact_sha256=spec['artifact_sha256'],device=device)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--sha256',required=True,help='Manifest SHA256 from trusted configuration')
    parser.add_argument('--input',type=Path,help='Conversation JSON; otherwise stdin')
    parser.add_argument('--focus',type=int,help='Zero-based focal turn; otherwise final turn')
    parser.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    args=parser.parse_args()
    try:
        if args.input:
            with args.input.open('rb') as stream: raw=stream.read(1_000_001)
        else: raw=sys.stdin.buffer.read(1_000_001)
        if len(raw)>1_000_000: raise ValueError('Conversation JSON is too large')
        payload=json.loads(raw);prepare_input(payload,args.focus)
        model=load_bundle(args.bundle,expected_sha256=args.sha256,device=args.device)
        print(json.dumps(model.predict(payload,focus=args.focus),ensure_ascii=False))
    except (OSError,ValueError) as error: parser.error(str(error))


if __name__=='__main__': main()
