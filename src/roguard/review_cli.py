"""Review a Romanian or Ukrainian conversation with an installed model profile."""
import argparse
import json
from pathlib import Path
import sys
from .model_install import cache_directory, install, profile
from .native_review import prepare_input


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input',nargs='?',type=Path,help='Conversation JSON; defaults to stdin')
    parser.add_argument('--text',help='One message to review locally')
    parser.add_argument('--language',choices=('ro','uk'),help='Required with --text')
    parser.add_argument('--focus',type=int,help='Zero-based focal turn; defaults to final turn')
    parser.add_argument('--profile',default='v9')
    parser.add_argument('--v10',action='store_true',help='Use the packaged opt-in v10 joint review model')
    parser.add_argument('--cache',type=Path,default=cache_directory())
    args=parser.parse_args()
    extra="gemma4-v10" if args.v10 else "fusion"
    try:
        if args.text is not None:
            if args.input is not None or args.language is None:
                raise ValueError('Use --text with --language ro or uk, without an input file')
            payload=dict(language=args.language,turns=[dict(role='minor',text=args.text)])
        else:
            if args.language is not None:raise ValueError('JSON conversations declare their own language')
            if args.input:
                with args.input.open('rb') as stream:raw=stream.read(1_000_001)
            else:raw=sys.stdin.buffer.read(1_000_001)
            if len(raw)>1_000_000:raise ValueError('Conversation JSON is too large')
            payload=json.loads(raw)
        prepare_input(payload,args.focus)
        raw,digest=profile(args.profile)
        bundle,bundle_sha=install(raw,digest,args.cache,download=False)
        if args.v10:
            if args.profile!='v9':raise ValueError('The v10 option requires the pinned v9 base profile')
            from .gemma4_rbf_v10 import Gemma4RBFV10Model
            model=Gemma4RBFV10Model.from_v9_bundle(bundle,language=payload['language'])
        else:
            from .entailment_bundle import load_bundle
            from .runtime_requirements import runtime_extra
            extra=runtime_extra(bundle,bundle_sha)
            model=load_bundle(bundle,expected_sha256=bundle_sha)
        print(json.dumps(model.predict(payload,focus=args.focus),ensure_ascii=False))
    except ImportError:
        parser.error(f'Install the model runtime: pip install --upgrade "roguard-child-safety[{extra}]"')
    except (OSError,ValueError) as error:parser.error(str(error))


if __name__=='__main__':main()
