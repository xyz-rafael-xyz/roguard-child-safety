"""Load portable bilingual entailment profiles or existing composite profiles."""
from pathlib import Path
from .field_pipeline import _local_path
from .language_bundle import _read_manifest
from .native_review import _digest

FORMAT='roguard-entailment-review-bundle-v1'


def runtime_extra(path,expected_sha256):
    spec=_read_manifest(Path(path).resolve(),expected_sha256)
    if not isinstance(spec,dict):raise ValueError('Expected model bundle object')
    if spec.get('format') == 'roguard-gemma4-tail-review-bundle-v1':return 'gemma4'
    return 'semantic' if spec.get('format') in {FORMAT,'roguard-native-review-bundle-v1','roguard-native-field-bundle-v1'} else 'fusion'


def load_bundle(path,*,expected_sha256,device='cpu'):
    path=Path(path).resolve();spec=_read_manifest(path,expected_sha256)
    if not isinstance(spec,dict):raise ValueError('Expected model bundle object')
    if spec.get('format')!=FORMAT:
        from .consensus_bundle import load_bundle as existing
        return existing(path,expected_sha256=expected_sha256,device=device)
    if (set(spec)!={'format','kind','artifact','checkpoint','base'} or spec['kind'] not in ('review','support-revision','separate-reasons')
            or not isinstance(spec['artifact'],dict) or set(spec['artifact'])!={'path','sha256'}):raise ValueError('Invalid entailment model bundle')
    artifact=_local_path(path.parent,spec['artifact']['path']);checkpoint=_local_path(path.parent,spec['checkpoint']);base=_local_path(path.parent,spec['base']);expected=_digest(spec['artifact']['sha256'])
    if spec['kind']=='review':
        from .paired_nli import PairedNLIReview as Model
    elif spec['kind']=='support-revision':
        from .policy_entailment import PolicyEntailmentReview as Model
    else:
        from .reason_entailment import ReasonEntailmentReview as Model
    return Model(artifact,checkpoint,base,expected_sha256=expected,device=device)
