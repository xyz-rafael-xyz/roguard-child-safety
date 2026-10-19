"""Local loading and bilingual dispatch for scope-consensus review bundles."""
import argparse
import json
from pathlib import Path
import sys
from .field_pipeline import _local_path
from .language_bundle import LanguageReviewModel, _read_manifest
from .native_review import prepare_input
from .scope_consensus import ScopeConsensusReview


def load_bundle(path,*,expected_sha256,device='cpu'):
    path=Path(path).resolve();spec=_read_manifest(path,expected_sha256)
    if not isinstance(spec,dict):raise ValueError('Expected a model bundle object')
    if spec.get('format')=='roguard-language-review-bundle-v1':
        return ConsensusLanguageModel(path,expected_sha256=expected_sha256)
    if spec.get('format')!='roguard-scope-consensus-bundle-v1':
        from .model_bundle import load_bundle as original_load
        return original_load(path,expected_sha256=expected_sha256,device=device)
    if set(spec)!={'format','base','consensus'}:
        raise ValueError('Unsupported consensus bundle')
    for name in ('base','consensus'):
        if not isinstance(spec[name],dict) or set(spec[name])!={'path','sha256'}:
            raise ValueError('Invalid consensus bundle reference')
    base_path=_local_path(path.parent,spec['base']['path'])
    artifact=_local_path(path.parent,spec['consensus']['path'])
    from .three_expert_pipeline import ThreeExpertPipeline
    pipeline=ThreeExpertPipeline.from_bundle(base_path,expected_sha256=spec['base']['sha256'],device=device)
    fusion=ScopeConsensusReview(artifact,pipeline.fusion,expected_sha256=spec['consensus']['sha256'])
    return ThreeExpertPipeline(fusion,pipeline.native,pipeline.semantic,pipeline.interpreter)


class ConsensusLanguageModel(LanguageReviewModel):
    def _load(self,language):
        if language!=self._active_language:
            self._model=None;self._active_language=None
            entry=self._entries[language]
            self._model=load_bundle(_local_path(self._root,entry['path']),
                expected_sha256=entry['sha256'],device=entry['device'])
            self._active_language=language
        return self._model


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,required=True);p.add_argument('--sha256',required=True)
    p.add_argument('--input',type=Path);p.add_argument('--focus',type=int)
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu');a=p.parse_args()
    try:
        if a.input:
            with a.input.open('rb') as stream:raw=stream.read(1_000_001)
        else:raw=sys.stdin.buffer.read(1_000_001)
        if len(raw)>1_000_000:raise ValueError('Conversation JSON is too large')
        payload=json.loads(raw);prepare_input(payload,a.focus)
        model=load_bundle(a.bundle,expected_sha256=a.sha256,device=a.device)
        print(json.dumps(model.predict(payload,focus=a.focus),ensure_ascii=False))
    except (OSError,ValueError) as error:p.error(str(error))


if __name__=='__main__':main()
