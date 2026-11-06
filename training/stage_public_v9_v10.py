"""Stage the published v9 adapters for research without private archive access."""
import argparse
import json
from pathlib import Path
import shutil

from roguard.field_pipeline import _local_path
from roguard.gemma4_contract import BASE_REVISION
from roguard.gemma4_review import Gemma4ReviewModel
from roguard.language_bundle import _read_manifest
from roguard.model_install import profile, install, cache_directory
from roguard.trajectory_data import file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--model-cache',type=Path,default=cache_directory())
    a=parser.parse_args()
    if a.output.exists():raise ValueError('Refusing to overwrite staged v9')
    raw,sha=profile('v9-gemma4-20261019')
    path,bundle_sha=install(raw,sha,a.model_cache,download=False)
    bundle=_read_manifest(path,bundle_sha)
    if set(bundle['languages'])!={'ro','uk'}:raise ValueError('Expected bilingual v9 profile')
    components={lang:Gemma4ReviewModel.from_bundle(_local_path(path.parent,entry['path']),expected_sha256=entry['sha256'])
                for lang,entry in bundle['languages'].items()}
    metadata={}
    for lang,component in components.items():
        source=Path('research/v9-stabilization/reported-tail-fit-196/actual')/(lang+'-review.json')
        spec=json.loads(source.read_text())
        if spec['checkpoint_sha256']!=file_sha(component._args[1]):raise ValueError('Public adapter differs from research record')
        metadata[lang]=source
    a.output.mkdir(parents=True);actual=a.output/'actual';actual.mkdir()
    for lang,component in components.items():
        shutil.copyfile(component._args[1],actual/(lang+'-selected.safetensors'))
        shutil.copyfile(metadata[lang],actual/(lang+'-review.json'))
    base=a.output/BASE_REVISION
    base.symlink_to(components['ro']._args[0].resolve(),target_is_directory=True)
    from eval.run_trajectory_v10 import LocalFrozenV9
    LocalFrozenV9(base,a.output,a.output/'binding')
    record={'format':'roguard-public-v9-research-stage-1','profile_sha256':sha,
        'adapters':{lang:file_sha(actual/(lang+'-selected.safetensors')) for lang in components},
        'base_revision':BASE_REVISION,'download_performed':False,'private_archive_required':False}
    (a.output/'staging.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({'candidate':str(a.output),'base':str(base),**record},indent=2))


if __name__=='__main__':main()
