import hashlib
import json
import zipfile

import pytest

from scripts.verify_gemma4_release import verify_wheels


def wheels(tmp_path, change=None):
    bundle = b'{}'
    sha = hashlib.sha256(bundle).hexdigest()
    download = json.dumps(dict(format='roguard-model-download-v1', id='v9-test',
        bundle=dict(path='bundle.json', sha256=sha), terms=['https://example.org/terms'],
        files=[dict(path='bundle.json', url='https://example.org/bundle.json', sha256=sha, bytes=2)])).encode()
    entry = dict(manifest='model-download-v9-test.json', sha256=hashlib.sha256(download).hexdigest(), runtime='apple-silicon-python312')
    previous = {'roguard/__init__.py': b'original', 'roguard/model.json': b'unchanged-data'}
    current = dict(previous, **{'roguard/model-profiles.json': json.dumps(dict(profiles={'v9': entry, 'v9-test': entry})).encode(),
        'roguard/model-download-v9-test.json': download})
    if change == 'code':current['roguard/__init__.py'] = b'changed'
    if change == 'extra-module':current['roguard/extra.py'] = b'unqualified'
    if change == 'manifest':current['roguard/model-download-v9-test.json'] = b'{}'
    if change == 'catalog':current['roguard/model-profiles.json'] = b'{}'
    if change == 'data':current['roguard/model.json'] = b'changed'
    for filename, version, files in [('qualified.whl','9.0.0a1',previous), ('release.whl','9.0.0',current)]:
        directory='roguard_child_safety-'+version+'.dist-info/'
        dependency='mlx==0.32.2' if not (change == 'dependency' and filename == 'release.whl') else 'mlx>=0.32.2'
        metadata=f'Name: roguard-child-safety\nVersion: {version}\nRequires-Python: >=3.9\nRequires-Dist: {dependency}\n\n'
        files[directory+'METADATA'] = metadata.encode()
        files[directory+'entry_points.txt'] = b'roguard = roguard.cli:main'
        if change == 'command' and filename == 'release.whl':files[directory+'entry_points.txt'] = b'roguard = other:main'
        if change == 'outside' and filename == 'release.whl':files['unexpected.py'] = b'outside'
        with zipfile.ZipFile(tmp_path/filename,'w') as archive:
            for name, raw in files.items():archive.writestr(name,raw)
    return tmp_path/'qualified.whl',tmp_path/'release.whl',download


def test_stable_wheel_can_add_only_bound_download_files_and_version_metadata(tmp_path):
    result=verify_wheels(*wheels(tmp_path))
    assert result['unchanged_package_files']==2
    assert result['added_model_files']==['roguard/model-download-v9-test.json','roguard/model-profiles.json']


@pytest.mark.parametrize('change',['code','data','extra-module','manifest','catalog','dependency','command','outside'])
def test_unqualified_release_changes_are_rejected(tmp_path,change):
    with pytest.raises(ValueError):verify_wheels(*wheels(tmp_path,change))
