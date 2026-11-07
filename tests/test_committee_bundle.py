"""Portable artifact integrity and relocation checks for research committees."""
import hashlib
import json
import shutil

import pytest

from roguard.assurance import policy_digest
from roguard.committee_bundle import export_committee_bundle, load_committee
from roguard.gemma4_committee import COMPONENTS
from roguard.gemma4_contract import BASE_MODEL, BASE_REVISION, POLICY_SHA256
from roguard.gemma4_review import FORMAT, LORA


def assets(root):
    root.mkdir(); result = {}
    for lang in ('ro', 'uk'):
        result[lang] = {}
        for name in COMPONENTS:
            adapter = root/f'{lang}-{name}.safetensors'; adapter.write_bytes((lang+name).encode())
            spec = {'format': FORMAT, 'language': lang, 'base_model': BASE_MODEL,
                'base_revision': BASE_REVISION, 'policy_sha256': POLICY_SHA256, 'policy_digest': policy_digest(),
                'adapter_sha256': hashlib.sha256(adapter.read_bytes()).hexdigest(), 'mode': 'binary',
                'couplings': None, 'threshold': .5, 'boundary': 24, 'lora': LORA}
            metadata = root/f'{lang}-{name}.json'; metadata.write_text(json.dumps(spec))
            result[lang][name] = (metadata, adapter)
    return result


def test_export_relocate_and_load_without_neural_dependencies(tmp_path):
    source = assets(tmp_path/'sources')
    path, sha = export_committee_bundle(tmp_path/'export', source)
    relocated = tmp_path/'relocated'; shutil.copytree(path.parent, relocated)
    for lang in ('ro', 'uk'):
        model = load_committee(relocated/path.name, expected_sha256=sha, base=tmp_path/'base', language=lang)
        assert model.language == lang and model._engine is None
        assert all(value[1].parent == relocated for value in model._args.values())
        for name in COMPONENTS:
            assert model.component_identities[name] == hashlib.sha256(source[lang][name][0].read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        export_committee_bundle(path.parent, source)


@pytest.mark.parametrize('problem', ['checksum', 'qualified', 'missing', 'escape', 'symlink', 'metadata', 'adapter', 'language'])
def test_tampering_and_invalid_bundles_are_rejected(tmp_path, problem):
    path, sha = export_committee_bundle(tmp_path/'bundle', assets(tmp_path/'sources'))
    spec = json.loads(path.read_text())
    entry = spec['languages']['uk']['attention']
    if problem == 'qualified':
        spec['qualified'] = True
    elif problem == 'missing':
        del spec['languages']['ro']['readout']
    elif problem == 'escape':
        entry['adapter'] = '../outside.safetensors'
    elif problem == 'symlink':
        outside = tmp_path/'outside'; outside.write_bytes(b'x')
        asset = path.parent/entry['adapter']; asset.unlink()
        try:
            asset.symlink_to(outside)
        except OSError:
            pytest.skip('Platform does not permit symlink creation')
    elif problem in ('metadata', 'adapter'):
        asset = path.parent/entry['artifact' if problem == 'metadata' else 'adapter']
        asset.write_bytes(asset.read_bytes()+b'x')
    elif problem == 'language':
        spec['languages']['uk'], spec['languages']['ro'] = spec['languages']['ro'], spec['languages']['uk']
    path.write_text(json.dumps(spec)); sha = hashlib.sha256(path.read_bytes()).hexdigest() if problem != 'checksum' else sha
    with pytest.raises(ValueError):
        load_committee(path, expected_sha256=sha, base=tmp_path/'base', language='ro')


def test_export_rejects_bad_sources_before_creating_destination(tmp_path):
    sources = assets(tmp_path/'sources')
    sources['uk']['attention'][1].write_bytes(b'corrupt')
    with pytest.raises(ValueError):
        export_committee_bundle(tmp_path/'bundle', sources)
    assert not (tmp_path/'bundle').exists()
