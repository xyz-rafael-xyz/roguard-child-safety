import hashlib
import json
import zipfile

import pytest

from training.stage_shared_tail_bundle_v9 import review_metadata
from training.verify_shared_tail_installed_v9 import installed_files


def test_staged_metadata_is_accepted_without_loading_weights(tmp_path):
    from roguard.gemma4_contract import BASE_REVISION
    from roguard.gemma4_review import LORA, Gemma4ReviewModel
    spec = dict(base_revision=BASE_REVISION, boundary=24, lora=LORA, language='uk',
        checkpoint_sha256='a'*64, mode='binary', couplings=None, threshold=.25)
    result = review_metadata(spec)
    path = tmp_path/'review.json'
    path.write_text(json.dumps(result))
    identity = hashlib.sha256(path.read_bytes()).hexdigest()
    model = Gemma4ReviewModel(path, tmp_path/'adapter', tmp_path/'base', expected_sha256=identity)
    assert model.language == 'uk' and model._engine is None
    assert result['adapter_sha256'] == spec['checkpoint_sha256'] and result['threshold'] == .25
    with pytest.raises(ValueError):
        review_metadata(dict(spec, boundary=38))


def test_installed_gate_rejects_wheel_changes_and_checkout_imports(tmp_path):
    package = tmp_path/'roguard'
    package.mkdir()
    module = package/'gemma4_review.py'
    module.write_text('verified bytes')
    wheel = tmp_path/'candidate.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        archive.writestr('roguard/gemma4_review.py', 'verified bytes')
    # This test process has already imported RoGuard from the checkout; even
    # exact files in another directory must not count as installed execution.
    with pytest.raises(ValueError, match='outside the installation'):
        installed_files(wheel, tmp_path)
    module.write_text('altered bytes')
    with pytest.raises(ValueError, match='differs from wheel'):
        installed_files(wheel, tmp_path)
