"""The structured research wrapper is explicit and has no default-profile effect."""
import hashlib,json,os
from pathlib import Path
import subprocess,sys
from roguard import experimental_structured as structured
from roguard import experimental_role_residual as residual
from roguard import role_alignment

def test_reviewed_structured_source_is_packaged_byte_identically():
    assert hashlib.sha256(Path(structured.__file__).read_bytes()).hexdigest() == '4cd50de67e072d9382a7ecab1a6c4ab5a332cf145ac2d54ffcbb001b7673291a'
    assert hashlib.sha256(Path(residual.__file__).read_bytes()).hexdigest() == 'f7baa18a4a7eff8949aab420f18a9b30a85b3fc045bd180e0a9d2f94445a963d'
    assert hashlib.sha256(Path(role_alignment.__file__).read_bytes()).hexdigest() == '0c65cfcb800a445c52e9c880475e485ea9d35e8241468c92b7f861c37658ad6a'

def test_normal_import_stays_lazy_and_explicit_import_needs_no_numeric_runtime():
    code='''
import sys
import roguard
assert 'roguard.experimental_structured' not in sys.modules
from roguard import experimental_structured
assert not any(name in sys.modules for name in ('numpy','torch','mlx.core','transformers','safetensors'))
'''
    env=dict(os.environ,PYTHONPATH=str(Path(structured.__file__).resolve().parents[1]))
    subprocess.run([sys.executable,'-S','-c',code],env=env,check=True)

def test_no_structured_default_profile_or_global_loader_is_added():
    root=Path(structured.__file__).parent
    profiles=json.loads((root/'model-profiles.json').read_text())
    assert 'structured' not in json.dumps(profiles)
    public=(root/'__init__.py').read_text()
    assert 'experimental_structured' not in public
    assert not hasattr(structured,'load_model')
