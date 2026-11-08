"""Dependency-free import and exact-source checks for experimental offset code."""
import hashlib
import sys
from pathlib import Path
from roguard.response_offset import runtime as rt, closure as cl, lexical as lex

HERE = Path(rt.__file__).resolve().parent

def test_packaged_reviewed_source_bytes_and_import_origins():
 import roguard
 package=Path(roguard.__file__).resolve().parent
 assert HERE==package/'response_offset'
 for name,expected in [('runtime.py','ed72c75c8cee05c929511253f76808049bfb8258fa96ca4e190f85a351ea6e58'),('closure.py','a5bb55bf904cb4bd84ab09e7b6db713c2cdc5f25ecf8616f764ea6ad6046d11a'),('lexical.py','443e91dccae7fc2b6051790a58c4eac3755df5798de158086726d99c569366b5')]:
  assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected
 assert rt.__name__=='roguard.response_offset.runtime' and cl.__name__=='roguard.response_offset.closure' and lex.__name__=='roguard.response_offset.lexical'


def test_import_only_has_no_training_or_encoder_side_effects(tmp_path):
 import subprocess
 code="import sys; import roguard.response_offset; from roguard.response_offset import runtime, lexical; assert not any(k=='scipy' or k.startswith('scipy.') or k=='mlx' or k.startswith('mlx.') for k in sys.modules); print(runtime.__file__)"
 done=subprocess.run([sys.executable,'-c',code],cwd=tmp_path,text=True,capture_output=True,check=True)
 assert Path(done.stdout.strip()).resolve()==Path(rt.__file__).resolve()
