"""Export a frozen selection and verify it with the installed response runtime."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from eval.verify_response_attention_runtime_v10 import development_rows
import roguard.gemma4_response_attention as runtime
from roguard.trajectory_data import file_sha, read_jsonl
from scripts.package_response_attention_v10 import export


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('model_study', 'model_private', 'rows', 'reference_artifact',
                 'reference_adapter', 'base', 'artifact', 'study', 'export_record'):
        p.add_argument('--' + name.replace('_', '-'), dest=name, type=Path, required=True)
    p.add_argument('--expected-development-rows', type=int, required=True)
    p.add_argument('--preflight', action='store_true')
    a = p.parse_args()
    for name, value in vars(a).items():
        if isinstance(value, Path):
            setattr(a, name, value.resolve())
    root = Path(__file__).resolve().parents[1]
    installed = Path(runtime.__file__).resolve()
    source = root / 'src/roguard/gemma4_response_attention.py'
    if installed.is_relative_to(root) or file_sha(installed) != file_sha(source):
        raise ValueError('Runtime must be the matching installed package outside the checkout')
    rows = development_rows(read_jsonl(a.rows), read_jsonl(a.model_study / 'feature-inputs.jsonl'))
    if len(rows) != a.expected_development_rows:
        raise ValueError('Development population differs from registration')
    for target in (a.artifact, a.study, a.export_record):
        if target.exists():
            raise ValueError('Refusing to overwrite package verification evidence')
    preflight = {'complete': True, 'installed_runtime_matches_source': True,
                 'runtime_sha256': file_sha(installed), 'rows_sha256': file_sha(a.rows),
                 'feature_inputs_sha256': file_sha(a.model_study / 'feature-inputs.jsonl'),
                 'development_rows': len(rows), 'test_opened': False,
                 'encoder_loaded': False, 'stable_release_ready': False}
    if a.preflight:
        print(json.dumps(preflight, indent=2))
        return
    # Export validates the completed report, immutable selection and weight hashes.
    manifest = export(a.model_study, a.model_private / 'selected.safetensors', a.artifact)
    a.export_record.parent.mkdir(parents=True, exist_ok=True)
    a.export_record.write_text(json.dumps(manifest, indent=2) + '\n')
    command = [sys.executable, str(root / 'eval/verify_response_attention_runtime_v10.py')]
    values = {'artifact': a.artifact / 'response.json', 'weights': a.artifact / 'adapter.safetensors',
              'reference-artifact': a.reference_artifact, 'reference-adapter': a.reference_adapter,
              'base': a.base, 'rows': a.rows, 'model-study': a.model_study,
              'model-private': a.model_private, 'study': a.study,
              'artifact-sha256': manifest['artifacts']['response.json'],
              'reference-sha256': file_sha(a.reference_artifact)}
    for key, value in values.items():
        command.extend(['--' + key, str(value)])
    result = subprocess.run(command, cwd=root.parent, env=dict(os.environ, PYTHONPATH=str(root)), check=False)
    raise SystemExit(result.returncode)


if __name__ == '__main__':
    main()
