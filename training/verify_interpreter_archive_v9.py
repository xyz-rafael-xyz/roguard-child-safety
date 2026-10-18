"""Restore a published adapter archive and compare every tensor to its source."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True, type=Path)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    import numpy as np
    from safetensors.numpy import load_file
    with tempfile.TemporaryDirectory(prefix='roguard-interpreter-roundtrip-') as temporary:
        restored = Path(temporary)/'restored'
        subprocess.run([sys.executable, '-m', 'training.archive_interpreter_v9', 'restore',
                        '--input', str(args.archive), '--output', str(restored)], check=True)
        original = load_file(str(args.source/'adapters.safetensors'))
        reloaded = load_file(str(restored/'adapters.safetensors'))
        if set(original) != set(reloaded):
            raise ValueError('Restored tensor names differ')
        for key in original:
            left, right = original[key], reloaded[key]
            if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
                raise ValueError('Restored tensor differs: '+key)
        config = json.loads((args.source/'adapter_config.json').read_text())
        expected = {key: config[key] for key in ('fine_tune_type', 'num_layers', 'lora_parameters')}
        if json.loads((restored/'adapter_config.json').read_text()) != expected:
            raise ValueError('Restored adapter configuration differs')
        record = dict(exact_tensor_roundtrip=True, tensor_count=len(original),
                      parameter_count=sum(v.size for v in original.values()),
                      source_safetensors_sha256=hashlib.sha256((args.source/'adapters.safetensors').read_bytes()).hexdigest(),
                      archive_sha256=hashlib.sha256(args.archive.read_bytes()).hexdigest(),
                      stable_release_ready=False,
                      scope='Serialization fidelity only; model behavior requires separate inference checks.')
    args.output.write_text(json.dumps(record, indent=2)+'\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()
