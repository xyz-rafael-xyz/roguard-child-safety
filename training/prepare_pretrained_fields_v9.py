"""Pin an untouched pretrained encoder with an explicitly unused binary head."""
import argparse
import gzip
import json
from pathlib import Path

from training.train_native_review_v9 import BASES, digest
from roguard.native_assets import reject_native_overrides


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--language', choices=('ro', 'uk'), required=True)
    parser.add_argument('--base', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from safetensors.torch import save_file
    from transformers import AutoModelForSequenceClassification
    root = Path('research/v9-stabilization')
    original = root / ('native-full-22-' + args.language)
    output = root / 'pretrained-fields-47/encoders' / args.language
    model_id, revision, filename, expected = BASES[args.language]
    manifest = json.loads((root / 'native-full-22-base-manifest.json').read_text())['models'][args.language]
    reject_native_overrides(args.base, manifest['files'])
    if digest(args.base / filename) != expected:
        raise ValueError('Original encoder weights differ')
    for name, spec in manifest['files'].items():
        if digest(args.base / name) != spec['sha256']:
            raise ValueError('Pinned pretrained asset differs: ' + name)
    torch.set_num_threads(4)
    torch.manual_seed(947)
    model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=2,
        local_files_only=True, trust_remote_code=False, weights_only=True,
        use_safetensors=filename.endswith('.safetensors')).float().eval()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / 'selected-model.safetensors'
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(checkpoint))
    descriptor = json.loads(gzip.decompress((original / 'review.json.gz').read_bytes()))
    descriptor['checkpoint_sha256'] = digest(checkpoint)
    descriptor['thresholds'] = {args.language: .5}
    (output / 'review.json.gz').write_bytes(gzip.compress(json.dumps(descriptor).encode(), mtime=0))
    for split in ('train', 'dev', 'cal'):
        (output / (split + '.jsonl.gz')).write_bytes((original / (split + '.jsonl.gz')).read_bytes())
    provenance = dict(model=model_id, revision=revision, original_weights_sha256=expected,
        checkpoint_sha256=digest(checkpoint), artifact_sha256=digest(output / 'review.json.gz'),
        seed=947, encoder_updated=False, binary_head_is_untrained_and_unused=True,
        data_sha256={split: digest(output / (split + '.jsonl.gz')) for split in ('train', 'dev', 'cal')},
        purpose='Storage-compatible encoder for six semantic heads; not a binary review candidate.',
        reserved_probes_used=False, stable_release_ready=False)
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print(json.dumps(provenance), flush=True)


if __name__ == '__main__':
    main()
