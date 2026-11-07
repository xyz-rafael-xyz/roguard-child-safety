"""Load or export checksum-bound experimental committee adapters."""
import argparse
import json
from pathlib import Path
import sys

from .field_pipeline import _local_path
from .gemma4_committee import COMPONENTS, Gemma4ReviewCommittee
from .gemma4_review import Gemma4ReviewModel
from .interpreter import prepare_input
from .language_bundle import _read_manifest
from .native_review import _read, _sha


FORMAT = 'roguard-review-committee-bundle-research-1'


def load_committee(path, *, expected_sha256, base, language):
    """Check all six local adapter identities, then configure one language.

    The pinned base is supplied separately and verified when inference loads it.
    No neural dependencies, downloads or model allocation are needed here.
    """
    path = Path(path).resolve()
    spec = _read_manifest(path, expected_sha256)
    if (not isinstance(spec, dict) or set(spec) != {'format', 'qualified', 'languages'}
            or spec['format'] != FORMAT or spec['qualified'] is not False
            or not isinstance(spec['languages'], dict) or set(spec['languages']) != {'ro', 'uk'}
            or language not in ('ro', 'uk')):
        raise ValueError('Unsupported research committee bundle')
    models = {}
    for lang, components in spec['languages'].items():
        if not isinstance(components, dict) or set(components) != set(COMPONENTS):
            raise ValueError('Incomplete committee component manifest')
        models[lang] = {}
        for name, entry in components.items():
            if not isinstance(entry, dict) or set(entry) != {'artifact', 'adapter', 'artifact_sha256'}:
                raise ValueError('Invalid committee asset declaration')
            model = Gemma4ReviewModel(_local_path(path.parent, entry['artifact']),
                _local_path(path.parent, entry['adapter']), base, expected_sha256=entry['artifact_sha256'])
            if model.language != lang or _sha(_read(model._args[1], 16_000_000)) != model._args[2]:
                raise ValueError('Committee adapter language or checksum differs')
            models[lang][name] = model
    return Gemma4ReviewCommittee(**models[language])


def export_committee_bundle(destination, components):
    """Write reviewed metadata and adapters to a new local research directory.

    ``components`` maps each language and component name to (metadata, adapter).
    The exporter preserves exact bytes and never copies the pretrained base.
    """
    if not isinstance(components, dict) or set(components) != {'ro', 'uk'}:
        raise ValueError('Both committee languages are required')
    prepared = {}; spec = {'format': FORMAT, 'qualified': False, 'languages': {}}
    for lang, entries in components.items():
        if not isinstance(entries, dict) or set(entries) != set(COMPONENTS):
            raise ValueError('All three committee components are required')
        spec['languages'][lang] = {}
        for name, pair in entries.items():
            if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                raise ValueError('Expected metadata and adapter paths')
            artifact, adapter = map(Path, pair)
            metadata = _read(artifact, 65_536); adapter_bytes = _read(adapter, 16_000_000)
            model = Gemma4ReviewModel(artifact, adapter, Path('.'), expected_sha256=_sha(metadata))
            if model.language != lang or _sha(adapter_bytes) != model._args[2]:
                raise ValueError('Committee source adapter language or checksum differs')
            metadata_name = f'{lang}-{name}.json'; adapter_name = f'{lang}-{name}.safetensors'
            prepared[metadata_name] = metadata; prepared[adapter_name] = adapter_bytes
            spec['languages'][lang][name] = {'artifact': metadata_name, 'adapter': adapter_name,
                                           'artifact_sha256': _sha(metadata)}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name, data in prepared.items():
        (destination/name).write_bytes(data)
    path = destination/'committee.json'
    path.write_text(json.dumps(spec, indent=2, sort_keys=True)+'\n', encoding='utf-8')
    return path, _sha(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--input', type=Path, help='Conversation JSON; defaults to stdin')
    parser.add_argument('--focus', type=int)
    args = parser.parse_args()
    try:
        if args.input:
            raw = _read(args.input, 1_000_000)
        else:
            raw = sys.stdin.buffer.read(1_000_001)
            if len(raw) > 1_000_000:
                raise ValueError('Conversation JSON is too large')
        payload = json.loads(raw); prepared = prepare_input(payload, args.focus)
        model = load_committee(args.bundle, expected_sha256=args.sha256, base=args.base,
                               language=prepared[0]['language'])
        # ASCII JSON remains valid Unicode JSON under legacy terminal encodings.
        print(json.dumps(model.predict(payload, focus=args.focus), ensure_ascii=True))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
