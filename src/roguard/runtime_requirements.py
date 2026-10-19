"""Identify optional dependencies from pinned local bundle manifests."""
from pathlib import Path
from .entailment_bundle import runtime_extra as single_bundle_extra
from .field_pipeline import _local_path
from .language_bundle import LanguageReviewModel, _read_manifest


def runtime_extra(path, expected_sha256):
    path = Path(path).resolve()
    spec = _read_manifest(path, expected_sha256)
    if not isinstance(spec, dict):
        raise ValueError('Expected model bundle object')
    if spec.get('format') != 'roguard-language-review-bundle-v1':
        return single_bundle_extra(path, expected_sha256)
    # This validates both child hashes, language entries and paths without
    # loading any neural model. Nested language dispatch is rejected there.
    bundle = LanguageReviewModel(path, expected_sha256=expected_sha256)
    extras = {single_bundle_extra(_local_path(bundle._root, entry['path']), entry['sha256'])
              for entry in bundle._entries.values()}
    if len(extras) == 1:
        return next(iter(extras))
    return 'fusion,gemma4' if 'gemma4' in extras else 'fusion'
