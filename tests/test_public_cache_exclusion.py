from pathlib import Path
import subprocess

from scripts.verify_public_tree import PUBLIC_MODEL_HEADS, forbidden


def test_generated_tensor_caches_are_ignored_and_rejected_from_public_history():
    root = Path(__file__).resolve().parents[1]
    caches = [
        'research/v9-stabilization/shared-tail-corpus-164/actual/states/state-000000.npz',
        'research/v9-stabilization/shared-tail-corpus-164/actual/reference-000001.npz',
        'research/v9-stabilization/shared-tail-prior-166/actual/ro-prior-logits.npz',
    ]
    metadata = 'research/v9-stabilization/shared-tail-corpus-164/actual/report.json'
    result = subprocess.run(['git', 'check-ignore', '--no-index', '--stdin'], cwd=root,
        input='\n'.join(caches+[metadata])+'\n', text=True, capture_output=True, check=True)
    assert set(result.stdout.splitlines()) == set(caches)
    assert all(forbidden(path) for path in caches)
    assert not forbidden(metadata)


def test_only_pinned_public_model_heads_are_exempt_from_tensor_exclusion():
    assert len(PUBLIC_MODEL_HEADS) == 4
    assert all(not forbidden(path) for path in PUBLIC_MODEL_HEADS)
    assert forbidden('src/roguard/v10_models/unreviewed.npz')
