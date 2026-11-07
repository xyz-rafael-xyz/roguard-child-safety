"""Recount the captured selected epoch4 DEV logits without model inference."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Refuse overwrite'
    import numpy as np
    from roguard.trajectory_model import RESPONSE_LABELS
    from training.train_large_response_attention_v10 import load_inputs
    from training.train_large_response_v10 import population_metrics
    study = a.root / 'research/v10-evidence/mixed-large-response-attention-65'
    snapshot_path = a.private / 'mixed-response-epoch4-progress-65.json'
    logits_path = a.private / 'mixed-response-epoch4-selected-dev-logits-65.npy'
    progress_proof_path = a.private / 'mixed-response-epoch4-65-verification.json'
    snapshot = json.loads(snapshot_path.read_text())
    progress_proof = json.loads(progress_proof_path.read_text())
    protocol = json.loads((study / 'protocol.json').read_text())
    command_path = a.private / 'mixed-large-response-attention-65-command.json'
    command = json.loads(command_path.read_text())
    args = SimpleNamespace(**{command[i][2:].replace('-', '_'): Path(command[i + 1]) for i in range(2, len(command), 2)})
    rows, features, train, dev, identities, _, _, bindings = load_inputs(args)
    del features
    assert bindings == protocol['bindings']
    assert identities == [json.loads(line) for line in (study / 'feature-inputs.jsonl').read_text().splitlines()]
    assert progress_proof['verification'] == 'PASS' and progress_proof['progress_sha256'] == sha(snapshot_path)
    assert progress_proof['protocol_sha256'] == sha(study / 'protocol.json')
    assert snapshot['selected']['epoch'] == 4 and [r['epoch'] for r in snapshot['epochs']] == [0, 1, 2, 3, 4]
    weights_path = a.private / 'mixed-large-response-attention-65/epoch-4.safetensors'
    assert sha(weights_path) == snapshot['selected']['adapter_sha256']
    # Capture-time pointer check: fail if selection advanced before this record.
    assert sha(study / 'progress.json') == sha(snapshot_path)
    assert sha(a.private / 'mixed-large-response-attention-65/selected.safetensors') == sha(weights_path)
    logits = np.load(logits_path, allow_pickle=False)
    assert logits.shape == (len(dev), len(RESPONSE_LABELS)) == (592, 5)
    assert np.isfinite(logits).all()
    dev_rows = [rows[i] for i in dev]
    reproduced = population_metrics(dev_rows, logits)
    assert reproduced == snapshot['selected']['development']
    prediction = logits.argmax(axis=1)
    counts, errors = Counter(), Counter()
    for row, pred in zip(dev_rows, prediction):
        key = (row['origin'], row['language'] + ':' + row['gold'])
        counts[key] += 1
        errors[key] += RESPONSE_LABELS[int(pred)] != row['gold']
    for origin, metrics in reproduced['populations'].items():
        for group, count in metrics['groups'].items():
            assert count == {'errors': errors[(origin, group)], 'total': counts[(origin, group)]}
    assert sum(errors.values()) == reproduced['errors'] == 12
    assert sha(study / 'progress.json') == sha(snapshot_path)
    proof = {'complete': True, 'verified': True, 'source_sha256': sha(Path(__file__)),
             'snapshot_sha256': sha(snapshot_path), 'progress_verification_sha256': sha(progress_proof_path),
             'actual_protocol_sha256': sha(study / 'protocol.json'), 'command_sha256': sha(command_path),
             'selected_checkpoint_sha256': sha(weights_path), 'selected_dev_logits_sha256': sha(logits_path),
             'rows': 592, 'errors': 12, 'historical_errors': 4, 'generated_errors': 8,
             'all_saved_group_counts_and_losses_reproduced': True,
             'scope': 'Captured selected epoch4 DEV logits, exact source order/gold and registered metric calculation; direct argmax group recount. No new model inference or paired diagnostic evaluation.',
             'selected_pointer_matched_at_capture': True, 'model_inference_performed': False,
             'training_settings_changed': False, 'held_out_test_opened': False, 'stable_release_ready': False}
    with a.output.open('x') as stream:
        stream.write(json.dumps(proof, indent=2) + '\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__':
    main()
