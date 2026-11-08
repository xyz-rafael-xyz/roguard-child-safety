"""Frozen exposed regression inputs and admission after the paired DEV audit."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from roguard.trajectory_data import file_sha, input_identity, prefix, read_jsonl, validate_case
from roguard.trajectory_evaluation import checkpoint_id


SELECTOR_SHA = '81493f522678301dfc3196345cd18a577ad39d3b4cc386260cd10866d85a6eea'
PROSPECTIVE = 'research/v10-evidence/matched-role-attention-prospective-104.json'
PROSPECTIVE_SHA = '5ca39dcc4a57e935626877981f0c9392269105074da7de4bebaed264c451f405'
PROOF = 'matched-role-attention105-106-completion-verification-for107.json'
POPULATIONS = {
    'broad': ('research/v10-evidence/broad-v9-regression/cases.jsonl',
              '09b2d452c9b5585c325b29edf9bfb7e1e853fea8539dc3e67fe57a69d04481c9', 5022),
    'dev': ('research/v10-evidence/data/dev.jsonl',
            'af4234260ae547ae2dcb7230fbdd715f027f2d6efeec86f74039cf1bf05cd409', 348),
    'challenge': ('research/v10-evidence/data/challenge.jsonl',
                  '9debb385a233e6247058988b5f7d92ca0cf6f4160f5e39fb9ae1a45d45d63356', 32),
    'v9-regression': ('research/v10-evidence/data/v9-regression.jsonl',
                      '2c9cff32ac8d202412f4101083af05bbdf5977050c926f0bfb129f875eb57a77', 600),
    'fresh44': ('research/v10-evidence/committee-challenge-data-44/cases.jsonl',
                '28bbca4da6f4f511434c5cdf8ed40d8e2487e89d0a96f3de72b8723b1b1d856f', 72),
}
# Verified byte-for-byte against the published v9.0.0 tag. The research memory
# wrapper is separately frozen in the run protocol; it is not part of v9.
PUBLISHED_RUNTIME = {
    'src/roguard/gemma4_review.py': '259a2a189b74cc8ab31d022e82b5a3b1308648aacfe15a14bd20560546200715',
    'src/roguard/gemma4_attention.py': 'c004be4960fd84c8f3b024d6b996ac36f2f28bb5d9c11004586c3a439d355fa6',
    'src/roguard/gemma4_contract.py': '34d45bfbbd4e03b622031d61b762dc598a56800e75a6e7f21e6f47cf393aa200',
}


def require(value, message):
    if not value:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def require_development_pass(proof, arm):
    """An integrity PASS alone does not admit either model to this stage."""
    require(arm in ('control', 'treatment'), 'Unknown paired arm')
    require(proof.get('verification') == 'PASS'
            and proof.get('decision') == 'DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION'
            and proof.get('model_qualified') is False and proof.get('held_out_test_opened') is False
            and proof.get('encoder_loaded') is False, 'Paired development prerequisite failed')
    require(proof.get('checkpoint_count') == 16 and proof.get('checkpoint_dev_rows_recounted') == 70696
            and proof.get('reload_rows_recounted') == 17674 and proof.get('initial_new_rows_compared') == 6336,
            'Paired audit population incomplete')
    require(arm in proof.get('development_pass_arms', []), 'Requested arm did not pass development')
    record = proof['arms'][arm]
    require(record['decision'] == 'DEVELOPMENT_PASS_REQUIRES_FULL_REGRESSION'
            and set(record['languages']) == {'ro', 'uk'}, 'Arm development record incomplete')
    for lang, count in (('ro', 4202), ('uk', 4635)):
        summary = record['languages'][lang]
        require(summary['rows'] == count, 'Language development count differs')
        for key in ('binary_errors', 'joint_argmax_errors', 'introduced_v9'):
            require(type(summary[key]) is int and summary[key] == 0, 'Nonzero development errors or regressions')


def revalidate_admission(root, private, arm, producer_pid):
    """Recount current retained artifacts, not just a previously written PASS."""
    proof_path = private/PROOF
    require(proof_path.is_file(), 'Paired audit107 is not complete; no regression encoder may start')
    proof_sha = file_sha(proof_path)
    proof = read(proof_path)
    require_development_pass(proof, arm)
    require(file_sha(root/PROSPECTIVE) == PROSPECTIVE_SHA, 'Prospective training plan changed')
    with tempfile.TemporaryDirectory(prefix='roguard-regression-admission-') as temp:
        output = Path(temp)/'verification.json'
        subprocess.run([sys.executable, 'scripts/audit_registered_matched_role_v10.py',
            '--root', str(root), '--private-root', str(private), '--prospective', str(root/PROSPECTIVE),
            '--prospective-sha256', PROSPECTIVE_SHA, '--producer-pid', str(producer_pid),
            '--output', str(output)], cwd=root, check=True)
        require(read(output) == proof, 'Current paired artifacts differ from completed audit107')
    require(file_sha(proof_path) == proof_sha, 'Admission proof changed during recount')
    return proof, proof_sha


def load_inputs(root, selector_path):
    """Keep every checkpoint, even when several share the same selected input."""
    from roguard.trajectory_model import TrajectoryModel
    from roguard.trajectory_neural import prepare_memory
    selector = TrajectoryModel(selector_path, expected_sha256=SELECTOR_SHA)
    cases, inputs = {}, []
    for name, (relative, expected, count) in POPULATIONS.items():
        path = root/relative
        require(file_sha(path) == expected, 'Exposed regression population changed: '+name)
        rows = read_jsonl(path); cases[name] = rows
        seen = set(); checkpoints = 0
        for case in rows:
            validate_case(case)
            for point in case['checkpoints']:
                key = checkpoint_id(case, point)
                require(key not in seen, 'Duplicate regression checkpoint')
                seen.add(key); checkpoints += 1
                payload = prefix(case, point['focus'])
                view, inference = prepare_memory(payload, selector, 1200, preserve_complete=True)
                # Preserve an unsupported input as an abstention, never drop it.
                inputs.append(dict(population=name, checkpoint_id=key, language=case['language'],
                    focus=point['focus'], input_digest=input_identity(payload),
                    inference_input_digest=input_identity(inference), inference=inference,
                    unsupported=bool(view['mandatory_overflow'] or len(inference['turns']) > 12),
                    evidence=[{k: span[k] for k in ('turn', 'start', 'end')} for span in view['spans']]))
        require(checkpoints == count, 'Regression checkpoint count differs: '+name)
    return cases, inputs


def verify_runtime(root):
    for name, expected in PUBLISHED_RUNTIME.items():
        require(file_sha(root/name) == expected, 'Published-v9 runtime source changed: '+name)
