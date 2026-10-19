"""Stage a local bilingual adapter bundle only after frozen prior qualification."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

from roguard.assurance import policy_digest
from roguard.gemma4_contract import BASE_MODEL, BASE_REVISION, FILES, POLICY_SHA256
from roguard.gemma4_review import FORMAT, LORA, Gemma4ReviewModel
from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest
from training.fit_scope_entailment_v9 import metrics
from training.shared_tail_qualification_v9 import candidate_bindings


def qualification(root, prior=None):
    prior = root/'grounding-tail-prior-178' if prior is None else Path(prior)
    freeze = prior/'source-freeze.json'
    for path, expected in json.loads(freeze.read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen qualification source differs: ' + path)
    report_path = prior/'actual/report.json'
    report = json.loads(report_path.read_text())
    if (report.get('complete') is not True or report.get('prior_counts_pass') is not True
            or report.get('source_freeze_sha256') != digest(freeze)
            or set(report.get('languages', {})) != {'ro', 'uk'}):
        raise ValueError('Completed bilingual prior qualification is required')
    bindings = {str(freeze): digest(freeze), str(report_path): digest(report_path)}
    inputs = read_rows(prior/'inputs.jsonl.gz')
    for language in ('ro', 'uk'):
        path = prior/'actual'/f'{language}-prior-predictions.jsonl'
        predictions = read_rows(path)
        expected = {r['row']['case_id']: r for r in inputs if r['row']['language'] == language}
        if len(predictions) != len(expected) or {r['case_id'] for r in predictions} != set(expected):
            raise ValueError('Prior prediction population differs')
        for row in predictions:
            original = expected[row['case_id']]
            if (row['language'] != language or row['group'] != original['group'] or row['gold'] != original['gold']
                    or type(row['predicted']) is not int or row['predicted'] not in (0, 1)
                    or type(row['score']) not in (int, float) or not math.isfinite(row['score']) or not 0 <= row['score'] <= 1
                    or row['predicted'] != int(row['score'] >= report['languages'][language]['threshold'])):
                raise ValueError('Prior prediction identity or decision differs')
        measured = metrics(predictions)
        if set(measured) != {'formal', 'conversational', 'exposed'} or not all(v['passes'] for v in measured.values()):
            raise ValueError('All prior groups must pass before bundle staging')
        if measured != report['languages'][language]['metrics']:
            raise ValueError('Prior predictions differ from qualification report')
        bindings[str(path)] = digest(path)
    return bindings


def review_metadata(spec):
    if spec['base_revision'] != BASE_REVISION or spec['boundary'] != 24 or spec['lora'] != LORA:
        raise ValueError('Qualified architecture differs from installed runtime')
    return dict(format=FORMAT, language=spec['language'], base_model=BASE_MODEL, base_revision=BASE_REVISION,
        policy_sha256=POLICY_SHA256, policy_digest=policy_digest(), adapter_sha256=spec['checkpoint_sha256'],
        mode=spec['mode'], couplings=spec['couplings'], threshold=spec['threshold'], boundary=24, lora=LORA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, default=Path('research/v9-stabilization/grounding-tail-fit-177'))
    parser.add_argument('--prior', type=Path, default=Path('research/v9-stabilization/grounding-tail-prior-178'))
    args = parser.parse_args()
    root = Path('research/v9-stabilization')
    candidate = args.candidate
    specs, bindings = candidate_bindings(candidate)
    bindings.update(qualification(root, args.prior))
    if args.base.name != BASE_REVISION:
        raise ValueError('Wrong base revision')
    if shutil.disk_usage(args.output.parent).free < sum(v['bytes'] for v in FILES.values())+20*1024**3:
        raise ValueError('Insufficient staging disk reserve')
    args.output.mkdir(exist_ok=False)
    base = args.output/'base'
    base.mkdir()
    for name, expected in FILES.items():
        size, h = 0, hashlib.sha256()
        with (args.base/name).open('rb') as source, (base/name).open('xb') as output:
            while True:
                block = source.read(1024*1024)
                if not block:
                    break
                size += len(block)
                if size > expected['bytes']:
                    raise ValueError('Base asset exceeds expected length')
                h.update(block)
                output.write(block)
        if size != expected['bytes'] or h.hexdigest() != expected['sha256']:
            raise ValueError('Base asset differs: ' + name)
    languages = {}
    for language, spec in specs.items():
        adapter = args.output/f'{language}-adapter.safetensors'
        shutil.copyfile(candidate/'actual'/f'{language}-selected.safetensors', adapter)
        if digest(adapter) != spec['checkpoint_sha256']:
            raise ValueError('Copied adapter differs')
        artifact = args.output/f'{language}-review.json'
        artifact.write_text(json.dumps(review_metadata(spec), indent=2)+'\n')
        Gemma4ReviewModel(artifact, adapter, base, expected_sha256=digest(artifact))
        bundle = args.output/f'{language}-bundle.json'
        bundle.write_text(json.dumps(dict(format='roguard-gemma4-tail-review-bundle-v1',
            artifact_sha256=digest(artifact), assets=dict(artifact=artifact.name, adapter=adapter.name, base='base')), indent=2)+'\n')
        languages[language] = dict(path=bundle.name, sha256=digest(bundle), device='mps')
    bundle = args.output/'bundle.json'
    bundle.write_text(json.dumps(dict(format='roguard-language-review-bundle-v1', languages=languages), indent=2)+'\n')
    report = dict(complete=True, bundle_sha256=digest(bundle), parents=bindings, runtime_policy_digest=policy_digest(),
        installed_execution_verified=False, fresh_test_inferred=False, stable_release_ready=False,
        local_staging_only=True, publication_performed=False)
    (args.output/'staging-evidence.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
