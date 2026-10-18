"""Export a completed experimental fit; export never approves stable release."""
import hashlib
import json
from pathlib import Path

from roguard.interpreter import CONTRACT_DIGEST
from roguard.review_fusion import FEATURE_DIGEST, ReviewFusion


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path('research/v9-stabilization')
    interpreter = json.loads((root/'interpreter-full-21/selection.json').read_text())
    for language in ('ro','uk'):
        source = root/'review-fusion-36'/language
        if not (source/'audit.json').exists():
            continue
        selection = source/'selection.json'; record = json.loads(selection.read_text())
        audit = json.loads((source/'audit.json').read_text())
        if audit['selection_sha256'] != sha(selection):
            raise ValueError('Fusion selection changed')
        native_path = root/('native-full-22-'+language)/'review.json.gz'
        if record['parent_hashes'][str(native_path.relative_to(root))] != sha(native_path):
            raise ValueError('Native parent changed')
        if record['parent_hashes']['interpreter-full-21/selection.json'] != sha(root/'interpreter-full-21/selection.json'):
            raise ValueError('Interpreter parent changed')
        spec = dict(format='roguard-review-fusion-experiment-v1', language=language,
            native_artifact_sha256=sha(native_path), interpreter_archive_sha256=interpreter['archive_sha256'],
            interpreter_contract_digest=CONTRACT_DIGEST, feature_contract_digest=FEATURE_DIGEST,
            selection_sha256=sha(selection), model={key:record['model'][key] for key in ('mean','scale','coefficients')},
            threshold=record['selected']['threshold'])
        output = root/'review-fusion-38'/language; output.mkdir(parents=True,exist_ok=False)
        artifact = output/'fusion.json'; artifact.write_text(json.dumps(spec,sort_keys=True,separators=(',',':'))+'\n')
        ReviewFusion(artifact,expected_artifact_sha256=sha(artifact))
        evidence = dict(artifact_sha256=sha(artifact), selection_sha256=sha(selection), audit_sha256=sha(source/'audit.json'),
                        parent_counts_pass=audit['counts_pass'], constructor_validated=True, stable_release_ready=False)
        (output/'export.json').write_text(json.dumps(evidence,indent=2)+'\n')
        print(language,json.dumps(evidence),flush=True)


if __name__ == '__main__':
    main()
