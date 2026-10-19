"""Freeze additional exact attention states for the new grounding corpus."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from training.diagnose_contextual_complementarity_v9 import read_rows
from training.extract_gemma4_qualified_scope_v9 import digest


def main():
    root = Path('research/v9-stabilization')
    source, old = root/'reported-support-194', root/'shared-tail-corpus-164'
    for path, expected in json.loads((source/'source-freeze.json').read_text()).items():
        if digest(path) != expected:
            raise ValueError('Frozen grounding data differs: '+path)
    rows = read_rows(source/'inputs.jsonl.gz')
    maximum = max(r['token_count'] for r in rows)
    estimate = len(rows)*(maximum*20000+65536)
    if shutil.disk_usage(root).free < estimate+20*1024**3:
        raise ValueError('Insufficient disk for extra states and reserve')
    study = root/'reported-tail-corpus-195'
    study.mkdir(exist_ok=False)
    shutil.copyfile(source/'inputs.jsonl.gz', study/'inputs.jsonl.gz')
    for name in ('base-manifest.json', 'software.json', 'common-prefix.json'):
        shutil.copyfile(old/name, study/name)
    old_protocol = json.loads((old/'protocol.json').read_text())
    protocol = dict(study=195, created_utc=datetime.now(timezone.utc).isoformat(), boundary=24,
        rows=len(rows), partitions=dict(Counter(r['partition'] for r in rows)),
        language_partitions=dict(Counter(r['row']['language']+'_'+r['partition'] for r in rows)),
        batch_size=4, chunk_size=100, compute_dtype='float32',
        common_prefix_tokens=old_protocol['common_prefix_tokens'], reference_indices=old_protocol['reference_indices'],
        reference_use='Same preselected16 training references; no evaluation references',
        estimated_cache_bytes=estimate, maximum_tokens=maximum, protected_token_overlap=0,
        calibration_inferred=False, fresh_test_inferred=False, synthetic_only=True,
        independently_annotated=False, stable_release_ready=False)
    (study/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    paths = list(study.glob('*'))+[source/'source-freeze.json', source/'protocol.json', old/'source-freeze.json', old/'actual/report.json']
    paths += [Path('training')/name for name in ('prepare_reported_tail_corpus_v9.py', 'extract_additional_tail_corpus_v9.py',
        'shared_tail_cache_v9.py', 'gemma4_shared_tail_v9.py', 'gemma4_shared_prefix_v9.py', 'gemma4_prompt_v9.py',
        'prepare_instruction_sft_v8.py', 'evaluate_gemma_review_v9.py', 'probe_instruction_v8.py',
        'extract_gemma4_qualified_scope_v9.py', 'diagnose_contextual_complementarity_v9.py')]
    (study/'source-freeze.json').write_text(json.dumps({str(p): digest(p) for p in paths}, indent=2)+'\n')
    print(json.dumps(protocol, indent=2), flush=True)


if __name__ == '__main__':
    main()
