"""Compare retained completed59/65 predictions on identical exposed populations."""
import argparse
import json
from pathlib import Path
from roguard.trajectory_data import file_sha, read_jsonl, write_jsonl

p = argparse.ArgumentParser()
for key in ('root', 'private', 'output'): p.add_argument('--'+key, type=Path, required=True)
a = p.parse_args(); paired_path = a.output.with_name(a.output.stem+'-rows.jsonl')
assert not a.output.exists() and not paired_path.exists()
e = a.root/'research/v10-evidence'
read = lambda path: json.loads(path.read_text())
studies = ('large-response-attention-59', 'mixed-large-response-attention-65')
proof_paths = (a.private/'completed-response-attention-59-reusable-verifier.json',
               a.private/'mixed-response-completion-65-paired-verification.json')
proofs = [read(path) for path in proof_paths]
for study, proof in zip(studies, proofs):
    assert proof['verification'] == 'PASS' and proof['study'] == study
    for name in ('report', 'protocol', 'selection'):
        assert file_sha(e/study/(name+'.json')) == proof[name+'_sha256']
protocols = [read(e/study/'protocol.json') for study in studies]
assert protocols[0]['bindings'] == protocols[1]['bindings']
assert protocols[0]['prompt'] == protocols[1]['prompt'] and protocols[0]['labels'] == protocols[1]['labels']
summaries = {}; records = []
for population in ('historical-development', 'generated-development', 'context34', 'old-diagnostic'):
    populations = []
    for study, proof in zip(studies, proofs):
        path = a.private/study/(population+'-predictions.jsonl')
        assert file_sha(path) == proof['paired_predictions'][population]['predictions_sha256']
        populations.append(read_jsonl(path))
    old, new = populations
    assert [r['case_id'] for r in old] == [r['case_id'] for r in new]
    assert len({r['case_id'] for r in new}) == len(new)
    counts = dict(rows=len(new), previous_errors=0, current_errors=0, fixes=0, introductions=0,
                  both_wrong=0, both_correct=0, changed_predictions=0)
    for before, after in zip(old, new):
        assert {k:v for k,v in before.items() if k not in ('prediction','baseline26')} == {
                k:v for k,v in after.items() if k not in ('prediction','baseline26')}
        assert before['baseline26']['label'] == after['baseline26']['label']
        gold = before['gold']; b = before['prediction']['label']; c = after['prediction']['label']
        counts['previous_errors'] += b != gold; counts['current_errors'] += c != gold
        counts['fixes'] += b != gold and c == gold; counts['introductions'] += b == gold and c != gold
        counts['both_wrong'] += b != gold and c != gold; counts['both_correct'] += b == gold and c == gold
        counts['changed_predictions'] += b != c
        records.append({'population':population, 'case_id':before['case_id'], 'language':before['language'],
                        'gold':gold, 'selected59':b, 'selected65':c})
    assert counts['previous_errors'] == proofs[0]['paired_predictions'][population]['errors']
    assert counts['current_errors'] == proofs[1]['paired_predictions'][population]['errors']
    assert counts['fixes']-counts['introductions'] == counts['previous_errors']-counts['current_errors']
    summaries[population] = counts
summaries['combined-development'] = {
    key:sum(summaries[name][key] for name in ('historical-development','generated-development'))
    for key in summaries['historical-development']}
write_jsonl(paired_path, records)
report = {'verified':True, 'source_sha256':file_sha(__file__),
          'completed59_proof_sha256':file_sha(proof_paths[0]), 'completed65_proof_sha256':file_sha(proof_paths[1]),
          'previous_study':studies[0], 'current_study':studies[1], 'populations':summaries,
          'paired_rows_sha256':file_sha(paired_path), 'total_paired_rows':len(records),
          'scope':'Same exposed development and post-selection diagnostic populations only. Both response models '
                  'remain rejected. No v9-superiority, independent annotation or sealed evaluation claim.',
          'model_inference_performed':False, 'training_changed':False, 'held_out_test_opened':False, 'qualified':False}
with a.output.open('x') as output: output.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
