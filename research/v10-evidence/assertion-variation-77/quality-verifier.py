"""Verify the curated slice is an exact, complete subset of audited records."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import runpy


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def main():
    p = argparse.ArgumentParser()
    for name in ('root', 'private', 'output'): p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists(): raise ValueError('Refusing to overwrite verification')
    study = a.root/'research/v10-evidence/assertion-variation-77'
    plan_path = study/'quality-plan.json'; plan = json.loads(plan_path.read_text())
    source = study/'quality-selection-source.py'
    assert sha(source) == plan['selection_source_sha256']
    allow = runpy.run_path(str(source))['ALLOW']  # Explicit curation data only.
    for name, expected in plan['private'].items(): assert sha(a.private/name) == expected
    base = a.private/'assertion-variation-77'; curated = base/'curated'
    report = json.loads((curated/'report.json').read_text())
    assert report['complete'] and report['source_sha256'] == sha(source) and report['plan_sha256'] == sha(plan_path)
    for name, expected in report['artifacts'].items(): assert sha(curated/name) == expected
    original = read(base/'actual/additional-train.jsonl'); decisions = read(curated/'parent-decisions.jsonl')
    by_parent = {}
    for row in original:
        key = (row['language'], row['parent']['case_id'])
        by_parent.setdefault(key, row)
    assert len(decisions) == len(by_parent)
    chosen = set(); exclusions = Counter(); decision_keys = set()
    for decision in decisions:
        key = (decision['language'], decision['parent_case_id']); row = by_parent[key]
        assert key not in decision_keys; decision_keys.add(key)
        core = row['turns'][row['focus']]['text'].split('«')[1].split('»')[0]
        assert decision['core'] == core and decision['attributes'] == row['attributes']
        reasons = []
        if core not in allow[row['language']]: reasons.append('core_outside_explicit_current_first_person_declarative_allowlist')
        if row['parent']['fields'][0] != 0: reasons.append('outer_subject_not_self')
        if row['parent']['fields'][1] != 0: reasons.append('outer_time_not_current')
        assert decision['exclusions'] == reasons and decision['selected'] == (not reasons)
        exclusions.update(reasons)
        if not reasons: chosen.add(key)
    assert report['selected_parents'] == len(chosen) == 108
    assert report['excluded_parents'] == len(by_parent)-len(chosen)
    assert report['exclusion_reasons'] == dict(exclusions)
    artifact_rows = {}
    for name, directory in (('additional-train', 'actual'), ('matched-control', 'actual'), ('runtime-inputs', 'verification')):
        full = read(base/directory/(name+'.jsonl'))
        expected = [r for r in full if (r['language'], r['parent']['case_id']) in chosen]
        actual = read(curated/(name+'.jsonl'))
        assert actual == expected and len(actual) == report['rows_per_arm'] == 864
        artifact_rows[name] = actual
    treatment = artifact_rows['additional-train']; control = artifact_rows['matched-control']; inputs = artifact_rows['runtime-inputs']
    assert [r['pair_id'] for r in treatment] == [r['pair_id'] for r in control] == [r['pair_id'] for r in inputs]
    assert [r['case_id'] for r in treatment] == [r['case_id'] for r in inputs]
    counts = Counter((r['language'], r['attributes']['assertion'], r['review']) for r in treatment)
    assert report['counts'] == [{'language': k[0], 'assertion': k[1], 'review': k[2], 'rows': n} for k, n in sorted(counts.items())]
    for lang in ('ro', 'uk'):
        assert report['unique_cores_by_language'][lang] == len({d['core'] for d in decisions if d['selected'] and d['language'] == lang})
    for name, expected in plan['private'].items(): assert sha(a.private/name) == expected
    proof = {'complete': True, 'verified': True, 'source_sha256': sha(Path(__file__)),
             'quality_plan_sha256': sha(plan_path), 'quality_report_sha256': sha(curated/'report.json'),
             'full_parent_decisions_verified': len(decisions), 'selected_parents': len(chosen), 'rows_per_arm': len(treatment),
             'all_selected_records_are_exact_subsets_of_verified_full_expansion': True,
             'all_contexts_labels_tokens_and_pairing_preserved': True,
             'selection_uses_model_outputs': False, 'training_admitted': False,
             'independent_language_or_label_validation': False, 'qualified': False}
    a.output.write_text(json.dumps(proof, indent=2)+'\n')
    print(json.dumps(proof, indent=2))


if __name__ == '__main__': main()
