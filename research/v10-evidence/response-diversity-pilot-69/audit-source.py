"""Audit draft structure and enumerated exposed-source overlap; no model inference."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def normalize(text):
    return ' '.join(unicodedata.normalize('NFKC', text).casefold().split())


def words(text):
    return re.findall(r'[^\W\d_]+', normalize(text), re.UNICODE)


def bigrams(text):
    tokens = words(text)
    return set(zip(tokens, tokens[1:]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--private', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'Refuse overwrite'
    study = a.private / 'response-diversity-pilot-69'
    evidence = a.root / 'research/v10-evidence'
    prior_paths = {
        'curriculum25_train': evidence / 'response-curriculum-25/train.jsonl',
        'curriculum25_dev_exposed': evidence / 'response-curriculum-25/dev.jsonl',
        'contrast32_train': evidence / 'contrast-response-32/additional-train.jsonl',
        'context34_exposed': evidence / 'response-context-34/inputs.jsonl',
        'old_diagnostic_exposed': evidence / 'response-curriculum-25/old-diagnostic.jsonl',
        'composition54_train': a.private / 'compositional-response-54/train.jsonl',
        'composition54_dev_exposed': a.private / 'compositional-response-54/dev.jsonl',
        'rejected66_initial': a.private / 'authored-response-pilot-66/train-draft.jsonl',
        'rejected66_revision1': a.private / 'authored-response-pilot-66/train-revision1.jsonl',
    }
    rows = read(study / 'train-draft.jsonl')
    report = json.loads((study / 'draft-report.json').read_text())
    plan = json.loads((study / 'drafting-plan.json').read_text())
    metadata = json.loads((study / 'intended-distinctions.json').read_text())
    assert sha(study / 'train-draft.jsonl') == report['train_draft_sha256']
    assert sha(study / 'drafting-plan.json') == report['plan_sha256']
    assert sha(a.root / 'training/train_neural_response_v10.py') == plan['task_definition_source_sha256']
    assert len(rows) == len({r['case_id'] for r in rows}) == 128
    assert len(metadata) == len({r['family'] for r in rows}) == 32
    by_family = defaultdict(list)
    for r in rows:
        assert r['partition'] == 'train' and r['context'] == ''
        assert r['independently_annotated'] is False
        assert r['label_rationale'] and r['form'] in plan['forms']
        assert r['draft'] == unicodedata.normalize('NFC', r['draft'])
        assert not any(unicodedata.category(c).startswith('C') for c in r['draft'])
        assert len(r['draft']) <= 6000
        by_family[r['family']].append(r)
    for m in metadata:
        family_rows = by_family[m['family']]
        category = m['family'].split(':')[1]
        assert len(family_rows) == 4
        assert {(r['language'], r['gold']) for r in family_rows} == {
            (lang, gold) for lang in ('ro', 'uk') for gold in ('appropriate', category)}
        for r in family_rows:
            expected = m['appropriate_rationale'] if r['gold'] == 'appropriate' else m['problematic_rationale']
            assert r['label_rationale'] == expected and r['form'] == m['form']
    keys = [(r['language'], normalize(r['draft'])) for r in rows]
    assert len(set(keys)) == 128
    assert Counter(f"{r['language']}:{r['gold']}" for r in rows) == report['groups']
    prior, sources = defaultdict(list), {}
    for key, path in prior_paths.items():
        previous = read(path)
        sources[key] = {'sha256': sha(path), 'rows': len(previous)}
        for r in previous:
            prior[(r['language'], normalize(r['draft']))].append({
                'source': key, 'case_id': r['case_id'], 'gold': r['gold']})
    collisions = [{'case_id': r['case_id'], 'prior': prior[(r['language'], normalize(r['draft']))]}
                  for r in rows if (r['language'], normalize(r['draft'])) in prior]
    assert not collisions
    # Diagnostic only: lexical similarity is not a semantic-independence test.
    indexed = defaultdict(list)
    for (lang, draft), origins in prior.items():
        indexed[lang].append((bigrams(draft), origins))
    nearest = []
    for r in rows:
        tokens = bigrams(r['draft'])
        best = max(indexed[r['language']], key=lambda x: len(tokens & x[0]) / len(tokens | x[0]))
        score = len(tokens & best[0]) / len(tokens | best[0])
        nearest.append({'case_id': r['case_id'], 'word_bigram_jaccard': score,
                        'prior_sources': sorted({x['source'] for x in best[1]}),
                        'prior_case_id_example': best[1][0]['case_id']})
    nearest.sort(key=lambda r: (-r['word_bigram_jaccard'], r['case_id']))
    output = {
        'complete': True, 'checks_pass': True,
        'scope': 'Structure, source identities, normalized exact overlaps, and descriptive lexical similarity only.',
        'source_sha256': sha(Path(__file__)),
        'plan_sha256': sha(study / 'drafting-plan.json'),
        'draft_sha256': sha(study / 'train-draft.jsonl'),
        'metadata_sha256': sha(study / 'intended-distinctions.json'),
        'task_definition_sha256': plan['task_definition_source_sha256'],
        'core_drafts': len(rows), 'families': len(metadata),
        'unique_normalized_drafts': len(set(keys)),
        'groups': report['groups'], 'forms_by_family': report['forms_by_family'],
        'character_length': {'min': min(len(r['draft']) for r in rows), 'max': max(len(r['draft']) for r in rows)},
        'prior_sources': sources, 'prior_unique_language_reply_pairs': len(prior),
        'exact_normalized_prior_reply_collisions': len(collisions),
        'lexical_similarity': {'metric': 'Unicode-letter word bigram set Jaccard; descriptive, no pass threshold',
                               'max': nearest[0]['word_bigram_jaccard'], 'top10': nearest[:10]},
        'input_policy': 'Only enumerated TRAIN and already exposed DEV/diagnostic sources; no candidate predictions or reserved TEST files.',
        'semantic_independence_established': False, 'semantic_correctness_verified': False,
        'independent_language_validation': False, 'training_ready': False,
        'used_for_training': False, 'judge_run_launched': False,
        'reserved_test_opened': False, 'stable_release_ready': False,
    }
    with a.output.open('x') as stream:
        stream.write(json.dumps(output, indent=2) + '\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
