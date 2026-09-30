# Cross-pair surface overlap in consumed abstract batches

The independent-batch intake now reports normalized character five-gram Dice similarity **between different candidate pairs** as well as against older batches. It excludes the adjacent negative/positive mate: similarity within that mate is deliberate. The report contains card numbers and scores, never card text. A score of at least 0.8 prompts human inspection; it cannot decide whether two cards are semantically redundant, whether a pair changes one fact, or whether authors are independent.

## Retrospective diagnostic

The check was run against three already consumed, generator-authored Romanian batches. Their source SHA-256 digests and results are:

| Batch | Source SHA-256 | Cross-pair card comparisons | Cards with a nearest cross-pair score ≥ 0.8 | Highest score |
|---|---|---:|---:|---:|
| `batch-0035`, 48 cards | `3ce7c1350795368114d0ab50d0aaa43e55ec9acdb2b61ce88d5ed1ae2ea475fb` | 1,104 | 48/48 | 0.9756 |
| `batch-0037`, 96 cards | `aa712b987ee9f9aa4db1280350ccd78c99e0bc6e1599aae8ce0eb22c3c245dc9` | 4,512 | 96/96 | 0.9814 |
| `batch-0038`, 96 cards | `b18f73118b403e2dc9b29f65619953e63f06d52b56c89998ec7d07b8afa4f6af` | 4,512 | 96/96 | 0.9815 |

These counts mean the abstract generator reused much of its wording. They do **not** mean every card is an exact duplicate or that a model leaked labels. They strengthen the reason to use independent language-native authors and a separate content reviewer for a new test. The metric alone cannot establish diversity: two semantically equivalent cards can have low lexical similarity, and common required scaffolding can make valid cards score high.

Reproduce a row without printing card text:

```sh
PYTHONPATH=src:. python - <<'PY'
import json
from pathlib import Path
from roguard.intake_overlap import audit_prior_overlap

root = Path.cwd()
batch = 'batch-0037'
rows = [json.loads(line) for line in (root / 'data/synthetic' / f'{batch}.jsonl').read_text(encoding='utf-8').splitlines()]
report = audit_prior_overlap(root, rows, batch)['within_candidate_cross_pair']
print(len(rows), report['cross_pair_comparisons'], report['near_overlap_cards'], report['maximum_similarity'])
PY
```

For a future independent candidate, the author kit saves the same warning in the private assembly manifest. The content reviewer must inspect every warning and the full pair set, including low-score pairs, before approving exact bytes. This diagnostic adds no new held-out model result and changes no past score.
