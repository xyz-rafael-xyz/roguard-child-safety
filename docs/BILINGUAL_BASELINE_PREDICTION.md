# Blinded Romanian and Ukrainian baseline prediction

`roguard-predict-bilingual` scores an opaque `reviewer-a.jsonl` packet with the **frozen V25 research comparator**. It supports D1 and S1 in Romanian and Ukrainian. It reads only the packet's abstract cards, language, source kind, and opaque IDs. It writes category decisions and the packet hash; it writes no card text, source IDs, author labels, or model scores. The model operator must not receive the source batch, owner map, or reviewer answers.

The model weights and four cell-specific freeze records are held in the private research archive and are **not** in the public package. Each cell wrapper under `models/` contains a byte-identical copy of the original V25 adapter weight and config, a manifest with that cell's fixed cutoff, and an exact digest of the original model manifest. The existing `roguard-author-kit` and `roguard-human-eval-registered` accept its separately committed, pre-author freeze record without changing the historical V16 freeze or evaluator. The wrapper check rejects a mismatched language, category, cutoff, weight, or original artifact.

After independent taxonomy approval, the study owner must verify a cell's freeze record **before** assigning authors their workbooks. After an approved new batch is sealed and its blind packets exist, the model operator can run:

```sh
roguard-predict-bilingual \
  --root /path/to/private/research-repo \
  --packet /path/to/private/research-repo/review_runs/study/reviewer-a.jsonl \
  --wrapper /path/to/private/research-repo/models/v25-independent-uk-d1 \
  --base-model-path /path/to/pinned/mmbert-base \
  --output /path/to/private/research-repo/eval/prospective/uk-d1-frozen-predictions.json
```

Use the matching wrapper for `ro:D1`, `ro:S1`, `uk:D1`, or `uk:S1`. The output file must not already exist and is created with private permissions. Commit and push that prediction file **before** opening adjudicated labels or publishing the labeled batch, following the [prediction seal](STUDY_PREDICTION_SEAL.md). The public evaluator checks the packet hash, ID order, source alignment, and frozen model identifier. The study owner must still verify the human chronology and that the claimed model actually produced the predictions.

V25 failed its original sealed synthetic target, so it is a **comparator**, not a claimed v1 detector. This path makes a blinded bilingual study technically possible; only independently authored and adjudicated cards can supply new abstract-task evidence. No result here validates real child-message screening or authorizes an external action.

The study owner can independently reproduce the submitted decisions after receiving the frozen packet and prediction file, using the same private model archive and pinned base:

```sh
roguard-replay-bilingual \
  --packet /path/to/private/research-repo/review_runs/study/reviewer-a.jsonl \
  --predictions /path/to/private/research-repo/review_runs/study/predictions.json \
  --wrapper /path/to/private/research-repo/models/v25-independent-uk-d1 \
  --freeze /path/to/private/research-repo/eval/prospective/v25-independent-uk-d1-freeze.json \
  --base-model-path /path/to/pinned/mmBERT-base \
  --root /path/to/private/research-repo
```

Replay verifies the pre-author freeze, model ID, packet hash, and every saved decision against fresh inference. Its result contains only counts and hashes. The general v1 evidence ledger does not run this expensive private-weight replay, so its `model_execution_verified_by_software` field remains false; keep the replay report as separate study evidence. Replay cannot verify who authored the cards or whether human judgments are correct.
