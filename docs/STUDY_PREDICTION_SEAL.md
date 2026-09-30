# Seal blind predictions before publishing labeled abstract cards

The approved independent batch contains **author-intended labels**. The model operator must not receive the source JSONL, preview, provenance, owner map, or reviewer answers. The owner gives the operator only `reviewer-a.jsonl`, the pinned base and adapter, and the frozen prediction command. The model and evaluation plan were already committed before authors received assignments under the [pre-author freeze](STUDY_FREEZE.md). This sequence is necessary for the future abstract-card result to be a credible held-out comparison.

After separate content review, `roguard-author-kit seal` writes the four approved batch files to the owner's local `data/synthetic/`. Run `roguard-independent-check` and `roguard-blind-packets` there, but **do not commit or push the labeled batch files yet**. Keep the owner map private. Have the separate operator produce an opaque-ID prediction file from the reviewer A packet. The owner checks that the prediction file names the exact packet hash and registered model ID, then commits and pushes **only that prediction file** under `eval/prospective/`. Record the remote commit and push evidence before releasing adjudications or staging the labeled batch files. The operator must not have access to the owner's worktree or repository copy that contains the uncommitted batch.

Only after the blind prediction is pushed should the owner commit and push the four approved batch files. The two reviewers may annotate in parallel while blind to intended labels and model output; the separate adjudicator should not receive model predictions. Preserve original annotations and packet bindings, then adjudicate and run `roguard-human-eval-registered`. The [Git-order verifier](../eval/verify_independent_study_git_order.py) checks the final repository history:

```sh
python eval/verify_independent_study_git_order.py \
  --batch batch-XXXX \
  --predictions eval/prospective/batch-XXXX-v16.json
```

It requires one immutable addition for the freeze record and blind prediction, one common addition commit for the four labeled batch files, and strict commit order: freeze, prediction, labeled batch. It also checks the registered model identifier and exact freeze digest. `git_commit_order_verified: true` does **not** prove remote push order, operator blinding, independent authorship, or adjudication validity; the report leaves those flags false. A study owner must retain the remote push record and private handoff evidence separately. If the labeled batch reaches the remote before predictions are sealed, do not call the later score a fresh model test. Keep it as a failed chronology attempt and start a new independently authored batch.
