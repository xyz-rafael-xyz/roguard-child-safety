# RoGuard 8 experimental evidence preview

This preview implements a larger evidence workflow while retaining the stable
v7 defaults. It is not a claim of near-perfect child-language understanding,
independently validated Romanian/Ukrainian fluency, or regional priority.

## What is implemented

- Separate subject, time, assertion, source, relation and concern heads.
- Fast character model, adapted contextual semantic model, optional cascade,
  and guarded mode that preserves released context review signals.
- Original text offsets, candidate claim references and proposed statement links.
- Append-only evidence history and explicit host-reviewed supersession.
- Exact enumeration of possible response actions over unresolved attributes,
  including reviewer question ranking and decision witnesses.
- Response proposals that preserve host-declared task/support/review coverage
  and check current source permissions without executing anything.
- A bilingual browser workspace, CLI, Python API and authenticated gateway route.

This combines established techniques. The useful research question is whether
structured interpretation and explicit action uncertainty improve review
workload without losing concerns or useful assistance. Feature count alone does
not answer it.

## Data and separation

The initial controlled corpus contains 20,000 training, 4,000 development,
4,000 calibration and 8,000 test conversations. Training-only paraphrases and
scope-separated contexts add 39,984 distinct inputs. **75,984 distinct fictional
conversation inputs** are recorded across these partitions; there is no exact
cross-split input overlap in the recorded audit.

These counts represent combinations of a small number of content families and
phrases. The initial split has 24/8/8/8 concern-family IDs across train/dev/cal/test
and 48/16/16/24 ordinary-family IDs, counted across both languages. Romanian and
Ukrainian versions are aligned, so these are not independent semantic families.
Source and subject are correlated in the first controlled curriculum.

All examples are authored within the project. No real child messages were used.
The corpus is largely controlled, explicit phrasing; it should not be described
as tens of thousands of independently authored realistic child messages.

A follow-up adapts the encoder using 5,000 structured and 3,000 existing
conversational training examples. Existing v7 frames provide coarse factor labels;
correction frames are excluded. There is no exact conversation overlap between
this mixed training sample and the controlled test or either prose diagnostic.
The inherited v7 adaptation and coarse label projection remain important limits.

## Training iterations

Five initial configurations were fitted: two character models and three frozen
multilingual encoder models, each with three regularization candidates. Selection
used development performance, and calibration used a separate partition.
Training-only paraphrases improved the worst development task/language macro-F1
from 0.0446 to 0.6385 for the character model and from 0.4377 to 0.7578 for the
selected contextual semantic model. These are development comparisons.

The first held-out comparison exposed poor prose transfer despite better
controlled-scenario review workload. It was not promoted as a generally superior
replacement. A subsequent two-epoch LoRA experiment mixes conversational and
structured supervision and selects its epoch on development results. A further
pass continues from its selected weights with 8,000 structured and 2,000
conversational training rows. The minimum development task/language macro-F1
improves from 0.6133 to 0.6216; the continued candidate remains experimental.

## Results observed so far

Routing counts below score the focus turn. “Concern reviewed” is recall for the
specified synthetic policy, not a verified detection of harm. Prior review
signals remain in runtime session history even when the focus is benign.

### Initial and adapted model comparison

| Controlled regression, 4,000 cases per language | RO benign reviews / 2,727 | UK benign reviews / 2,727 | RO concerns reviewed / 386 | UK concerns reviewed / 386 |
|---|---:|---:|---:|---:|
| Released v7 joint workflow | 1,775 | 2,017 | 386 | 385 |
| Initial semantic-03 evidence | 759 | 471 | 386 | 350 |
| Mixed semantic-04 evidence | 752 | 564 | 386 | 292 |
| Character-first cascade with semantic-04 | 898 | 1,416 | 386 | 386 |

The mixed model improved complete six-attribute argmax agreement from
1,460 to 2,207 Romanian cases and 1,611 to 2,000 Ukrainian cases. Complete
assertion-contrast pairs improved from 92/1,334 to 507/1,334. These remain far
from perfect interpretation. Ukrainian subject/source errors caused a serious
routing regression: most missed personal concerns were interpreted as fiction.
A union of semantic-03 and semantic-04 prediction sets still missed 26 of 386
Ukrainian concerns; it was not adopted as a solution.

On the earlier 48-case prose regression, semantic-04 reviewed all 22 concerns
and flagged 2 of 26 benign cases. On the subsequent frozen 56-case probe it
reviewed all 26 concerns and flagged none of 30 benign cases. The released v7
workflow reviewed 25/26 concerns and flagged 2/30 benign cases on that probe.
This small result did **not** negate the larger controlled regression.

The character-first cascade also performed worse than the direct adapted model
on these prose probes. Its raw argmax metrics remain those of its primary model;
its selective label reconciliation is a separate operation.

### Continued adaptation and guarded integration

The bundled `semantic-05` continues training on 10,000 existing training-only
rows. Its complete six-attribute argmax agreement reaches 2,428/4,000 Romanian
and 2,545/4,000 Ukrainian cases. A later engineering audit corrected a mismatch
between unresolved prediction sets and the top-scoring label: action hypotheses
now retain that disagreement. The original outputs and amended-policy replay
are both retained. This is a posthoc correction, not a fresh test result.

| Current-policy controlled regression | RO benign reviews / 2,727 | UK benign reviews / 2,727 | RO concerns reviewed / 386 | UK concerns reviewed / 386 |
|---|---:|---:|---:|---:|
| Continued semantic-05 alone | 664 | 549 | 386 | 320 |
| Guarded semantic-05 + released v7 workflow | 1,815 | 2,060 | 386 | 386 |

Guarded mode preserves every released review signal in this replay. That avoids
silencing existing signals, but raises review workload above the released model:
**3,875 of 5,454 benign controlled cases are still flagged**. This is unsuitable
for a claim of low false-alert rates or autonomous deployment. Factor labels can
help inspect a decision; they are not established grounds for overriding it.

A new 64-case project-authored probe was frozen before inspecting semantic-05.
Its initial factor-only result missed 7 of 32 concerns and flagged 7 of 32 benign
cases. Correcting uncertainty handling reduced the misses to 5. After guarded
integration, an actual runtime regression on the same cases reviewed **32/32
concerns**, with **8/32 benign reviews**, versus **28/32 concerns and 8/32 benign
reviews** for v7. The guarded comparison is explicitly a regression, because
these cases were already inspected. It is not independent confirmation.

On the previous 56-case probe, semantic-05 with corrected uncertainty handling
reviewed 26/26 concerns and flagged 1/30 benign cases. These differently composed
probes illustrate how strongly apparent performance depends on the test surface.
No result here establishes authentic-child accuracy or near-perfect language
understanding. Stable v7 defaults remain available; version 8 ships as an
explicit prerelease workbench.

## Evaluation provenance

The first 8,000-case test and 48-case prose diagnostic were held out for the
initial selection. Once inspected, they became regression sets for the mixed
adaptation. A further 56-case prose diagnostic was frozen before inspecting the
mixed candidate's results. Neither prose set has independent annotation.

A policy audit also changed hypothetical concerns about real people from
`CONTINUE` to review. The initial-policy outputs are preserved. Saved predictions
were replayed under the amended policy, and reference policy actions were
recomputed consistently for both new and v7 models. That replay is not a fresh
model evaluation.

The cascade retains primary argmax values for provenance; its final selective
labels can differ. Raw argmax metrics for the cascade therefore describe the
primary model, not a newly trained combined classifier.

## Engineering verification

The declared response planner passes 4,800 exhaustive permission/coverage cases.
The action envelope passes all 46,305 nonempty combinations of attribute
hypothesis sets, checking actions, witnesses and reviewer-question ranking.
That is deterministic software evidence, not language-model accuracy or proof
that generated response prose is helpful. Coverage and disruption costs are host
declarations.

The software suite runs 429 tests: 415 pass and 14 historical private-adapter
checks skip because those artifacts are absent from the public checkout.
The core evidence API also runs on local Python 3.9.

Browser checks cover Romanian and Ukrainian, original-text selection, stale
report clearing, mobile layout and browser errors. Unit/integration tests cover
cascade disagreement, unresolved action spaces, history preservation, explicit
host review, permissions/revocation, authentication and unsupported input.

## Reproduce

From a repository checkout, install the training/evaluation dependencies and use
the pinned model returned by `roguard-model locate` as `BASE_MODEL`:

```sh
PYTHONPATH=src:. python -m training.fit_evidence_v8 --backend character --augment research/v8-evidence/data/train-paraphrases.jsonl.gz --output /tmp/roguard-character-new
PYTHONPATH=src:. python -m training.adapt_evidence_v8 --base "$BASE_MODEL" --device cpu --epochs 2 --output /tmp/roguard-adapt-new
```

The frozen data loaders verify file hashes. Training/evaluation output directories
must be new. All reported candidates, data freezes, selected training rows and
results are retained under `research/v8-evidence`. Large reproducible embedding
caches are excluded from Git.

[Usage and API](EVIDENCE.md) · [Stable v7.1 evidence](V7_1_RELEASE.md)
