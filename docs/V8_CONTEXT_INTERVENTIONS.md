# v8 context-intervention study

The stable-release blockers are measured classifier errors. The target remains
at least 99% review recall and at most 5% benign review for each language and each
formal/conversational evaluation group. A model also needs fresh evaluation and
verification in the integrated runtime before a stable accuracy claim.

The review-05 development audit showed low scores for explicit personal concerns
following ordinary tasks or fictional passages. The new review-06 experiment
starts from review-04, which had better Ukrainian results, and introduces 3,200
training-only examples: 1,600 Romanian and 1,600 Ukrainian. Eight new explicit
topic-reset phrases combine with training-only ordinary and concerning content.
Both positive and negative examples use the reset phrases, reducing the risk of
learning that a topic switch itself implies a concern. These are combinations of
a small authored phrase inventory, not 3,200 independent human messages.

## Method and limits

This is a targeted test of whether earlier topics are distracting the classifier.
It does not assume that prior context is always irrelevant. Implicit references,
fictional scope, and retractions remain in rehearsal training. Selection uses the
same development partitions and recall targets as earlier studies.

The idea is related to work on [concept-level spurious correlations](https://aclanthology.org/2024.acl-long.28/).
Counterfactual augmentation can also introduce new shortcuts or fail to improve
generalization; [Joshi and He](https://aclanthology.org/2022.acl-long.256/) document
these limitations. Improvement must therefore be measured rather than inferred
from the training method.

Existing calibration results have been inspected in earlier rounds and are now
regression audits, not untouched final validation. The frozen 48-case contrast
probe remains separate and is opened only after development prerequisites pass.
No real child messages or new independent language reviews are involved.

A paired development diagnostic removes preceding text only from generated cases
that explicitly switch to a personal topic. It compares the same final message
with and without prior conversation. It is a targeted diagnostic, not a measure
of overall accuracy or false-alert rates.

## Reproduce

```sh
PYTHONPATH=src:. python -m training.adapt_review_v8 --base "$BASE_MODEL" --device mps --output /tmp/review06 --epochs 2 --initial-review research/v8-stabilization/review-04/review.json.gz --contrasts --context-interventions --structured-rows 4000 --conversational-rows 4000 --learning-rate 0.00002 --label-smoothing 0.05
PYTHONPATH=src:. python -m training.check_review_candidate_v8 --candidate /tmp/review06
PYTHONPATH=src:. python -m training.audit_review_ancestors_v8 --candidate /tmp/review06 --output /tmp/review06-ancestors.json
```

The prerequisite checker requires valid counts for every group, checks the
selected artifact hash, and rejects exact normalized input overlap against the
recorded training history. Its passing result is necessary but does not itself
authorize stable publication. The separate ancestor audit checks 17,724 normalized
historical context inputs from the recorded v7 candidate training sets and the
semantic-04/05 training selections; none overlaps review-06 development or
calibration inputs. This does not prove semantic independence or cover unknown
pretraining data.

The exact running trainer is retained as `review-06/training-script.py` with
its hash and environment. Later trainer changes save per-example scores for
future runs; they do not alter this run's archived implementation.

## Language-specific selection and disclosure hesitation

Before the second epoch's results, a separate rule was registered to select the
retained checkpoint per declared language using development recall and worst-group
false-review rate. This prevents one language's gain from forcing a regression in
the other. The rule never uses calibration labels or a dataset-family identifier
at runtime. Its selector is `training/select_review_languages_v8.py`.

The first epoch's Romanian error diagnostic concentrated on inability to express
feelings. A conditional Romanian-only continuation is registered with 24 authored
hesitation phrases and eight vocabulary/task contrasts per language, expanded
through explicit assertion/source scopes. Only Romanian examples enter that
continuation. The training source is development-error-driven; no exact normalized
input or focal-text overlap with the reserved contrast probe was found. These
synthetic phrases have not received independent linguistic review.

## Completed review-06 results

Two passes completed; development selection retained epoch 1. The per-language
rule also selects epoch 1. The second epoch was not retained by the original
trainer because its worst-group development false-review rate was worse; the
current trainer saves every epoch for future comparisons.

| Selected epoch / group | Development misses / positives | Development false reviews / negatives | Calibration misses / positives | Calibration false reviews / negatives |
|---|---:|---:|---:|---:|
| Romanian formal | 1/587 | 237/1413 | 0/614 | 308/1386 |
| Romanian conversational | 2/216 | 14/242 | 0/216 | 79/243 |
| Ukrainian formal | 0/587 | 38/1413 | 0/614 | 277/1386 |
| Ukrainian conversational | 1/164 | 3/223 | 4/164 | 20/223 |

Ukrainian meets the development targets but fails the calibration regression
audit. Romanian also remains above the false-review target. These weights are
not a stable replacement. A Romanian-only continuation now targets the diagnosed
hesitation pattern while preserving the Ukrainian checkpoint separately.

## Romanian specialist and Ukrainian scope repair

The first Romanian-only continuation (`review-07-ro`) completed one pass on
9,684 unique inputs. Its Romanian development false reviews are 207/1413 formal
and 6/242 conversational, with 1/587 and 2/216 misses. Calibration regression
false reviews are 247/1386 and 71/243, with zero misses in either group. This is an
improvement over review-06 but still misses the false-review target.

The Ukrainian regression diagnostic attributes 230 of 297 false reviews to
explicitly negated concerns. Its four misses concern past support requests.
The separately registered `review-08-uk` continuation adds denials of direct and
reported concerns, late fictional clarification, past support statements, and
ordinary knowledge/task requests. It starts from the frozen review-06 Ukrainian
checkpoint, not the Romanian specialist.

A further Romanian continuation (`review-09-ro`) is registered for mild personal
hesitation without overt distress vocabulary, past clarification in a later turn,
and retractions under the existing policy. Its generator is a new revision;
prior runs retain their original generator snapshots and exact training rows.
The new revision has no exact normalized-input or focal-text overlap with the
reserved contrast probe. Existing calibration inputs remain inspected regression
material throughout this work.

The full software suite passed 455 tests: 441 passed and 14 private-artifact tests
were skipped. These software results do not substitute for classifier evaluation.
