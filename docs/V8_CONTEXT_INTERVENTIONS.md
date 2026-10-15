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
