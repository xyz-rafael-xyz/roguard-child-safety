# v8 stabilization status

Version **8.0.0rc2 remains a prerelease**. It is published on PyPI and GitHub;
installation is working. The reason it is not the default stable download is
remaining classifier error, not an unsuccessful package upload.

```sh
pip install --upgrade --pre roguard-child-safety
```

The ordinary command without `--pre` selects the latest stable release, currently
7.1.1. Installing the preview does not automatically select the guarded semantic
backend; see [the evidence workspace guide](EVIDENCE.md).

## Distribution fixes

The publishing workflow now calls the same CI workflow at the release commit
and waits for it before building or publishing. This includes the Python
3.9–3.14 unit matrix and the Python 3.9/3.12/3.14 installed-wheel matrix, plus Windows UTF-8 input
checks on Python 3.9 and 3.14. Previously,
CI and publication could proceed independently after a tag push. GitHub documents
this [same-commit reusable-workflow mechanism](https://docs.github.com/en/actions/how-tos/reuse-automations/reuse-workflows).

The README now distinguishes stable and preview installation, links the current
studies, and removes two outdated claims about v4 and adapter availability.
These changes do not imply improved classifier accuracy.

## Error audit and development experiments

All data below are synthetic. Existing v8 tests are regressions after inspection.
They cannot serve as fresh validation for another candidate selected using them.
The new studies use training and development partitions; calibration is audited
separately. No candidate from this round replaces the bundled models.

The controlled regression audit identifies negation as the largest false-alert
source: 563 of 664 Romanian false reviews and 468 of 549 Ukrainian false reviews
in the standalone semantic model involve negated concerns. Ukrainian misses also
involve affirmed, uncertain, and hypothetical statements. The guarded model
preserves the older model's review signals, which also preserves its false alerts.
The [v8 preview report](V8_PREVIEW.md) gives the denominators and comparisons.

### Direct review heads

Rather than combine six uncertain attributes, two experiments predict review
from the same frozen encoder's context and focal-message features:

- **Linear:** four regularization strengths, trained on the existing 10,000-row
  selection. Development thresholds require at least 99% recall in each language
  and each of the formal/conversational groups before minimizing false reviews.
- **Nonlinear:** 64- and 128-unit hidden-layer candidates, using the same data and
  threshold-selection procedure. Early stopping uses a training-only subset.

Neither met the registered maximum 5% benign-review target:

| Selected candidate / calibration group | Required reviews missed | Benign messages flagged |
|---|---:|---:|
| Linear / Romanian formal | 0/614 | 1215/1386 |
| Linear / Romanian conversational | 1/216 | 140/246 |
| Linear / Ukrainian formal | 0/614 | 1187/1386 |
| Linear / Ukrainian conversational | 0/164 | 101/225 |
| Nonlinear / Romanian formal | 0/614 | 1325/1386 |
| Nonlinear / Romanian conversational | 0/216 | 176/246 |
| Nonlinear / Ukrainian formal | 0/614 | 1292/1386 |
| Nonlinear / Ukrainian conversational | 0/164 | 137/225 |

A later [partition audit and repair](V8_REPAIRS.md#evaluation-partition-isolation)
identified shared ordinary-prefix inputs in these early splits. The historical
counts remain diagnostic; corrected continuation uses disjoint normalized inputs.

These positive labels include uncertain and retracted concerns, so their counts
must not be compared directly with the clear-concern-only denominator in older
reports. High recall alone is insufficient when benign-review workload is this
large. The heads are retained as research artifacts, not shipped classifiers.

### Encoder adaptation and scope contrasts

The next pass updates the encoder's LoRA weights and direct review head together.
An initial completed epoch supplies initialization only; its overlapping split
scores are diagnostic. The corrected continuation (`review-04`) trains for two
epochs on 20,199 unique inputs, including 4,200 generated scope contrasts. It uses
a lower learning rate and label smoothing, selecting the second epoch on
4,845 development inputs before auditing 4,846 calibration inputs.

| Selected continuation / calibration group | Required reviews missed | Benign messages flagged |
|---|---:|---:|
| Romanian formal | 0/614 | 405/1386 |
| Romanian conversational | 0/216 | 124/243 |
| Ukrainian formal | 0/614 | 342/1386 |
| Ukrainian conversational | 1/164 | 32/223 |

This improves several diagnostic counts but still exceeds the false-review
target. The bundled classifiers therefore retain their released weights.

A further continuation (`review-05`) adds 2,520 generated short-message contrasts
and balances training loss across language, short/long input and review target.
Its 14,688 unique training inputs are accompanied by a transitive 22,687-input
training history. Both development and calibration have zero exact normalized
input overlap with that history. The phrases were authored from development error
classes; these data do not establish independent linguistic review or semantic
family separation. Protocols, selected rows, scripts, hashes and metrics are
retained under `research/v8-stabilization/`.

The short-message continuation's calibration audit is:

| Group | Required reviews missed | Benign messages flagged |
|---|---:|---:|
| Romanian formal | 0/614 | 256/1386 |
| Romanian conversational | 0/216 | 51/243 |
| Ukrainian formal | 0/614 | 1030/1386 |
| Ukrainian conversational | 2/164 | 171/223 |

The Romanian counts improve, while Ukrainian behavior regresses. This checkpoint
is not promoted. A predeclared length-conditioned calibration comparison uses
only development positives to set cutoffs for focal messages of at most 12 words
versus longer messages, separately by language. Calibration labels are used only
to audit the resulting decisions, and the fresh contrast probe stays unopened.

The length-conditioned calibration result also fails the target:

| Calibration group | Required reviews missed | Benign messages flagged |
|---|---:|---:|
| Romanian formal | 0/614 | 247/1386 |
| Romanian conversational | 0/216 | 43/243 |
| Ukrainian formal | 10/614 | 311/1386 |
| Ukrainian conversational | 4/164 | 162/223 |

Lower false-review counts here come with additional Ukrainian misses. Neither
the adapted weights nor the length-conditioned cutoffs enter the release.
Exported inference reproduces the training script's development/calibration
counts for the original language cutoffs. All four encoder passes and the
calibration comparison are complete; the fresh contrast probe remains unused.

### Local instruction-model comparison

A cached Qwen3-4B-Instruct-2507 4-bit model is evaluated on 200 development cases:
50 per language and formal/conversational group. A second prompt explicitly
clarifies context references and the existing retraction policy after inspecting
the first development run. This is prompt development, not an independent test.
The model has no tool access, and its output is not used for permissions or actions.

See `research/v8-stabilization/instruction-01/` and `instruction-02/` for exact
inputs, prompts, model revision, raw outputs, durations, and group metrics.
Generic instruction-following capability is not evidence of task accuracy.

| Prompt / language (100 development cases each) | Required reviews missed | Benign messages flagged |
|---|---:|---:|
| Initial / Romanian | 10/44 | 6/56 |
| Initial / Ukrainian | 6/39 | 6/61 |
| Revised / Romanian | 17/44 | 1/56 |
| Revised / Ukrainian | 15/39 | 3/61 |

All outputs parsed as binary decisions. The revised prompt reduced false reviews
but increased misses; both prompts failed the target. No result here warrants
shipping this larger model as an accuracy upgrade.


## Context-intervention continuation

A targeted continuation addresses sensitivity to earlier ordinary and fictional
topics. See [the context-intervention study](V8_CONTEXT_INTERVENTIONS.md) for its
protocol, partition audit, and reproducible prerequisite checker. Its weights
remain experimental until the measured gates pass.

## Promotion target and remaining work

The recorded target is at least 99% concern-review recall and at most 5% benign
review in each language on declared evaluations, no regression on explicit
serious-concern cases, and separate unseen evaluation before a stable accuracy
claim. This is a development target, not a guarantee of real-world safety.

A new 48-case bilingual contrast probe was authored and frozen before model
predictions. It has no exact message overlap with the selected training turns;
that check does not establish semantic independence. It remains unevaluated
because no candidate has yet passed development gates. The translations and
labels are project-authored, not independently reviewed.

Further improvement needs better transferable handling of scope, negation,
references, and uncertainty. Merely adding more combinations of the same synthetic
phrases or changing a release label would not demonstrate that improvement.

## Reproduction

From the repository, with training dependencies and a pinned local mmBERT base:

```sh
PYTHONPATH=src:. python -m training.fit_review_v8 --base "$BASE_MODEL" --device cpu --output /tmp/roguard-review-new
```

The nonlinear comparison consumes the recorded `review-01` rows and generated
feature caches; run `python -m training.fit_review_nonlinear_v8 --output /tmp/roguard-nonlinear-new`
with `PYTHONPATH=src:.`. The script refuses to overwrite its output.
The instruction comparison accepts `--base`, `--output`, and `--policy-version 1`
or `2`. Feature caches are excluded from Git. None of these scripts reads the new
contrast probe for fitting or selection.

The encoder continuations and calibration audit are reproducible separately:

```sh
# Requires the retained review-03 checkpoint and local training dependencies.
PYTHONPATH=src:. python -m training.adapt_review_v8 --base "$BASE_MODEL" --device cpu --output /tmp/review04 --epochs 2 --initial-review research/v8-stabilization/review-03/review.json.gz --contrasts --learning-rate 0.00003 --label-smoothing 0.05
PYTHONPATH=src:. python -m training.adapt_review_v8 --base "$BASE_MODEL" --device cpu --output /tmp/review05 --epochs 1 --initial-review /tmp/review04/review.json.gz --contrasts --short-contrasts --balance-short --structured-rows 4000 --conversational-rows 4000 --learning-rate 0.00003 --label-smoothing 0.05
PYTHONPATH=src:. python -m training.evaluate_review_v8 --base "$BASE_MODEL" --device cpu --artifact /tmp/review05/review.json.gz --data /tmp/review05/dev.jsonl.gz --output /tmp/review05-scores/dev
PYTHONPATH=src:. python -m training.evaluate_review_v8 --base "$BASE_MODEL" --device cpu --artifact /tmp/review05/review.json.gz --data /tmp/review05/cal.jsonl.gz --output /tmp/review05-scores/cal
PYTHONPATH=src:. python -m training.calibrate_review_length_v8 --candidate /tmp/review05 --scores /tmp/review05-scores --output /tmp/review05-length
```

Recorded runs used Apple MPS, not CPU; numerical differences are possible.
Per-run script snapshots and environment records preserve the implementation
used at training time. The current script also records transitive training history.
