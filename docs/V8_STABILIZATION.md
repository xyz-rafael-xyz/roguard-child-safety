# v8 stabilization status

Version **8.0.0rc1 remains a prerelease**. It is published on PyPI and GitHub;
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
3.9–3.14 unit matrix and the Python 3.9/3.12/3.14 installed-wheel matrix. Previously,
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

These positive labels include uncertain and retracted concerns, so their counts
must not be compared directly with the clear-concern-only denominator in older
reports. High recall alone is insufficient when benign-review workload is this
large. The heads are retained as research artifacts, not shipped classifiers.

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
