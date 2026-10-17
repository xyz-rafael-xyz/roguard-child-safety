# V9 stable qualification work

The release target is a default stable v9 installation. Publication of the
assurance alpha does not satisfy that target. The language-model gates remain
at least 99% concern recall and at most 5% benign review in each language and
formal/conversational group, followed by fresh evaluation and integrated checks.

## Scope curriculum experiment

The starting candidates are Romanian `review-09-ro` and Ukrainian
`review-08-uk`. Their inspected calibration failures concentrate on denied
concerns and interpretation of a personal disclosure after an unrelated topic.
The baseline slice report is in
[`baseline-error-slices.json`](../research/v9-stabilization/baseline-error-slices.json).
Those findings motivate training; they are not independent evidence of success.

The new curriculum contains 3,456 synthetic combinations (1,728 per language).
It crosses personal, reported and fictional scope with affirmation, denial,
uncertainty and retraction. It uses only the existing training content slices,
new authored framing language and training-side prior topics. It includes
hesitancy about expressing feelings, so negation of an ability to disclose must
not be treated as denial of the underlying concern. These combinations are
correlated, not 3,456 independently authored situations. The new framing has
not received independent language review.

Each language starts from its own prior adapter and receives two passes at a
learning rate of 0.00002, with label smoothing 0.02. Epoch selection and cutoffs
use development data. Calibration remains an already inspected regression audit.
The protocol and source hashes were saved before results were available:

- [Protocol](../research/v9-stabilization/scope-01-protocol.json)
- [Source freeze](../research/v9-stabilization/scope-01-source-freeze.json)
- [Training environment](../research/v9-stabilization/training-environment.json)

`training.compare_review_candidates_v9` rejects changed populations, duplicate
predictions and changed gold labels. It reports both fixed and newly introduced
errors on identical inputs. The original reserved probes are not consumed by
this experiment unless development and calibration prerequisites first pass.

No experimental adapter is installed in the package merely because training
completes. Promotion requires measured improvement and every remaining gate.

## Interpretation supervision and mixtures

The second experiment adds six auxiliary classification heads to the shared
encoder. Their mean cross-entropy loss is added to the binary review loss with
weight 1. The auxiliary heads are training-only; the exported binary adapter
retains its existing inference format. Each language again starts from its
original specialist, not the rejected scope-only candidate. See the
[second protocol](../research/v9-stabilization/scope-02-protocol.json).

A separate predeclared comparison mixes predecessor and candidate logits at
candidate weights 0, 0.25, 0.5, 0.75 and 1. It chooses a cutoff that preserves
99% recall in each development group, then minimizes the worst false-review
rate, the sum of those rates and finally the candidate weight. Calibration is
scored only with the selected setting. A unit test changes calibration scores
and verifies that this cannot change selection. Mixture outputs describe a
research configuration, not an installed runtime model.

The Romanian mixture with 25% of the interpretation-supervised candidate
passes the two development gates: formal misses 1/587 and false reviews
66/1,413; conversational misses 2/216 and false reviews 0/242. The predecessor
had 86 formal false reviews at the same recall. However, calibration remains
outside target: formal misses 0/614 and false reviews 241/1,386;
conversational misses 0/216 and false reviews 43/243. This is development
progress, not qualification for stable release.

## Reproduce a candidate

Use the dependency versions and pinned mmBERT revision in the environment
record. Set `ROGUARD_ENCODER_BASE` to that verified local model directory.
The examples below use Apple MPS; use `--device cpu` on a CPU host.
Output directories must not already exist.

```sh
PYTHONPATH=src:. python -m training.adapt_review_aux_v9 \
  --base "$ROGUARD_ENCODER_BASE" --device mps \
  --output /tmp/roguard-scope-02-ro-replay \
  --initial-review research/v8-stabilization/review-09-ro/review.json.gz \
  --train-language ro --scope-curriculum \
  --structured-rows 3000 --conversational-rows 3000 --epochs 2 \
  --learning-rate 0.00002 --label-smoothing 0.02 --auxiliary-weight 1
```

For Ukrainian, use `--train-language uk`, its `review-08-uk/review.json.gz`
checkpoint and a separate output directory. For the scope-only experiment,
use `training.adapt_review_v8` and omit `--auxiliary-weight`.

```sh
python -m training.check_review_candidate_v8 \
  --candidate /tmp/roguard-scope-02-ro-replay --language ro
python -m training.compare_review_candidates_v9 \
  --baseline research/v8-stabilization/review-09-ro \
  --candidate /tmp/roguard-scope-02-ro-replay --language ro \
  --output /tmp/roguard-scope-02-ro-comparison.json
```

A nonzero exit from the promotion check is the expected result for a candidate
that fails a gate. It is not repaired by suppressing the exit code or changing
the gate. GPU training need not be bit-identical across machines; the saved
artifacts and input hashes identify the measured runs.

## Completed results

Four candidates completed eight training passes, totaling 37,824 row
presentations. This count includes repeated inputs. Every standalone candidate
failed its prerequisites and was excluded from the runtime package.
The [machine-readable results](../research/v9-stabilization/results.json)
include selected checkpoints, hashes, group counts and mixture results.

The development-selected mixtures of the second experiment pass all four
development groups. Their calibration results are:

| Group | Missed concerns | False reviews on benign inputs | Gate |
| --- | ---: | ---: | --- |
| Romanian formal | 0/614 | 241/1,386 | Fail: false reviews |
| Romanian conversational | 0/216 | 43/243 | Fail: false reviews |
| Ukrainian formal | 58/614 | 84/1,386 | Fail: recall and false reviews |
| Ukrainian conversational | 8/164 | 7/223 | Fail: recall |

The Ukrainian mixture reduces formal false reviews from the predecessor's 127
to 84 and formal misses from 59 to 58, but conversational misses increase from
7 to 8. Its development counts are 2/587 misses and 27/1,413 false reviews for
formal inputs, and 1/164 misses and 3/223 false reviews for conversational inputs.
Neither language mixture qualifies for promotion. No fresh probe was used and
no new accuracy claim is made for authentic child messages.

Local verification: the full software suite passed 480 tests with 14 skips
for private artifacts. The subsequently added mixture-selection test and the
comparison test both passed in the focused run. These software checks do not
turn the failed model gates into passes.
