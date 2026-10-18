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

## Further retries: policy probabilities, replay and lexical scope

`policy-03-ro` trains six interpretation heads and derives a review score by
summing the relevant policy worlds under an independent-marginals assumption.
This is an approximation, not a calibrated probability of a child's situation.
Its arithmetic matches exhaustive enumeration of all 432 policy worlds in the
tests. Three training epochs still produce too many false reviews: the selected
model has 599/1,386 formal and 89/243 conversational calibration false reviews.
Correct policy arithmetic does not compensate for incorrect interpretations.

`replay-04` returns to the original language specialists, replaying 8,000
historical training inputs per language alongside 1,296 new operator contrasts.
The contrasts distinguish denial, double negation and shifts between fictional
and personal scope. The **v2 protocol** is the protocol actually used; its
predecessor was revised before training to add fictional-current examples.
Romanian completed two epochs. Its calibration false reviews remain 274/1,386
formal and 69/243 conversational, with no missed positive cases in either group.
The development-selected mixture rejects this candidate in favor of the
original Romanian model. Ukrainian also completed two epochs. Its selected first
epoch meets both formal calibration targets: misses 2/614 (previously 59/614)
and false reviews 59/1,386 (previously 127/1,386). On identical formal inputs it
fixes 131 errors and introduces 6. Conversational calibration still fails recall:
misses 8/164, false reviews 6/223. The development-selected score mixture chooses
the predecessor; the standalone improvement does not qualify the overall model.

`sparse-05` tests a different representation: separately fitted character and
word features for the current message, plus word features for preceding context.
It trains binary and six interpretation heads on historical training inputs and
the operator contrasts. Four alternatives per language (two regularization
values and direct versus policy-derived scores) are selected using development
data only. Neither selected model meets the gates. A 25% lexical-score mixture
reduces Ukrainian formal development false reviews to 21/1,413, but calibration
still has 61/614 misses and 96/1,386 false reviews. Romanian's corresponding
mixture also fails. Reloading the JSON exports reproduces every development
score exactly in both languages; the problem is model quality, not serialization.

The next prepared retry, `natural-06`, adds 36 new fictional content families per
language: 24 emotional-support situations and 12 benign everyday requests. It
crosses these with denial, affirmation, reported speech and fictional speakers;
Romanian also includes variants without diacritics. Its **v2 protocol**, frozen
before training, adds reported and fictional denial contrasts to avoid teaching
denial only in the first person. There are 2,520 Romanian and 1,260 Ukrainian
combinations, not that many independent situations. None exactly overlaps the
existing development or calibration inputs. These machine-authored examples
have not received independent language review.

All these retries use already inspected calibration data as regression evidence.
No reserved probe has been evaluated, no failed candidate is installed, and no
stable-release accuracy claim follows from the number of training examples.

Software verification for this round: 471 tests passed and 14 skipped in the
sandbox. The 12 Studio tests initially could not bind a localhost socket; all
12 passed when rerun with that access (483 passing tests in total).

The `sparse-07` control repeats lexical fitting with the natural-06-v2 content.
The Romanian mixture selects the original encoder (zero lexical weight). The
Ukrainian mixture selects 25% lexical weight and has the same development
counts as sparse-05; calibration formal false reviews decrease from 96 to 94,
while formal misses remain 61/614. Neither passes. Both JSON exports reproduce
their saved development scores exactly. CI for commit `db6fa46` passed all
11 jobs, including the Python 3.9–3.14 matrix and Windows installation checks.

A separate `sensitivity-threshold-08` diagnostic chooses the lowest threshold
allowed by the 5% development false-review budget in both groups. Calibration
does not choose this cutoff, although its previously observed misses motivated
the experiment. Ukrainian then misses 4/164 conversational cases instead of 8,
but formal false reviews rise to 117/1,386. Romanian also fails. These results
rule out this simple cutoff adjustment as a sufficient repair. The diagnostic
is reproducible with `training.select_sensitivity_threshold_v9`.

## Twelve-hour improvement window (18–19 October 2026)

The authorized window runs from 12:16:49 UTC on 18 October until 00:16:49 UTC
on 19 October. GPU experiments run serially on the local machine and stop at
that deadline. Experiment volume does not change the release requirements.

`natural-06` completed both languages. Romanian calibration has no misses but
248/1,386 formal and 46/243 conversational false reviews. Ukrainian has 51/614
formal and 11/164 conversational misses, with 3/1,386 and 2/223 false reviews.
Neither qualifies. `scope-gate-10` learns whether quoted material applies to the
speaker separately from its content. It preserves the words outside closed
quotes, including negation. Its Ukrainian result reduces conversational misses
from 8 to 7/164 versus replay-04; it still fails. Romanian also fails. Reloaded
scope exports reproduce development and calibration scores within 1e-12.

`gemma-screen-11` evaluates unadapted Gemma 3 4B on the fixed 400-case development
screen: 345 correct (86.25%), no invalid outputs. It uses the complete model
output path and verifies cached versus uncached scoring. This result does not
meet the release requirements. Gemma 3 12B is scheduled on the same screen.

`attention-12` learns which context and current-message tokens to pool. Zero
queries reproduce the original mean representation; masking and gradient checks
pass. Ukrainian training completed: development qualifies, but calibration
misses 19/164 conversational concerns. The Romanian experiment is running.

Two further studies are prepared, not yet results. `interpreter-14` trains a
12B model on 2,000 synthetic examples to predict six interpretation fields.
Only those six class tokens contribute to the loss. Its selected checkpoint
minimizes loss on the same 80 development examples at each checkpoint, with
weights saved at the exact evaluated state. `capacity-15` expands the encoder's
rank-four attention adapters to rank sixteen and adds MLP adapters. A matrix
check verifies that this expansion preserves the initial function. Both studies
have frozen protocols and sources before training.

The full software suite for this checkpoint passed 487 tests, with 14 skipped
and 123 subtests passed. CI for the preceding commit `55b8cf2` is green. The
reserved probes remain untouched. These are synthetic research results;
no new candidate has replaced the installed runtime model.

`policy-decode-16` adds adaptive integration of the structured model's
conditional probabilities. It explores the largest unresolved branch first,
collapses a subtree only when every ontology completion has the same review
decision, and retains unqueried or invalid-output mass in its bounds. It never
renormalizes away omitted branches. The bounds apply to the constrained model
distribution, not real-world confidence. Seven tests pass, including comparisons
with exhaustive enumeration of all 432 worlds for eight correlated distributions
at five query budgets. Unadapted and adapted 12B comparisons on the same
400-case development screen are queued; no accuracy result is claimed yet.

`nli-17` reuses the pinned multilingual entailment base previously studied on
abstract cards, now as an unadapted focal-message interpretation control.
`nli-transfer-18` will fit rank-eight query/value adapters and the classifier and
pooler on balanced historical training inputs plus the new contrasts. Two
learning rates are frozen before training. The 887,811 trainable parameters
pass a nonzero-gradient check and export/reload produces identical logits on a
training input. Neither setup check establishes classification accuracy.
The model card reports Ukrainian NLI training; it does not report RoGuard
accuracy or validate Romanian transfer. See the
[upstream model card](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7).

CI for `ddb7ddf` passed, including the supported Python matrix and Windows checks.

The candidate checker now recomputes error counts from every saved prediction,
checks exact case coverage and language/gold binding, and verifies each decision
against the selected artifact's threshold. Missing or duplicate cases, nonfinite
scores and inconsistent reported counts fail the check. Fifteen focused tests
pass, including adversarial report mutations. Rechecking natural-06-ro and
attention-12-uk reproduces their existing failures without new inconsistencies.

The `nli-17` control completed at 231/400 correct (57.75%): it misses one
Romanian formal concern and produces 168 false reviews across the four groups.
This rules out using these unadapted interpretation prompts as a repair.
The supervised transfer trials remain separate, unmeasured experiments.
The NLI queue's pre-training dependency handling was corrected to continue when
an earlier adapted comparison is explicitly skipped; training settings did not
change, and the runner source hash was refreshed before it started training.
