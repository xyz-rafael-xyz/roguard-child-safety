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

`attention-12-ro` completed both epochs and selected the first. Calibration has
no missed concerns but 262/1,386 formal and 50/243 conversational false reviews;
it does not qualify. Its second epoch worsened development false reviews.

### Execution corrections before structured training

The Gemma 12B screen was interrupted for performance diagnosis, first with the
generic scorer and then with bounded final-token projection. Both partial sets
are explicitly marked interrupted; neither is a completed accuracy benchmark.
Float32 scoring often took approximately 30–35 seconds per example on this
16 GB M1 Pro. Lower-precision probes changed scores; float16 changed important
classifications. Those results do not justify a numerical shortcut.

The actual `interpreter-14` training configuration is therefore **v3: Gemma 3
4B in float32**, selected before any interpreter training. The same 2,000
training inputs, 80 validation inputs, class-token loss and 400-case comparison
remain. The adaptive policy comparison uses that same 4B base. Earlier 12B
configurations are retained as superseded plans. The trainer now handles the
actual `language_model` wrapper, and the bounded adapter archive recognizes the
pinned wrapper names and dimensions for both bases. Final-token projection is
checked against the complete model output path before scoring continues.

`kernel-19` is a separate CPU-only control: four nonlinear RBF decision heads
on existing semantic feature caches, with input-identity checks and the exact
current development/calibration populations. Even a passing cached result would
require independent encoder replay. No reserved probe is used.

The kernel control completed all four settings and failed. Selected calibration
false reviews are 946/1,386 Romanian formal, 123/243 Romanian conversational,
685/1,386 Ukrainian formal and 79/223 Ukrainian conversational, with no missed
concerns. This does not support replacing the trained language specialists with
a nonlinear head on these older cached features. Exported support-vector scores
were checked against the fitted model on all development and calibration rows.

The actual 4B structured baseline completed all 400 cases but failed the strict
schema check on every case. These are counted as abstentions/errors. Its
unadapted **binary-prompt** reference remains 345/400; merely learning to emit
schema digits will not by itself establish better language understanding.
The initial structured LoRA state passes a separate bounded archive round-trip:
all 112 tensors (3,506,176 parameters) are exactly preserved. That verifies
serialization, not a trained model's accuracy.

### Coverage, diagnostics and CI

`interpreter-20` is a frozen 4,096-row training comparison using the same pinned
4B base, prompt, adapter architecture and learning rate. Its sampling explicitly
includes negation, uncertainty, retraction, past and hypothetical statements in
both languages. It has 157 Romanian and 139 Ukrainian retraction examples,
compared with 41 and 11 in interpreter-14. The existing validation set is reused
for checkpoint selection. The study was subsequently deferred before fitting
in favor of the shorter policy-weighted continuation described below; it has
no accuracy result.
An audit of the 38,771-row historical-plus-contrast training pool found no
identical normalized inputs with conflicting six-field labels. This checks
internal consistency, not independent correctness of synthetic annotations.

The structured diagnostic recomputes the decision from the decoded attributes
and preserves strict schema abstentions. For incorrect interpretations it lists
all inclusion-minimal sets of gold field corrections that would repair the
policy action. These are offline diagnostics, never substituted predictions or
"corrected accuracy." Paired counts compare with the 345/400 binary reference.
Three tests cover interacting scope corrections and inconsistent decisions.

CI now uses pytest instead of unittest discovery so function-based decoder and
diagnostic tests actually run alongside the existing unittest classes. The
corrected matrix passed in run `39134478139`. Locally, 486 tests passed in the
main run; 12 loopback-server tests initially hit the sandbox restriction and
then passed with socket access. Fourteen optional historical-artifact tests
remain skipped. The pinned Gemma base files and training-library versions are
recorded in `gemma4-base-manifest.json`; a reusable archive verifier compares
every restored adapter tensor and its configuration against the source.

`interpreter-full-21` will select the completed structured candidate using the
shared validation decision ranking described below and evaluate every existing
development/calibration row. The 400-case screen is not treated as the final
accuracy estimate. Passing this regression still requires the reserved fresh
probes and integrated-runtime checks.

`native-full-22` prepares a different capacity test: update all parameters of
the pinned [Romanian BERT](https://huggingface.co/dumitrescustefan/bert-base-romanian-cased-v1)
and [Ukrainian RoBERTa](https://huggingface.co/youscan/ukr-roberta-base) separately,
using the language-specific historical training pool plus new contrasts.
The Ukrainian base weights were downloaded and checked against their pinned
SHA-256. Loading uses `weights_only=True`; selected outputs use safetensors.
Three epochs, balanced sampling, optimizer settings and development selection
are frozen before training. Romanian cedilla letters are canonicalized to the
comma forms required by that base's model card. The completed Romanian result is reported below; upstream model cards do not
establish child-safety performance. Full checkpoints remain local until a candidate warrants
distribution. A manifest binds each selected checkpoint, and all development
scores must replay after reloading it before calibration is evaluated.

### Completed structured interpretation pilot

Interpreter-14 completed all 2,000 training steps. The selected state at step
1,999 has validation loss 0.2084; the final state is slightly worse at 0.2088.
On the 400-case development screen it scores **371/400 (92.75%)**, versus
345/400 (86.25%) for the unadapted binary reference. There are no invalid
outputs. Paired comparison finds 38 corrected reference errors and 12 new
errors. This screen includes the 80 examples used for checkpoint selection;
it is development evidence, not an independent final test.

| Group | Missed concerns | False reviews |
|---|---:|---:|
| Romanian formal | 1/50 | 8/50 |
| Romanian conversational | 6/50 | 0/50 |
| Ukrainian formal | 2/50 | 7/50 |
| Ukrainian conversational | 1/50 | 4/50 |

The candidate does not meet the release requirements. Of the 29 policy errors,
16 can be repaired by correcting concern recognition, 12 by correcting the
subject/source interpretation, and one by correcting assertion. These are
diagnostic gold-label interventions, not repaired model predictions. The
trained archive restores all 112 tensors exactly. The capacity comparisons
started after this run completed.

Before native-22 training, its documented v2 amendment adds `focus-23`:
4,214 training-only variants from 48 new fictional content families. They cover
ordinary task difficulty, real/fiction switches, unrelated topic changes, and
source/time clarifications. The generator reads no evaluation files; the
resulting inputs have no normalized overlap with the existing development or
calibration partitions. Evaluation messages and labels remain unchanged.
These are machine-authored, correlated variants with no new independent
language review. Native-22's optimizer, selection rule and evaluation remain
as declared; its original protocol is retained alongside the amendment.
Before training, gender-specific reporting frames were corrected and authored
feminine forms were added. These share the masculine forms' content-family IDs.

The first capacity trial, `capacity-15-low-ro`, completed and selected epoch 1.
Epoch 2 increased Romanian development false reviews and was rejected.
Calibration misses no Romanian concerns, but falsely reviews 244/1,386 formal
and 54/243 conversational benign cases. It fails the unchanged 5% false-review
requirements. The remaining capacity results are summarized below.

### Further capacity results and the remaining execution queue

The low-rate Ukrainian capacity trial selected epoch 1. Calibration misses
1/614 formal and 16/164 conversational concerns, with 41/1,386 and 0/223
false reviews respectively. The high-rate Romanian trial also selected epoch 1:
it misses no Romanian concerns but produces 241/1,386 formal and 50/243
conversational false reviews. Both fail. A separate development-only threshold
margin experiment reduces the Ukrainian conversational misses to 6/164, but
formal false reviews increase to 70/1,386 (5.05%); it also fails. Calibration
feedback remains diagnostic evidence, not a fresh final test.

The high-rate Ukrainian run encountered a Metal command-buffer stall. A second
attempt reached epoch 2 before another stall. Both partial attempts are retained
and explicitly marked incomplete. Attempt 3 uses the same data, seed, batch
sizes and optimization settings, with tighter MPS memory watermarks and periodic
release of unused cache. Attempt 3 completed both epochs and calibration. Its
selected epoch-1 archive is byte-identical to the completed checkpoint from
attempt 2, and the logged training losses match. Calibration has 0/614 formal
misses but 76/1,386 false reviews (5.48%); conversational results have 9/164
misses (94.51% recall) and 3/223 false reviews. It fails both applicable accuracy
requirements. The development-selected mixture chooses this candidate alone
and therefore does not repair those failures. All four capacity trials are
now complete; none qualifies.

Native-22 now includes 1,120 time-scope training variants, 560 per language,
within the same 48 content families as focus-23. Its training populations are
23,328 Romanian and 20,777 Ukrainian inputs. Before fitting, selection was
amended to inspect six development checkpoints at half-epoch intervals. All
inputs pass a no-truncation preflight. Base manifests pin tokenizer files as
well as weights; unregistered tokenizer or adapter overrides are rejected.
Its v6 pre-fit amendment uses right padding to multiples of 32 and periodic
MPS cache release to bound execution memory. Shape changes can alter rounding
and seeded dropout draws; no bit-identical comparison with the unrun earlier
recipe is claimed. Installed-runtime replay uses the same padding contract.

The queued structured continuation, `interpreter-policy-27`, starts from the
exact selected interpreter-14 adapter. Its 1,024 training rows combine replay,
focus-boundary and time-scope examples. Field weights depend on how many
alternative labels would change the policy decision with other gold fields
held fixed. This is training-only supervision. The initial unweighted six-field
loss on the same 80 validation rows must reproduce interpreter-14 before training.
A pre-fit amendment saves every scheduled validation checkpoint and decodes
those same 80 rows after training. Selection first prefers checkpoints passing
the unchanged recall and false-review requirements, then minimizes the worst
class error rate across the four groups, mean class error rate, and unweighted
loss, with earlier iteration breaking ties. Abstentions count as errors. The
initial adapter remains eligible. The original loss-selected weights and metrics
are retained separately. This combines curriculum, loss weighting and a lower
learning rate; it does not isolate their causal effects.

Full-21 will choose between completed, archive-verified studies 14 and 27 using
that same decision ranking. This amendment precedes policy-27 fitting and
full-21 inference; the completed native Romanian calibration is not a selection
input. All 80 validation rows belong to development, with no calibration ID
overlap. The binary-only Gemma comparison and supervised
NLI comparisons are deferred before fitting under the
[qualification-priority schedule](../research/v9-stabilization/qualification-priority-32.json).
Measured training times require reserving time for full regression and installed
runtime replay within the original work window. Their recipes and data remain
available; neither has a fitted-model accuracy result. Reserved neural probes remain
unscored. During code inspection, some reserved probe generator wording became
visible after the training datasets were frozen; these probes must not be
described as analyst-blind. The access is recorded in
[the probe audit](../research/v9-stabilization/reserved-probe-access-audit.json).

### Runtime readiness checks

The installed character evidence path scores 303/400 on the same development
screen. Interpreter-14 fixes 85 of its errors and introduces 17, for a net gain
of 68 correct decisions. This compares that particular evidence path, not the
whole stable v7.1.1 product. Complete character-path development and calibration
predictions are retained under `runtime-reference-28` for paired comparisons.

An explicit local native-checkpoint runtime now binds metadata, selected
weights and tokenizer files by hash. It excludes future turns, rejects invalid
prefixes before copying, serializes inference, and distinguishes overlength
abstention from a negative decision. These runtime checks do not approve the
model's accuracy or authorize guardian notifications.

`installed-native-29` is queued after the neural experiments. It builds and
installs a local wheel, verifies that imports come from that installation, and
replays every native development/calibration case on CPU. Parity requires zero
changed decisions, zero abstentions and maximum score error at most 0.0001.
Accuracy requirements are checked separately, including counting abstentions
as errors. This phase has no result yet and publishes no wheel or release.

The [structured interpreter API](INTERPRETER_RUNTIME.md) is also implemented
for explicit local adapters, with the frozen training prompt, conditional
decoding and full-vocabulary validity check. Its separate
`installed-interpreter-30` replay will compare all 400 screen interpretations,
validity flags and decisions with the selected research adapter. Unit tests
pass, but the neural runtime replay remains pending.

### Reserved-input audit and threshold comparison

The [reserved-input audit](../research/v9-stabilization/reserved-overlap-33.json)
finds no exact normalized input overlap, focal overlap, or reserved focal text
inside any training turn for the actual Romanian native training history and
the structured continuation's transitive history. It reads reserved inputs
programmatically to compute hashes and counts; it displays no message text and
runs no predictions. The Ukrainian native history will be checked when its
actual training input file exists. This establishes neither semantic independence
nor independent annotation. The original and lexical-reset family probes share
96 complete inputs, so they must not be pooled as independent observations.

Before native calibration results, `native-threshold-34` declares a separate
development-only cutoff comparison on each fixed selected checkpoint. Its
alternative allows at most 1% development false reviews in each stratum, with
an absolute 0.0001 numerical guard. Eligible original/margin settings must still
meet the unchanged 99% recall and 5% false-review requirements. Selection first
maximizes the worst stratum's recall, then minimizes false reviews, preferring
the original on a tie. The selection record is written before calibration
predictions are opened. This headroom is a heuristic, not a coverage guarantee.
A passing cached-score result would still need installed inference replay and
fresh actual-runtime evaluation. No threshold result is available yet.

### Reserved installed-runtime check prepared (study35)

The native holdout verifier is prepared but has not scored reserved inputs.
It requires passing installed development and calibration reports for both
languages before opening held-out messages. Each reserved set and language is
checked separately against the unchanged99% recall and5% false-review limits;
abstentions count as errors. Combined totals deduplicate exact inputs, since the
original family and lexical probes overlap. These small authored sets are not
independent observations or an estimate of real-world safety.

Six tests cover strict gate boundaries, abstentions, duplicate disagreement,
future-turn exclusion from input identity, missing strata and misleading pass
flags. The protocol and source hashes are in `reserved-runtime-35`. This
verifier currently accepts the original native22 artifacts; an alternative
threshold needs a separate installed replay with its own artifact identity.

### Native/structured fusion predeclared (study36)

A separate lightweight experiment will combine the fixed native22 score with
full21's predicted context fields and conditional probabilities. It uses no
message text or gold interpretation as a feature. The existing development
partition becomes fitting data for this additional component: five grouped
folds keep matching focal inputs and supplied pairs together. Four fixed
regularization strengths are compared using out-of-fold counts. The selected
model is then fitted on the full development partition, with its threshold
and identity recorded before calibration is opened.

This is not independent end-to-end cross-validation: the parent models already
used development for checkpoint selection. Calibration and reserved runtime
checks remain necessary. Invalid structured output stays an abstention and an
error. The unchanged99% recall and5% false-review limits apply separately to
both strata and both languages. No fusion results exist yet.

Four focused tests passed for feature provenance, invalid probability rejection,
connected fold grouping and abstention handling. A separate NumPy numerical
smoke check converged at all four predeclared regularization strengths. The
CPU-only runner waits for complete neural predictions and loads no neural model.

### Reserved input identity audit (studies40–41)

The three reserved files contain 384 rows, but only **233 distinct raw selected
conversation prefixes** and **176 distinct normalized inputs** in combination.
Their binary labels do not conflict on repeated inputs. Within-file duplicates
and overlaps between files mean these are not 384 independent observations.
The earlier overlap audit reported cross-file intersections, not the total
number of unique inputs; subtracting that intersection alone would overcount.

The actual Romanian tokenizer maps its 192 reserved rows to **56 distinct token
inputs** across the three files. None overlaps the 23,328 token inputs in its
training history. The shared interpreter has 176 distinct reserved token inputs
and zero overlaps with its 2,980-row transitive training history. Neither history
has conflicting binary targets for identical token inputs. The Ukrainian native
audit remains pending until its training history is materialized.

These checks load verified tokenizers only, without neural inference or message
text in their reports. The interpreter identity excludes the declared language
because that field is not in its prompt; v2 audit files correct the initial
language-scoped identity. Exact token non-overlap does not establish semantic
family independence, pretraining independence or real-world accuracy. Sources
and results are in `reserved-input-identities-40.json` and the
`reserved-token-overlap-41-*-v2.json` records.

### Completed native language runs

Native-22 Romanian completed three epochs and selected epoch 2.5 using the
frozen development rule. Reloaded development scores match exactly. Calibration
misses 0/614 formal and 0/216 conversational concerns, but produces 241/1,386
formal false reviews (17.39%) and 7/243 conversational false reviews (2.88%).
It fails the formal false-review requirement. The development-only threshold-34
comparison finds no eligible alternative and retains the original artifact.
These results do not qualify the model for stable release.

Ukrainian completed three epochs and selected epoch 1.5. Development passes both
strata: 5/587 formal and 1/164 conversational concerns missed, with 6/1,413 and
2/223 false reviews respectively. Reloaded scores match exactly. Calibration
passes formal counts (4/614 missed, 4/1,386 false reviews) but misses 26/164
conversational concerns, with 2/223 false reviews. The threshold-34 comparison
retains the original artifact and does not repair that failure. The difference
between development and calibration is a remaining generalization gap; neither
native candidate qualifies for stable release. Structured alternatives continue.

The policy, full-regression and installed-interpreter coordinators now record
terminal setup failures atomically, preserving earlier job history. Dependent
jobs can finish or skip explicitly instead of waiting indefinitely. Private
exception messages are kept out of public status records.

The reserved fusion check (`reserved-fusion-42`) is prepared but has not run.
It requires both fusion models to pass their original development and calibration
limits, and all installed components to reproduce their research outputs. A
hashed selection plan is then checked before each serial native/interpreter
process. The final step combines these newly generated installed-runtime reports.
Each reserved set is evaluated separately by language, with abstentions counted
as errors and duplicate inputs reported explicitly. Passing these small synthetic
probes would still not establish real-world population accuracy.

### Adaptive policy decoding

The completed adaptive-decoding comparison (`policy-decode-16`) uses the same
400-example development screen as interpreter-14. The unadapted base abstains
on every case because its full-vocabulary predictions fail the label contract.
The adapted model decides all cases and gets 376/400 correct, compared with
371/400 for greedy six-field decoding. It averages 3.5475 conditional queries.
This is a reused-screen comparison, not a fresh accuracy estimate or controlled
latency benchmark. It still misses concerns in every stratum and exceeds the
5% false-review limit in three strata, so it does not qualify for release.

### Native interpretation-head fallback

`native-fields-43` is a predeclared alternative to the fusion comparison.
It reuses each frozen native encoder and trains six small categorical heads on
the existing training labels. Features combine the final CLS representation and
the focal message's mean token representation. The model learns subject, time,
assertion, source, retraction relation and concern content separately. Review
mass is integrated over all 432 combinations under an explicitly factorized,
uncalibrated distribution. It is not a learned joint prediction set.

A fixed linear/256-unit comparison uses development-only checkpoint selection,
then reloads exported weights before the separate calibration audit. Native
encoder selection already used development; this is not untouched end-to-end
validation. The original 99% recall and 5% false-review limits remain unchanged.
It was initially queued after fusion. Once both native classifiers failed
calibration, a scheduling-only amendment moved head fitting and its installed
replay ahead of the language-model continuation. This leaves more time to act
on the result. Training and selection settings are unchanged, and substantial
model loads remain sequential. Tokenizer preflight covers all 44,105 training inputs without truncation.
No fitted-head accuracy or stable-release eligibility is claimed yet.

Before fitting native-fields-43, a second amendment binds exported heads to the
complete policy-table digest and uses an equivalent closed-form binary policy
calculation. It matches exhaustive enumeration on all 432 deterministic worlds,
uniform marginals and 50 seeded random marginal distributions (maximum allowed
difference 2e-15). The explicit `NativeFieldsModel` runtime validates head tensor
shapes and finite values before loading the pinned encoder. Overlength inputs
abstain. Its marginal argmax interpretation is diagnostic; the binary proposal
integrates the field distributions. Six runtime unit tests pass, but actual
fitted-model installed inference is still required.

The third pre-fit native-fields amendment compares independent decoding with
soft pair coupling. Training labels consistently align subject with source, and
assertion with retraction relation. Separate marginal decoding can add mass to
cross-combinations not observed in those labels. The new option estimates two
positive likelihood-ratio tables from training counts only, with one pseudocount
per cell and the same balanced source/class weighting. It never hard-excludes
unseen pairs. At each unchanged head checkpoint, development selects between
the two decoding options under the original recall/false-review limits.

The resulting distribution contains two learned pair factors, plus the concern
and time heads; independence is still assumed between those factors. Raw head
probabilities and head-argmax interpretations are distinguished from the
normalized pair probabilities in runtime reports. The coupled policy formula
matches exhaustive enumeration on 50 seeded random factor problems within
2e-15. A 16-candidate fit/export/reload/calibration smoke test passed on artificial
feature fixtures. This is implementation evidence, not message accuracy.

The final-layer extraction amendment avoids retaining every intermediate encoder
state. Tiny CPU BERT and RoBERTa fixtures reproduce both final states and parent
logits exactly. Full-checkpoint score replay remains a prerequisite during
extraction; measured runtime memory and latency gains have not been established.

`installed-native-fields-45` is queued after all fallback extraction and fitting
processes finish. It builds and installs a fresh wheel outside the checkout,
then replays every completed language candidate on the full development and
calibration inputs. The check requires zero changed decisions, no unexpected
abstentions, and score and field-probability differences no greater than 1e-4.
Accuracy is checked separately; reproducing an inaccurate model does not qualify it.

`reserved-native-fields-46` is prepared for candidates that pass those installed
checks and the unchanged accuracy limits in both languages. A frozen plan binds
the heads, parent encoders, selection records, installed reports and runtime
dependencies before opening the three reserved probe files. Ten gate tests cover
boundary counts, missed concerns, false alerts, missing strata, changed evidence,
nonfinite parity errors and parent-model mismatches. These are software checks;
reserved neural predictions and final release qualification are still pending.

### Original-encoder representation comparison

The first semantic-head comparison completed without meeting the accuracy gates.
Romanian calibration has the same 241/1,386 formal false reviews as its binary
parent. Ukrainian calibration misses 102/614 formal and 36/164 conversational
concerns. In Romanian development, the assertion head recognizes only 13 of 581
explicitly negated examples; concern recognition is substantially better. The
installed Romanian replay reproduces all decisions with score and field errors
below 1.2e-5, so these errors are not explained by a package/runtime discrepancy.

`pretrained-fields-47` tests whether the binary fine-tuning discarded useful
interpretation features. It uses the untouched original language encoders with
the same datasets, feature pooling, head architectures, optimization and
development selection as study43. Its storage wrapper has an unused binary
classifier; that classifier is not an evaluated review model. The learned six
semantic heads produce the actual proposal. Calibration remains separate from
head selection, while earlier calibration findings are acknowledged as available
experiment-design information. Study48 repeats the full installed-package check.
This comparison is queued before the language-model continuation; model loads
remain sequential. No improvement is claimed before its results exist.

The original-encoder comparison subsequently completed. Its Romanian development
assertion head recognizes 554/581 negations, versus 13/581 for the binary-tuned
encoder, but concern recognition is weaker. Neither language passes calibration.
Both study48 installed replays reproduce the selected research decisions.

Study45 exposed a separate numerical issue: the Ukrainian cutoff was almost
exactly on a development score, producing one CPU/MPS decision difference.
Centering that cutoff between neighboring development scores preserves every
development decision and removes the difference when replaying the already
installed neural outputs (`field-threshold-portability-50`). This is a deterministic
cutoff repair, not fresh neural inference or an accuracy improvement. One additional
calibration false review results. Study47 adopted cutoff centering before head
fitting, with the original installed parity requirement retained.

### Field-wise representation mixtures

`field-fusion-51` selects a convex mixture separately for each semantic field,
using development categorical loss balanced over language-register/class groups.
The fixed native-weight grid is 0, 0.25, 0.5, 0.75 and 1. Policy integration then
compares independent and training-fitted pair factors under the unchanged gates.
Calibration is opened after the mixture selection is saved.

Romanian development improves to 5/587 formal concerns missed and 9/1,413 false
reviews, with 0/216 conversational misses and 3/242 false reviews. Calibration
still fails: 65/614 formal misses and 110/1,386 formal false reviews. Ukrainian
also fails. Both study52 installed mixers reproduce all research decisions using
the actual bound neural field outputs from studies45/48. This checks the combined
runtime, not fresh neural inference. Neither mixture qualifies for release.

The source checkout includes `FieldReviewFusion` and `FieldReviewPipeline`. A
pipeline runs both explicit local encoders and returns one input-bound proposal.
Its optional bundle loader checks a trusted manifest digest, component identities
and paths confined to the bundle directory. It never downloads implicitly.
No qualifying default bundle is published by these experiments.

### Joint encoder and semantic-head training

Study54 starts from the original language encoders and study47 heads, then
jointly updates the last two encoder layers and all six heads. It adds 10,368
authored synthetic compositions covering assertion, subject and time to the
existing training pool. Exact tokenizer-input checks found zero overlap with
the three reserved probes in either language. This does not establish semantic
independence or validate the language as authentic child speech.

The recipe, data hashes and selection rules were frozen before training. It
retains the 99% concern-recall and 5% false-review gates for each language and
register, followed by full installed CPU replay. Study56 may run reserved neural
inference only if both installed candidates pass. Previous calibration failures
informed this design; calibration remains regression evidence rather than a
fresh blind test. Both languages completed training and installed CPU replay.
Runtime parity passes, but calibration accuracy does not:

| Language / register | Missed concerns | False reviews |
| --- | ---: | ---: |
| Romanian formal | 27 / 614 | 279 / 1,386 |
| Romanian conversational | 0 / 216 | 39 / 243 |
| Ukrainian formal | 13 / 614 | 284 / 1,386 |
| Ukrainian conversational | 2 / 164 | 40 / 223 |

Study56 therefore produced no reserved predictions.

A [single-model local bundle interface](NATIVE_FIELDS_RUNTIME.md) reduces loading
to one manifest and its trusted digest. This is an explicit candidate interface,
not an automatic model selection or a stable-release claim.

Study57 is a predeclared comparison of study43's binary-adapted representation
with study54's joint semantic representation. It reuses the study51 field-wise
mixture method, selects only on development data, and checks each parent's
training hash separately because their training pools differ. Study58 replays
the selected mixer over the actual installed component outputs. These steps
require no additional encoder training or reserved predictions. Runtime parity
passes, but the combined models also fail calibration. Romanian formal counts
are 7/614 misses and 220/1,386 false reviews; conversational counts are 0/216 and
10/243. Ukrainian counts are 116/614 and 165/1,386 formal, and 36/164 and 32/223
conversational. Study59 therefore produced no reserved predictions. These
comparisons do not alter study54's selected standalone model.

Study61 tests a direct binary review objective alongside the mean of the six
semantic losses. It starts from study54's development-selected representation,
reuses study22's binary classifier/pooler, and trains the final four encoder
layers plus both sets of heads for one epoch. The training pool is unchanged
from study54. Selection uses development counts; study62 checks the existing
`NativeReviewModel` against the selected research checkpoint through an isolated
wheel. Study63 can open reserved inputs only if both languages pass. This is a
multitask training comparison, not a claim of a new general learning algorithm.

Study62 confirms exact decision parity for both direct-review models, but neither
passes accuracy. Romanian calibration has 14/614 formal misses and 241/1,386
false reviews, plus 0/216 conversational misses and 25/243 false reviews.
Ukrainian selection retains the initial checkpoint: its later checkpoints have
an all-review cutoff under the original fixed numerical-margin rule. Calibration
has 0/614 formal misses and 1,365/1,386 false reviews, plus 3/164 conversational
misses and 165/223 false reviews. Study63 remains unopened. Before study66 starts,
a recorded amendment replaces the absolute threshold margin with the highest
score-gap midpoint preserving the same recall budget. Actual installed decision
parity remains mandatory; a midpoint is not an assumed device-error bound.

Policy continuation27 completed its training, but its first checkpoint-selection
attempt stopped before neural inference because the decoder's output parent
directory was absent. The selector now creates that directory. Recovery60 pins
the completed training and checkpoint bytes and resumes selection and evaluation
without repeating training. The original attempt remains in its evidence record.

Study64 tests a faster decoder before full interpreter evaluation. It batches
only identical token layouts and repeats the common prefix cache without sharing
mutable state between messages. Adoption requires all 400 existing screen cases
to preserve decisions, field labels and validity exactly, with conditional
probabilities within 0.001, and lower observed decoding time. Otherwise evaluation
uses the original decoder. This is a numerical/runtime comparison, not new
accuracy evidence or a controlled latency benchmark.

Study65 evaluates the auxiliary semantic heads of study61's fixed selected
encoder, independently of its direct binary head. Its Romanian development
selection passes both groups: 5/587 formal misses with 30/1,413 false reviews,
and no conversational errors. Calibration still fails: 12/614 formal misses
with 99/1,386 false reviews, and 0/216 conversational misses with 36/243 false
reviews. This is cached research inference; it has not qualified a deployed
semantic model.

Study65's Ukrainian auxiliary path reproduces study54's selected representation:
calibration has 13/614 formal misses and 284/1,386 false reviews, and 2/164
conversational misses with 40/223 false reviews. It also fails qualification.

Study66 targets two observed regression gaps: denial wording and a later turn
clarifying that a statement belongs to fiction. Each language receives 5,184
new authored compositions and 6,144 deterministically sampled training replay
rows. Affirmation, denial, uncertainty and retraction all occur with both concern
labels and all subject/time groups. Training starts from study61's selected
encoder and auxiliary heads, adapting the final four layers and semantic heads.
The calibration diagnosis informed this repair; calibration is therefore reused
regression evidence. No exact evaluation sentence was copied into the new data.
Both tokenizer overlap audits find zero exact training/reserved input overlap,
which does not establish semantic independence or real-world accuracy. Study67
requires actual installed CPU replay; study68 remains conditional on both
languages passing the unchanged accuracy and parity gates.

Study69 predeclares a fixed categorical comparison on study66's selected
checkpoint: take each head's highest-scoring label and apply the existing
policy. It introduces no threshold fit or new checkpoint choice. Research
decisions are compared with the same rule applied to actual installed study67
neural outputs. Both decision and interpretation parity must be exact, and the
same per-language/per-group accuracy gates apply. This separates semantic
interpretation errors from errors introduced by integrating marginal scores.
Study70 performs actual installed categorical inference on the reserved probes
only if both study69 language reports pass all of those checks. It runs before
interpreter recovery to preserve serial model execution. A failed qualification
skips reserved inference and leaves the probes unopened.

Study71 isolates a Romanian checkpoint-selection tradeoff observed during
study66 development. All three checkpoints pass the review gates, but the final
checkpoint lowers mean semantic loss from 0.26681 to 0.21425 while increasing
formal false reviews from 29 to 45 (both below the allowed limit). Study66 keeps
the initial checkpoint under its original ranking. Study71 repeats the same
Romanian training data, initialization and seed, choosing lower semantic loss
among checkpoints that already pass both review gates. Failed checkpoints cannot
outrank a passing one. It also retains raw intermediate checkpoints locally for
reproducibility. Study72 repeats actual installed Romanian inference; Ukrainian
qualification reuses study66/67 without claiming another training run. Study73
opens reserved inputs only if both language candidates qualify. Calibration is
still reused regression evidence.

### Completed scope and selection experiments (2026-10-18)

Study66 retained its initial checkpoint in both languages. Actual installed
CPU replay in study67 preserved every decision, but calibration accuracy failed.
The categorical alternative in study69 also reproduced installed decisions and
interpretations exactly without meeting the accuracy gates. Study70 consequently
skipped reserved inference.

| Candidate | Language | Formal misses / concerns | Formal false reviews / negatives | Conversational misses / concerns | Conversational false reviews / negatives |
| --- | --- | --- | --- | --- | --- |
| Scope repair66 | RO | 13 / 614 | 93 / 1,386 | 0 / 216 | 36 / 243 |
| Scope repair66 | UK | 13 / 614 | 284 / 1,386 | 2 / 164 | 40 / 223 |
| Categorical69 | RO | 2 / 614 | 169 / 1,386 | 0 / 216 | 45 / 243 |
| Categorical69 | UK | 24 / 614 | 252 / 1,386 | 3 / 164 | 39 / 223 |
| Semantic selection71 | RO | 41 / 614 | 1 / 1,386 | 0 / 216 | 32 / 243 |
| Native expert fusion75 | RO | 21 / 614 | 7 / 1,386 | 0 / 216 | 6 / 243 |
| Native expert fusion75 | UK | 1 / 614 | 120 / 1,386 | 5 / 164 | 38 / 223 |

Study71 selected the final Romanian checkpoint: lower semantic loss did not
guarantee better review decisions on calibration. Study75 fits a small classifier
on cached direct-classifier and semantic-head outputs, using connected development
folds and four predeclared regularization values. It uses no message text, case IDs
or evaluation-group labels as features. Its Romanian conversational result passes
the count gates, but formal recall fails; Ukrainian results also fail. Base models
were already selected using development data, so the folds do not constitute an
independent evaluation of the whole model. No installed composite runtime has
qualified this native-expert combination. All rows above are reused synthetic
regression evidence, not estimates of real-world child-safety performance.

Installed study72 reproduced study71 Romanian decisions exactly, with zero
abstentions; accuracy remained below the required gates. Study73 skipped reserved
inference. The complete repository test suite after the pipeline additions passed
743 tests and 129 subtests, with 23 optional tests skipped. Software test success
does not close the model-accuracy gates.

Study76 adds a predeclared three-expert comparison using the already computed
native classifier, semantic encoder and interpreter outputs. Separate native
marginals and interpreter conditional probabilities remain separate features;
same-class products measure agreement. They are not joint probabilities. The
small classifier uses the same connected folds and four regularization choices,
with no text or source-group labels as features. It requires no extra neural
training or inference. Calibration remains reused regression evidence; installed
composite parity and reserved checks would still be required before deployment.

Recovery60 completed checkpoint selection, the 400-case neural screen, archive
export and exact tensor round-trip verification. Policy continuation27 selected
its initial checkpoint, which is the same tensor state as interpreter14; all five
updated checkpoints ranked worse on the fixed 80-case validation population.
The replay has 371/400 correct review decisions and no invalid outputs, with
remaining errors in all four language/style groups. This does not meet the
release gates. Full evaluation retains interpreter14 under the fixed tie-break.

Study64 passed the actual 400-case decoder comparison: zero changed decisions,
interpretations or validity flags, zero abstentions, and maximum conditional
probability difference 0.000027925. Observed decoding time was 183.31 seconds
versus the cached single-message reference's 217.05 seconds (15.54% less time).
This is not a controlled hardware benchmark or a production-runtime speed claim.
The full research evaluation consequently uses batches of up to four equal-layout
inputs. The installed interpreter remains separately checked.

The first full batched run encountered severe resource pressure: only 32 cases
completed over several minutes. Its partial outputs and selection are preserved
in `interpreter-full-21/attempt-1`. A memory amendment changes the redundant
uncached probe to run the full-context transformer and project only the final
hidden state, avoiding a vocabulary-sized tensor at every context position.
The actual decoder is unchanged. Adoption still requires a new complete 400-case
comparison against the unchanged single-message reference. The original screen
and source freezes are retained under `interpreter-batch-64/attempt-1` and the
corresponding pre-amendment freeze files.

The amended 400-case screen passed with the same prediction-file SHA-256 as
the first batch screen, again zero decision/interpretation/validity changes and
zero abstentions. Observed time was 175.86 seconds. The bounded uncached probe
agreed within 0.000007332. Full evaluation restarted with this checked source.

### Full interpreter and model-combination results

The complete interpreter evaluation finished 4,845 development and 4,846
calibration cases. Every case produced valid output, but accuracy gates failed.
Calibration formal concern recall was 82.57% in Romanian and 84.53% in Ukrainian.
Installed native classifier replay29 preserved every decision in both languages;
this establishes runtime parity, not model accuracy.

The following are calibration counts for fixed development-selected combinations:

| Study | Language | Formal misses / 614 | Formal false reviews / 1,386 | Conversational misses | Conversational false reviews |
| --- | --- | --- | --- | --- | --- |
| review-fusion-36 | RO | 8 | 12 | 0 / 216 | 6 / 243 |
| review-fusion-36 | UK | 13 | 3 | 15 / 164 | 3 / 223 |
| three-expert-fusion-76 | RO | 26 | 0 | 0 / 216 | 3 / 243 |
| three-expert-fusion-76 | UK | 12 | 15 | 10 / 164 | 13 / 223 |

Study77 retained each combination's fitted weights and reduced its threshold
to cover every development concern in both OOF and refitted predictions. This
recovered misses but increased false alerts; none of its four parent/language
candidates passed all gates. A separate calibration oracle diagnostic showed that
no single threshold on either Ukrainian linear combination could satisfy both
requirements. These oracle thresholds are not deployed or treated as candidates.

Study78 tests shallow boosted trees on the same fixed three-expert features.
Four depth/leaf configurations use five connected development folds; thresholds
and the final fit are fixed before its calibration audit. It adds no neural
training or inference. Plain JSON tree export must reproduce scikit-learn to
1e-12, with a separate test on unseen vectors. Prior calibration diagnosis informed
this experiment, so its calibration result remains reused regression evidence.

The user resumed near the original 12-hour boundary. A recorded 30-minute
completion window allows the active evaluation and queued installed checks to
finish; the original schedule is preserved. Accuracy gates were not changed.

### Conservative boosted candidate and installed runtime

Study78 combined three fixed experts using shallow boosted trees. Study79 kept
those fitted trees and selected a threshold covering every development concern
in both out-of-fold and refitted scores. Calibration diagnosis motivated the
experiment, but calibration scores did not select the threshold.

| Study | Language | Formal misses / 614 | Formal false reviews / 1,386 | Conversational misses | Conversational false reviews |
| --- | --- | --- | --- | --- | --- |
| boosted-fusion-78 | RO | 10 | 0 | 0 / 216 | 3 / 243 |
| boosted-fusion-78 | UK | 6 | 1 | 26 / 164 | 1 / 223 |
| conservative-boosted-79 | RO | 0 | 52 | 0 / 216 | 7 / 243 |
| conservative-boosted-79 | UK | 0 | 10 | 19 / 164 | 2 / 223 |

The Romanian study79 candidate passes the unchanged development and calibration
count gates. Its calibration false-review rates are 3.75% formal and 2.88%
conversational, with no missed concerns. These are reused synthetic calibration
results, not independent or real-world accuracy estimates.

Study80 built and installed a wheel containing the explicit three-model report
combiner. Across 200 Romanian cases, combining cached actual installed neural
outputs produced zero decision changes and zero score difference from research.
The screen missed no concerns and falsely reviewed one conversational benign
case. Component identity/input binding and tree validation have dedicated tests.
This is a report-combination API, not a bundled automatic three-model pipeline.

The full installed interpreter screen30 and assurance replay37 completed.
Installed two-expert fusion38 also completed; reserved fusion42 was skipped
because bilingual qualification failed. Reserved neural inputs remain unopened.
Ukrainian conversational recall still blocks stable v9 promotion.

### Ukrainian follow-up controls

Study81 excludes the binary score and all score interactions, retaining 85
semantic features. It recovers all conversational calibration concerns, but
falsely reviews 38/223 benign conversational cases and 296/1,386 formal cases;
it also misses 22/614 formal concerns. A feature-invariance test verifies that
changing the excluded binary features cannot change its input vector.

Study82 retains all 121 features and samples a square-root-sized feature subset
at each tree split. Four depth/leaf settings use the same connected folds, 200
trees, and full development concern coverage. Calibration has 0/614 formal misses
and 11/1,386 formal false reviews; conversational results are 18/164 misses and
2/223 false reviews. It therefore fails the recall gate. Neither follow-up is
promoted. Both reuse calibration for diagnosis and introduce no new neural
training or reserved predictions.

### Continued Ukrainian training and integrated inference

Study83 warm-started native22 and adapted its top four encoder layers plus
classifier. Its 3,936 training examples combine 3,072 history rows with 864
authored distress, benign, denial and fictional-scope contrasts. Protected
normalized inputs are excluded; transitive training history is retained.
Optional focus handling and unsupported MPS attention dropout were corrected
before any optimizer step. Training uses eager attention.

The selected checkpoint passes development gates but misses 16/164 conversational
calibration concerns, compared with native22's 26/164. It is not promoted.
Study85 adds unmet-needs and emotional-disconnection contrasts and a separate
216-case development stress set. Original accuracy gates remain unchanged.

Study84 checks actual installed inference through `ThreeExpertPipeline` against
the existing 200-case Romanian screen. This also tests a combined Torch/MLX
environment: the previous `semantic` and `mlx` extras have incompatible
Transformers requirements. The separate `fusion` extra addresses this dependency
conflict without changing either existing extra.

Study84 completed actual installed inference for all 200 Romanian cases with
zero changed decisions and zero score difference from study80. The run used
Torch 2.14.0, Transformers 5.17.0, MLX 0.32.2 and MLX-LM 0.31.3 in one process.
Observed elapsed inference time was 292.10 seconds on the local machine, including
lazy model loading. This is a runtime compatibility check on a reused screen,
not an independent accuracy evaluation or a general latency benchmark.
