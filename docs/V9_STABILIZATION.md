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

### Completed native Romanian run

Native-22 Romanian completed three epochs and selected epoch 2.5 using the
frozen development rule. Reloaded development scores match exactly. Calibration
misses 0/614 formal and 0/216 conversational concerns, but produces 241/1,386
formal false reviews (17.39%) and 7/243 conversational false reviews (2.88%).
It fails the formal false-review requirement. The development-only threshold-34
comparison finds no eligible alternative and retains the original artifact.
These results do not qualify the model for stable release. Ukrainian training
and the structured interpretation comparison remain in progress.

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

### Native interpretation-head fallback

`native-fields-43` is a predeclared fallback if the fusion comparison still fails.
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
The fallback waits until the installed-runtime queue finishes and runs only if
the fusion accuracy requirements remain unmet and enough work-window time
remains. Tokenizer preflight covers all 44,105 training inputs without truncation.
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
