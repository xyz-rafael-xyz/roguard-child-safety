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

Study85 selected its half-epoch checkpoint on development and stress-set counts.
Calibration formal counts improved to zero misses and 12/1,386 false reviews;
conversational counts were 9/164 misses and 8/223 false reviews. The additional
stress set still falsely reviewed 25% of benign examples. This candidate is not
qualified; installed-needs-86 therefore skipped its conditional replay.

Study87 starts a separate Qwen3.5-9B control using a pinned MLX 4-bit conversion,
a fixed binary policy, disabled thinking, and the existing 400-case bilingual
screen. The next vocabulary token must be a valid decision digit, otherwise the
model abstains and that counts as error. No reserved examples are evaluated.
The initial bfloat16 output-projection preflight failed before predictions; its
source and status are preserved. A float32 arithmetic amendment must pass the
same projection and cache parity tolerances before the screen can run.

The float32 Qwen control completed 400/400 valid decisions. It did not meet
accuracy gates: conversational misses were 12/50 RO and 11/50 UK, and formal
misses were 17/50 RO and 21/50 UK. False reviews were respectively 3/50, 7/50,
2/50 and 4/50. The projection probability discrepancy was 0.000001401 and the
cached/uncached discrepancy 0.000003666. Larger model size alone did not solve
the task.

Study88 therefore tests task adaptation of only Qwen's final full-attention
block (about 639,000 trainable parameters). It uses 1,024 balanced synthetic
training examples and 80 fixed original-development examples, with protected
input exclusion. Checkpoints are selected by development target loss; the
existing 400-case screen is evaluated afterward. Study89 prepares an equal-length
batched decoder whose complete screen must match study88's single-message
outputs before it can be adopted for larger evaluations.

Study90 freezes the 9,691 original development/calibration inputs for the selected
study88 Qwen adapter. Language thresholds use development scores only; calibration
cannot move them. Invalid output tokens remain abstentions and count against the
same 99% recall / 5% false-review requirements. The study has not yet reported
qualification results.

The experimental Qwen runtime accepts a pinned local bundle through
`python -m roguard.model_bundle --device mps`. It verifies the base inventory and complete adapter
tensor schema, copies verified assets into a private snapshot before loading,
and validates the whole input batch before inference. Future turns are excluded;
outputs never grant notification or other external permissions. Study91 will
compare the installed wheel with all 400 study88 reference outputs. This runtime
is available for explicit experiments and does not change the default model or
the stable release. The updated software suite passed 797 tests and 129 subtests,
with 25 skipped; actual-model parity remains pending.

Study92 preregisters a complementary-score experiment using native22 for Romanian,
needs85 for Ukrainian, and the fixed study88 Qwen adapter. Seven features describe
the two model scores and their agreement; text, labels, source strata and case
identifiers are excluded from predictive features. Five connected development
folds select regularization, and development scores alone fix the final threshold.
Invalid Qwen tokens remain abstentions. Development, out-of-fold development and
calibration must each meet the existing count gates. Results are pending.

Study93 also retains all 216 authored Ukrainian stress cases from study85. It
evaluates the fixed Qwen and combined-model decisions without fitting on those
cases. Passing the main calibration audit does not waive this earlier failed
stress check.

Study94 is a conditional repair, queued only if the preceding studies still do
not provide a qualifying Ukrainian candidate. The study88 validation loss rose
from 0.264 at step255 to 0.950 at step511, while its earlier best checkpoint was
preserved. The repair warm-starts the final selected study88 checkpoint, lowers
the initial learning rate from 0.0001 to 0.00001, and decays it to 0.000001.
Training and development inputs are unchanged.

To reduce repeated work, the proposed trainer reuses only the common-prefix
states below the trainable final block. Before any optimizer update, the cached
calculation must match the original loss and gradients within 0.001 on fixed
Romanian and Ukrainian development cases. This acceleration has not yet passed
that actual-model preflight and is not claimed as a measured speed improvement.

**Study88 operational deviation:** training was stopped by the operator after the
last reported step800, before the planned 1,024 steps completed. Later validation
losses (0.950 at step511 and 0.432 at step767) were worse than the selected
step255 loss (0.264). Severe host memory pressure had also slowed training to
about 22 seconds per step. The selected adapter was verified against its saved
checkpoint hash before any screen result was read. The exact termination step
between reports is not claimed. See
[the deviation record](../research/v9-stabilization/qwen-adapter-88/early-stop-deviation.json)
and [recorded training metrics](../research/v9-stabilization/qwen-adapter-88/training-metrics.json).
Evaluation continues on that fixed selected checkpoint; the original training
protocol remains unchanged in the record.

Study95 prepares a smaller Ukrainian native-model repair before conditional
study94. Its 864 new synthetic variants contrast bare emotional-disconnection
signals with ordinary solitude and self-care, using the existing scope labels.
They do not overlap the original development/calibration inputs or the retained
216-case stress set after normalization. Training starts from study85, uses the
last four encoder blocks and classifier at a lower learning rate, and retains
the original calibration split. Checkpoint selection explicitly uses the known
stress set as development evidence; it is not described as independent testing.
The experiment is skipped if an earlier candidate already qualifies.

The selected study88 adapter completed its screen with 334/400 correct decisions,
zero abstentions, 39 missed concerns and 27 false reviews. This is an improvement
over the unadapted control's 323/400 correct decisions, but does not meet the
registered accuracy gates. Study89 reproduced all 400 decisions with a maximum
score difference of 0.000011713. Batching took 317.46 seconds versus 228.49 for
single-message inference on this host, so study90 selected the single-message
reference for its full audit before reading any full-audit outputs.

Study96 will replay every study95 development, calibration and stress example
through the separately installed wheel if study95 qualifies. It requires zero
decision differences, no abstentions, a maximum score difference of 0.0001,
and the same accuracy gates in every populated group. This check has been
prepared but has not yet run.

The proposed study94 prefix cache passed a small CPU test using actual Qwen3.5
hybrid recurrent/attention blocks. Losses and all trainable gradients matched
the uncached calculation within 0.00001 across repeated inputs and optimizer
updates, both with and without the trainer's gradient-checkpointing wrapper.
This checks the cache implementation; the full-size model must still
pass its separate preflight before training starts.

### One interface for separate language candidates

The source checkout now provides `roguard.language_bundle.LanguageReviewModel`
for explicitly configured Romanian and Ukrainian local bundles. It validates
the entire batch before loading a model, groups inputs by their declared
language, restores their original order, and returns each selected model's
report unchanged. It checks the input, focal turn and model bindings and rejects
reports claiming permission to notify or act. Future turns beyond the focus
never reach the selected model.

A local manifest contains exactly these fields, with actual SHA256 values from
your trusted model configuration:

```json
{
  "format": "roguard-language-review-bundle-v1",
  "languages": {
    "ro": {"path": "ro/bundle.json", "sha256": "<Romanian bundle SHA256>", "device": "cpu"},
    "uk": {"path": "uk/bundle.json", "sha256": "<Ukrainian bundle SHA256>", "device": "cpu"}
  }
}
```

Both child manifests must be inside the language bundle's directory. Choose
`mps` for a Qwen child on Apple Silicon. Other model requirements still apply;
the dispatcher does not install dependencies or download weights. It retains
only the most recently used language model and releases its own reference
before switching languages.

```sh
python -m roguard.language_bundle --bundle models/languages.json --sha256 "$TRUSTED_BUNDLE_SHA256" --input conversation.json
```

Twenty tests cover mixed-language ordering, model reuse and release, batch
validation before loading, future-turn isolation, output bindings, tampering,
path escapes, incomplete results and unsupported claims. These are software
checks using test doubles; the combined language dispatcher still needs an
actual-model check with the final qualified candidates. No default model is
selected by this addition.

Study97 tests whether the repaired Ukrainian native scores improve the existing
three-expert combination. Its first run uses study85; the same frozen recipe is
also available for study95 after training. The other two experts remain the
study66 semantic model and selected study21 interpreter. Five connected
development folds choose among four existing shallow-tree configurations, with
no text, case IDs or gold interpretation fields as predictive features. The
final threshold covers the development and out-of-fold concern predictions.
Calibration is audited separately. Even if those counts pass, the retained
216-case stress set and actual installed composite still require verification.
This study fits saved component outputs on the CPU and does not interrupt the
running Qwen inference audit.

The study97 run with native85 passes development and out-of-fold counts, but
fails conversational calibration: 14/164 concerns missed and 4/223 benign
messages reviewed. Formal calibration has zero misses and 9/1,386 false
reviews. It is not a qualifying candidate.

**Execution-order amendment:** study90 was paused after 2,501 of its 9,691
predictions to run the smaller, already frozen study95 repair sooner. The saved
prediction prefix, input set, adapter and original evaluator are hash-bound in
[the pause record](../research/v9-stabilization/qwen-qualification-90/pause-record.json).
Study95's data, training and selection rules remain unchanged. No partial Qwen
accuracy results selected this scheduling change. Study98 prepares continuation
of the same audit, requiring eight fixed saved-prefix replays to agree before
processing the remaining inputs. Its timing covers only the remaining work.
If study95 and its installed verification qualify, the incomplete Qwen study
will remain paused while final bilingual qualification takes priority.

Study95 completed both epochs and recovered every calibration concern, but
increased conversational false reviews to 51/223 and retained-stress false
reviews to 23/172. Its direct classifier fails qualification; study96 therefore
skipped direct-model replay. Development diagnosis found low scores for
fictional retractions after a repair corpus emphasizing affirmation and denial.

The same frozen study97 combination using the completed study95 scores passes
its development, out-of-fold and calibration count gates. Ukrainian calibration
has 0/614 missed formal concerns with 18/1,386 false reviews, and 1/164 missed
conversational concerns with 9/223 false reviews (99.39% recall; 4.04% false
reviews). This is reused synthetic calibration, not an independent accuracy
estimate. The original study97 recipe selected the tree and threshold using
development predictions only.

Study99 now checks this Ukrainian candidate through the separately installed
wheel with all three actual models. Its fixed 200-case screen must reproduce
the saved decisions without abstentions and with score differences at most
0.001; the 216 retained stress cases must then pass their unchanged counts.
The Qwen audit remains paused while this potentially qualifying candidate is
checked. A short restart reproduced its fixed saved-prefix checks and evaluated
31 additional inputs; those intermediate outputs remain recorded, without a
full-audit qualification claim.

Study100 is conditional on all study99 checks passing. It first exercises the
installed bilingual dispatcher on 40 existing screen inputs, interleaved across
languages and balanced by style and label. Every decision must match the prior
actual-model outputs, with no abstentions and score differences at most 0.001.
Only after that integration check passes does it freeze the candidate and
integration hashes and open the three previously reserved synthetic probes.
Those probes must pass per language and probe, with abstentions counted as
errors and repeated exact inputs also reported separately. If any prerequisite
fails, the reserved inputs remain unopened. The small related probe families
cannot establish population accuracy even if all their counts pass.

Study99 completed with exact parity on all 200 screen decisions and scores.
The retained stress check has 3/44 missed concerns and 6/172 false reviews.
All three misses are plain-message variants of one unmet-food/care family.
Study100 therefore skipped both bilingual inference and reserved evaluation.
The reserved probes remain unopened.

Study101 tested a new cutoff using only original development/out-of-fold and
known stress scores. Recovering the three misses required lowering the cutoff
from 0.04358 to 0.00342, which produced 144/172 stress false reviews and failed
the development groups. The candidate was rejected before a calibration audit.

Study102 is now training a fuller scope repair. Its 7,200 authored inputs cover
96 content families, including negated-access concerns, ordinary practical
questions, emotional disconnection and voluntary solitude. All four assertion
types, all three subject and time categories, and both single/multiple-turn
layouts are included. It starts from study95, retains that unmodified model as
a selection candidate, and adds historical replay from study85. On replay
examples only, a frozen study85 teacher constrains changes to its predictions;
new authored examples use their labels without that constraint. Sampling also
balances retractions. Selection uses development and the known stress set;
calibration remains separate. Three tests verify scope labels, input exclusion
and the masked distillation gradients. This is a development-informed repair,
not a new independent evaluation.

Study102's exact-token audit found zero overlap with the 88 distinct Ukrainian
reserved token inputs and no conflicting binary targets in its accumulated
training history. This checks exact input separation, not semantic independence.
Study103 will replay every development, calibration and stress row through the
installed native model if study102 qualifies. Otherwise study104 applies the
same fixed three-expert development recipe to its new native scores; any such
combined candidate still needs actual stress and runtime verification.

### Installed composite verification (study105)

If the distilled three-expert candidate in study104 passes its original development,
connected-fold and calibration gates, study105 replays the same 200 installed-package
screen inputs and 216 known development stress inputs used in study99. It requires
zero changed screen decisions, no abstentions and score differences at most 0.001;
the stress gate requires zero missed concerns and at most 5% false reviews. The
candidate, data and installed runtime are pinned before inference. Actual component
outputs are retained for diagnosis without replacing the pipeline execution path.
This is conditional verification, not a new accuracy result. Reserved neural
inference remains unopened.

### Completed distillation and development stress repair

Study102 selected epoch1 by the frozen development rule. Its original Ukrainian
calibration results are 0/614 missed concerns and 21/1,386 false reviews for
formal inputs, and 2/164 misses with 22/223 false reviews for conversational
inputs. It retains all 44 known stress concerns but incorrectly reviews 19/172
benign stress inputs. Study103 therefore skips installed qualification.

Study104 combines that checkpoint with the unchanged semantic and interpreter
models. Development and connected-fold counts pass, but conversational
calibration has 3/164 misses and 17/223 false reviews. Study105 consequently
skips; these candidates are not promoted and no reserved inference runs.

Studies106–107 return to the earlier study97/95 combination, which passed the
original development and calibration gates. Study106 captures each actual model's
outputs on the already inspected stress inputs, requiring exact decision replay
against study99. Study107 then treats this known stress population explicitly
as development data for the combiner. It holds out entire stress content families
across folds and retains the original connected development folds. The four
predeclared shallow-tree configurations receive equal class/population weights;
calibration is audited only if every development and fold group passes. These
reused stress cases are not an independent test. Neither the thresholds nor the
release accuracy requirements are relaxed.

Study108 is the conditional installed-package check for a qualifying study107
candidate. It requires the same 200-example screen parity and 216-example stress
counts, with the latter explicitly treated as development evidence.

Study109 supplements the prior training-history separation checks with every
original development and calibration input and the known stress set. All three
pinned tokenizer audits completed: 4,917 Romanian rows, 4,990 Ukrainian rows and
9,907 combined interpreter rows. Each found zero conflicting binary targets and
zero exact token-input overlap with any of the three reserved probes. No neural
model was loaded for these checks.

Study110 will verify the installed bilingual dispatcher only if study108 and
all prerequisite accuracy checks pass. Its 40-example integration screen must
match the selected models before the reserved probes are opened. Access to
those probes is recorded before loading them, and predictions are saved after
each batch. The existing per-language, per-probe and deduplicated accuracy gates
remain unchanged. No reserved neural results are available yet.

## Scope consensus repair and installed verification

Study106 replays all 216 known stress decisions and scores exactly. Its captured
model outputs show that the interpreter recognizes the three previously missed
food-and-care concerns; the learned combiner suppresses those proposals.
Study107's added stress-family tree fitting fails its development gates, so it
never audits calibration. Studies108 and110 consequently skip.

Studies111–113 test a narrow addition that preserves every original review.
Study111's confidence floor of 0.9 recovers none of the three misses. Study112
selects 0.8095 using only the original development and base fold results and
recovers two. Both fail the stress gate and skip calibration.

Study113 requires a positive native proposal, an interpreter review proposal,
and agreement between the two field models on real, non-negated scope. Within
those agreed scope categories it uses the stronger field confidence, combined
with the interpreter's concern confidence. These conditional model scores are
not calibrated probabilities. The cutoff is selected only on original
development and base fold outputs, requiring at most 1% false reviews in each
group; calibration and stress do not select it. A stress-audit unpacking bug was
fixed before any audit results were obtained; the selected cutoff was unchanged
and both source versions are retained.

The selected cutoff is 0.8472052216529846. Original calibration has 0/614 missed
formal concerns and 18/1,386 false reviews; conversational calibration has 1/164
misses and 11/223 false reviews (99.39% recall, 4.93% false reviews). Known stress
has 0/44 misses and 6/172 false reviews. All current count gates pass. This remains
a development-informed repair evaluated on reused synthetic data.

The new local runtime preserves component abstentions and all existing input,
model and permission checks. Its portable calculation matches every one of the
216 captured actual-model stress outputs exactly. Twelve runtime tests and
21 bilingual evidence-gate tests pass. Study114 now loads all models from a
newly installed wheel and compares 200 screen plus 216 stress results. Study115
is queued to verify the bilingual dispatcher and, only if every prerequisite
passes, evaluate the frozen reserved probes with recorded access and partial
results. Reserved neural inference has not run yet. The stable default is
unchanged until qualification and release work finish.

Study114 completed both actual installed-model splits with zero decision or
score differences and no abstentions. Its screen has 0/100 misses and 1/100 false
reviews; stress has 0/44 misses and 6/172 false reviews. Study115 initially failed
to initialize Metal in the sandbox before producing integration predictions.
The identical frozen run was restarted with GPU access enabled. Its 40-example
bilingual integration check passes exactly, and reserved access has now been
recorded. The reserved evaluation is running; its accuracy result is not yet
available.


### Study115 reserved result and next repair

The evaluation completed and failed. Its deduplicated inputs give Romanian
4/60 missed concerns and 1/60 false reviews; Ukrainian 2/56 missed concerns and
27/57 false reviews. There are no abstentions. Original and lexical-family
probes overlap and are not independent observations. Exact runtime parity did
not imply semantic generalization. No stable v9 promotion is authorized by
these results.

All three previously reserved probes are now exposed regression data. Study116
replays the unchanged candidate solely to capture component diagnoses. The next
training iteration will use broader paired scope examples and a new evaluation
frozen before candidate selection. Original count gates remain unchanged.

### Paired scope curriculum and entailment transfer

Study116 replays the exposed study115 inputs and captures all three components.
Both languages reproduce the previous scores and decisions exactly. This is
failure diagnosis on known regression inputs, not another independent test.

Study117 freezes 1,380 authored training rows, 180 development rows and 300 test
rows per language. Content phrases and scope framing differ between partitions.
Counterfactual pairs contrast fiction with real disclosure, denial with
confirmation, fictional characters with real peers, and topic changes with
continued concern. Cross entropy is supplemented by a logit-margin pair loss;
3,072 historical examples per language provide replay. Each native encoder
trains its final four blocks for two epochs. A token audit excludes all protected
inputs and checks the new test against inherited project training history.

Both candidates fail the unchanged accuracy gates. Romanian selection at epoch
1.5 has zero calibration misses but 244/1,386 formal and 37/243 conversational
false reviews. On the exposed probes, collapsed by the native normalized input,
it has zero misses and 8/44 false reviews. Ukrainian calibration has zero misses
but 90/1,386 formal and 31/223 conversational false reviews; exposed regression
has 1/44 misses and 21/44 false reviews. Normalized counts differ from study115's
raw-input counts because whitespace and Unicode variants collapse. Study118
verifies the installed native runtime; its conditional fresh test cannot open
unless both candidates qualify. The 600 new test messages remain uninferred.

Study119 prepares a separate supervised transfer of the pinned multilingual
mDeBERTa entailment encoder. The earlier study17 zero-shot control failed, so
this is not a claim that unadapted entailment solves the task. It trains the
final four blocks, pooler and classification head on the paired curriculum and
512 historical examples per language/style/label group. Entailment, neutral and
contradiction all remain in the score normalization. Development selects the
checkpoint and cutoffs; calibration and exposed115 inputs are post-selection
audits. No prior RoGuard checkpoint is inherited. The upstream model reports
Ukrainian NLI training, which does not establish Romanian or RoGuard accuracy.
[Upstream model card](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7).

The separate runtime validates pinned tokenizer/configuration bytes and finite
safetensors weights, uses only the selected conversation prefix, supports both
language thresholds, and abstains on excessive token length. It grants no action
or notification permission. Fifteen runtime tests pass. Study120 will verify
actual installed inference on Metal and CPU before any conditional fresh test.
The full suite after study118 passed 947 tests, with 30 optional tests skipped;
the added entailment loss and gate checks have separate focused coverage.

Study118 completed all nine installed native replay splits with zero changed
classifications. The largest CPU-versus-training score difference was
0.000007153. Its fresh test was skipped because the accuracy prerequisites
failed. A development-only threshold-feasibility check also confirms that no
single cutoff on either selected native checkpoint can satisfy all required
recall and false-review limits. The fix therefore needs better learned scores,
not merely another cutoff. Study119 is now running.

### Conditional separation of support and revision review

Study121 is frozen as a conditional follow-up after study120: it runs only if
study119 fails its prior accuracy gates. It learns two entailment tasks with one
shared encoder: a real-person support concern, and a withdrawal or conflicting
fiction/real-person interpretation requiring review. Their Boolean OR is tested
to reproduce the existing policy for all 432 complete attribute combinations.
Separate output fields identify the reason for review. Labels and release gates do not
change.

The study reuses the same 6,856 training inputs with two hypotheses per input,
weights each language/task/class group equally in the loss, and preserves the
complete authored counterfactual pairs. Partial pair groups from historical
replay remain individually supervised examples. It starts from the selected
study119 checkpoint and trains one further epoch. Parent data hashes are recorded
before training. Calibration and exposed115 are audit-only; the fresh600 remain
excluded from training and selection.

The runtime emits separate support and revision proposals, retains the neutral
NLI probability in normalization, and grants no action authority. Study122 will
compare both component scores, their maximum and every decision through an
isolated package, followed by a CPU check. Any earlier fresh-test access by118
or120 prevents it from claiming another first-use evaluation.

The full suite with these runtime changes passed 980 tests, with 31 optional
skips and 129 subtests. Subsequent CLI integration adds checked loading of either
entailment profile and points portable-model users to the semantic dependency
extra. These are preparatory capabilities; no qualified profile is published yet.

Study119 completed both epochs and selected the half-epoch checkpoint. Its
calibration remains below the release gates: Romanian has 1/614 formal misses,
1,008/1,386 formal false reviews and 143/243 conversational false reviews;
Ukrainian has zero calibration misses but 1,065/1,386 formal and 119/223
conversational false reviews. Exposed regression has zero misses in either
language, with 28/44 Romanian and 27/44 Ukrainian false reviews. This candidate is
rejected. These counts describe its selected conservative operating point.

Study120 reproduced all checked installed decisions, including 40 CPU cases;
the largest score difference was 0.000018478. It skipped the fresh test because
accuracy failed. Study121 has started its separate support/revision refinement.

### Reason coverage and separate score cutoffs (studies123–124)

An audit of study121's replay found 55 Romanian and only nine Ukrainian
withdrawal examples, despite 1,665 and 1,711 personal-support examples,
respectively. Binary-label sampling had obscured this imbalance. Increasing the
loss weight of rare examples does not increase their linguistic variety.

Study123 retains that replay, adds 2,304 authored withdrawal contrasts, includes
every eligible historical withdrawal example, and fills historical groups by
source, policy reason, subject, assertion and time to 32 distinct inputs where
available. It contains 19,109 conversations, including 1,359 Romanian and 728
Ukrainian withdrawal examples. Both sides of each new contrast withdraw a
statement; its concerning versus ordinary content determines the review label.
Real, reported and fictional contexts follow the unchanged evidence policy.
All data remain project-authored synthetic examples, without independent
annotation of this study. Tokenized training inputs have zero overlap with the
protected development, calibration, exposed regression and fresh evaluation
inputs.

The study is queued after study122, conditional on study121 failing its accuracy
prerequisites. It trains two epochs from the selected study121 checkpoint.
Support and revision have separate development-selected cutoffs per language.
An exact search minimizes the worst group false-review rate while covering
every development positive; a brute-force comparison verifies the search on
small populations including score ties and endpoints. Checkpoints are selected
on development only. Calibration and exposed115 are subsequent audits. The
600-case fresh evaluation remains excluded from training and selection.

The new runtime reports both scores, both cutoffs and the reason for review.
It grants no notification or external-action authority. Study124 checks the
actual installed package on Metal and CPU before conditionally opening the
fresh evaluation. No stable-release or real-world accuracy claim follows from
the preparation or the software tests.

The full software suite after these changes passed 1,008 tests and 129 subtests,
with 32 optional skips. The cutoff-search and authored-contrast checks also
passed all 10 focused tests in the NumPy-enabled environment.

Study121 completed and selected its half-epoch checkpoint. It missed no positive
calibration cases at its conservative cutoff, but produced 1,135/1,386 formal and
164/243 conversational false reviews in Romanian, and 1,015/1,386 formal and
101/223 conversational false reviews in Ukrainian. Exposed regression false
reviews were 26/44 and 19/44, respectively. These results reject the candidate.
Its development-only two-cutoff diagnosis also remains above the 5% false-review
limit. Study122 reproduced all 1,152 checked installed decisions and 40 CPU
decisions; maximum component-score differences were 0.000013829 on Metal and
0.000038385 on CPU. It skipped the fresh evaluation. Study123 then started the
expanded training run.

Studies125–126 prepare a conditional comparison using the selected study117
Romanian and Ukrainian encoders with the same study123 continuation curriculum.
Their separate tokenizer audit found 9,176 distinct Romanian and 9,933 distinct
Ukrainian inputs, with zero overlaps against protected evaluation inputs or
conflicting duplicate labels. Inherited training histories also have zero
protected-input overlap. The checkpoint histories differ from the entailment
model, so this comparison does not isolate architecture as the only variable.

The comparison waits for study124 and runs only if study123 fails its prior
accuracy gates. Each language receives two epochs with paired supervision,
development-only checkpoint selection, and post-selection calibration and
exposed regression audits. Study126 checks every prior split through the
installed CPU runtime before any conditional fresh evaluation. Its gate requires
every language-specific group and rejects missing counts, incomplete runs and
prior fresh-test access. Three focused gate tests pass.

The entailment route also has a conditional full CPU replay after study124
passes. The 40-case CPU smoke check is insufficient to establish accuracy for
all cases on the default CPU runtime. The additional replay covers all nine
prior populations and both already-opened fresh partitions, requires unchanged
decisions, checks all three score fields within 0.001, and recomputes every
accuracy group. It is a deployment verification, not a second first-use test.
Two prerequisite tests verify that success flags cannot override missing or
failing fresh counts. The final release package still needs an actual cached
bundle/CLI check and byte-identical verified inference code.

### Prepared grammatical-gender coverage intervention (study127)

At study123's half-epoch checkpoint, development false reviews fell from
723 to 328 on Romanian formal inputs and from 645 to 178 on Ukrainian formal
inputs, with zero development misses. The broader scope cases still failed:
21/90 Romanian and 35/90 Ukrainian negative examples were reviewed. The
reported-fiction subgroup contributed 12 false reviews in each language. The
training corpus contained no occurrences of the explicit feminine-character
stems `eroin` or `героїн`. This is a coverage gap consistent with the errors;
it does not prove that grammatical gender alone caused them.

Study127 prepares 3,456 additional training examples, balanced across masculine
and feminine fictional characters and real peers. Each language has 1,152
concerning real/fiction contrasts and 576 ordinary-request examples, so real-peer
wording alone does not imply concern. Quoted speech uses the corresponding
grammatical forms. The content comes from study117's training partition; the
framing is newly authored. Three focused tests check labels, pair completeness,
balance, inflection and absence of copied protected examples.

Separate entailment, Romanian and Ukrainian tokenizer audits found zero overlap
with protected evaluation inputs or the existing training inputs. The added
examples contain at most 112 entailment tokens or 72 native-encoder tokens.
This intervention is prepared only; it has not been added to the running study123
or the frozen study125 comparison. All examples remain project-authored
synthetic data without independent annotation.

The subsequent full suite in the neural environment passed 1,049 tests and 129
subtests. Seventeen checks were skipped there: three MLX checks were covered by
the separate 13-test MLX CPU run, and fourteen historical adapter checks require
historical weight files absent from this checkout. Environment versions and log
hashes are recorded in
[`software-verification-20261019.json`](../research/v9-stabilization/software-verification-20261019.json).
These software checks do not establish classifier accuracy.

Studies128–129 schedule the prepared study127 contrasts as a conditional native
encoder continuation. They wait for the current comparisons and run only if
both study123 and the bilingual study125 candidate fail their prior accuracy
gates. Each language retains its complete study125 training set and adds the
1,728 new contrasts and ordinary examples. Parent artifacts and input hashes are
bound before training; complete morphology pairs participate in the existing
paired loss. Initial and half-epoch development checks select the checkpoint.
Study129 then replays every prior population on the installed CPU runtime and
refuses fresh-test reuse if any earlier study opened it. The two native release
gate implementations pass all six focused checks. No study128 training has
started yet, and no result from this scheduled continuation is claimed.

### Cutoff feasibility and scope-learning context

The training runs above select cutoffs with zero development misses. A separate
diagnostic now searches the exact two-cutoff frontier using the actual release
requirement of at least 99% recall within each group. The allowable miss count is
computed separately for each group; a group with fewer than 100 positive cases
still permits no misses. Seven tests compare the search with exhaustive
enumeration, including tied scores and small groups.

Applied to the completed study121 predictions, this diagnostic still fails the
false-review requirement in both languages. Its best worst-group false-review
rates are 41.90% for Romanian and 36.67% for Ukrainian. The diagnostic reads only
development predictions, changes no model or labels, and provides no new
calibration or test result. The bound inputs and full counts are in
[`budgeted-development-diagnosis.json`](../research/v9-stabilization/policy-entailment-121/budgeted-development-diagnosis.json).

The use of controlled scope contrasts has prior research precedent.
[ScoNe (She et al., 2023)](https://aclanthology.org/2023.acl-short.154/)
tests whether negation changes an inference under controlled combinations and
reports successful many-shot fine-tuning of RoBERTa and DeBERTa on that task.
[Yang et al.](https://arxiv.org/abs/2410.20710) combine counterfactual sentence
pairs with contrastive training to improve NLI robustness. These findings
motivate the method; they do not establish Romanian or Ukrainian child-safety
accuracy, and RoGuard does not claim to have invented counterfactual training.

### Completed reason-covered entailment result (studies123–124)

Study123 completed both epochs and selected epoch2 by its frozen development
rule. It improves several earlier measurements but does not qualify for release.
The post-selection audit is:

| Population | Romanian misses / positives | Romanian false reviews / negatives | Ukrainian misses / positives | Ukrainian false reviews / negatives |
| --- | ---: | ---: | ---: | ---: |
| Formal calibration | 0 / 614 | 490 / 1,386 | 0 / 614 | 1 / 1,386 |
| Conversational calibration | 0 / 216 | 23 / 243 | 6 / 164 | 16 / 223 |
| Exposed scope regression | 0 / 44 | 6 / 44 | 0 / 44 | 5 / 44 |

All audit populations had zero abstentions. The development scope check also
fails, with 15/90 Romanian and 29/90 Ukrainian false reviews at the selected
cutoffs. A separate development-only diagnostic using the existing 99% recall
requirement reduces those counts to 15/90 and 21/90, still above the limit. That
search now resolves equal false-review objectives in favor of fewer misses and
recovers positives whenever the negative decisions can be preserved. Exhaustive
comparison tests cover its full objective. It changes no candidate artifact and
does not use calibration or reserved-test predictions.

Study124 reproduced all 1,152 checked installed decisions and all 40 CPU smoke
decisions. Maximum component-score differences were 0.0000025034 on Metal and
0.0000141711 on CPU. The fresh 600-case evaluation was skipped because the prior
accuracy requirements failed. The full CPU qualification branch was likewise
unnecessary for this rejected candidate. The queued study125 native comparison
can now proceed with the frozen continuation inputs.

### Additional scope contrasts prepared (study130)

The completed study123 development errors cluster in reported fiction and topic
changes for Romanian, and denial, reported fiction and performance contexts for
Ukrainian. Study130 therefore prepares 3,840 new training-only examples using
study117 training content and newly authored framing. Each language contributes
1,344 examples in opposed-label pairs and 576 ordinary requests under the same
scope wording. The pairs cover personal attribution, denial, fictional versus
autobiographical performance, and unrelated focal questions versus continued
concern. Topic-change pairs retain exactly the same prior concern in both
members; their labels describe the focal message and do not erase history.

Three data tests pass. Romanian, Ukrainian and entailment tokenizer audits find
zero overlap with protected evaluation inputs or the existing study123 and
study127 training inputs. Maximum lengths are 80, 87 and 123 tokens respectively.
No neural inference was performed for this preparation. The data and generator
are frozen in
[`scope-operators-130`](../research/v9-stabilization/scope-operators-130).
They have not been added to the running study125 or the already frozen study128.
This is a prepared follow-up intervention, with no accuracy result claimed.

Studies131–132 schedule that intervention only if the earlier candidates still
fail. Study131 retains the complete selected study128 continuation corpus and
adds the 1,920 new rows per language. Its two-epoch training objective remains
paired crossentropy, now including the new opposed scope pairs. Development
selection uses the existing 99% recall requirement separately for each group;
among feasible cutoffs it minimizes the worst false-review rate, their sum, and
then misses. This changes the earlier zero-miss development selection rule,
while keeping the release gates unchanged. Eight cutoff tests cover small-group
miss budgets, ties, inseparable scores and invalid inputs.

Study132 performs full installed CPU replay before any conditional reserved-test
access and rejects prior access by studies118,120,122,124,126 or129. The three
native verification implementations pass all nine shared gate tests. Both stages
are frozen before execution and wait for the earlier queued studies. No
study131 training or study132 accuracy result is claimed at preparation time.

Studies133–134 prepare a further conditional continuation with every native
encoder parameter trainable. It retains the study131 corpus, paired objective
and per-group selection rule. Learning rates decay by 0.85 toward earlier
layers, from 0.000005 at the final encoder layer and classifier to approximately
0.000000711 at the embeddings. The purpose is to test whether adapting only the
top four layers leaves a limitation in context handling. This sequential
continuation is not an isolated architectural ablation.

Four optimizer tests verify complete parameter coverage, gradients, updates,
schedule ratios and rejection of omitted or multiply assigned parameters.
Configuration-only checks on the pinned native architectures cover 124,442,882
Romanian and 125,979,650 Ukrainian parameters in 14 optimization groups each;
these checks perform no model inference. Study134 uses the same full installed
CPU replay and additionally rejects reserved-test access by study132. All twelve
shared native gate tests pass. This fallback waits for the prior experiments and
has no training or accuracy result yet.

### Completed native curriculum comparison (studies125–126)

Both study125 languages selected epoch0, retaining their starting study117
checkpoints because the broader replay curriculum did not improve the frozen
development objective. This rejects the curriculum-only continuation as an
accuracy improvement. Romanian calibration has zero misses, with 244/1,386
formal and 37/243 conversational false reviews. Ukrainian calibration has zero
misses, with 90/1,386 formal and 31/223 conversational false reviews. On exposed
scope regression, Romanian has zero misses and 8/44 false reviews; Ukrainian
has one miss among 44 positives and 21/44 false reviews.

Study126 reproduced all 10,443 decisions across the nine prior populations on
the installed CPU runtime. Its largest score difference was 0.0000071526.
The reserved 600-case evaluation remains uninferred because accuracy prerequisites
failed. The prepared study128 grammatical-gender continuation follows this
completed comparison; the original data, predictions and results remain available
for audit.

### Additional indirect-distress content prepared (study135)

Study135 prepares 2,544 new training-only examples, 1,272 per language, following
inspection of the completed study125 Romanian development scores. It tests a
coverage hypothesis: indirect descriptions of distress may need more content
variation as well as the scope variation already queued. Twelve new concern
bodies per language use bodily tension, difficulty speaking, concealed distress
or social isolation with personal context. Twelve ordinary bodies use similar
vocabulary in schoolwork, music, drawing and general biology questions. The
same study130 attribution and fiction contrasts appear around both kinds of
content. Topic-change pairs retain identical prior messages. Positive labels
request review; they are not diagnoses or findings about the truth of a message.

Each language has 336 opposed pairs, 588 ordinary controls and 12 bare positive
examples. Three generator tests verify policy consistency, paired history,
ordinary controls and separation from all previous scope content partitions.
Audits with the two native tokenizers and the NLI tokenizer find zero exact
input overlaps with protected evaluation, the reserved 600 cases, inherited
study125 training history, or the prepared study127/130 additions. The longest
new input has 147 NLI tokens. No neural inference or training is part of this
preparation. These are project-authored synthetic probes, not clinical evidence
or new independent expert review. The active and queued studies128–134 remain
unchanged; study135 has no accuracy result.

The private wheel also completed four actual `roguard.review_cli` invocations
through its packaged profile and verified offline cache: one previously evaluated
positive and negative development example per language. All four decisions
matched the selected study117 references; the largest score difference was
8.2e-10. Each report withheld message text and granted no notification or external
action authority. The subprocesses imported the separately installed wheel, with
Hugging Face offline settings enabled, and took 6.8–8.0 seconds including cold
startup and cache checks. This checks the installed CLI path only; these models
remain unqualified and the reserved test was not used. The measured record is
`research/v9-stabilization/cached-cli-preflight-20261019.json`.

### Completed grammatical-gender continuation (studies128–129)

Study128 selected epoch0 for Romanian and epoch1.5 for Ukrainian. The Romanian
results therefore remain those of study125. Ukrainian development scope false
reviews fell from 41 to 39 of 90 negatives, and stress false reviews from 44 to
41 of 172 negatives, with zero development misses. Calibration still fails:
there are zero misses, but 166/1,386 formal and 25/223 conversational false
reviews. Exposed regression has zero misses and 19/44 false reviews. This is
not sufficient evidence of an overall accuracy improvement or release readiness.

Study129 reproduced all 10,443 prior decisions on the installed CPU runtime,
with a maximum score difference of 0.0000099838. Accuracy prerequisites failed,
so the reserved 600-case test remained uninferred. The queued study131 scope
operator continuation follows this completed result.

### Conversational scope wording prepared (study136)

Study136 adds a small training-only set of 120 individually authored examples,
60 per language: 24 opposed pairs and 12 ordinary controls per language.
The exchanges describe fiction, personal distress, general vocabulary questions,
and changes of focal topic without a template cross-product. The ordinary
controls include schoolwork, games and explicitly pleasant anticipation, so
emotional vocabulary alone does not imply a positive label. Two tests check
policy consistency, complete pairs, identical history in topic-change pairs,
unique conversations and absence of unexpanded templates.

Both native tokenizers and the NLI tokenizer find zero exact input overlaps
with protected evaluation, reserved cases, inherited training, or additions
127/130/135. Maximum input length is 117 NLI tokens. These are project-authored
synthetic examples, with no independent annotation or clinical validation.
No neural inference or training is part of preparation; studies131–134 are
unchanged and study136 has no accuracy result.

Studies137–138 queue the use of those two new data sets only if the existing
candidates fail and study134 finishes. Study137 retains the selected study133
weights and complete corpus, then adds all 1,332 new examples per language from
studies135–136. Expected totals are 14,156 Romanian and 14,913 Ukrainian training
rows. All encoder parameters remain trainable with the same layer-wise rates,
paired objective, two epochs and development selection rule as study133; the
new seed is 1137. This continuation tests the added content and conversational
wording without changing the runtime or release thresholds.

Study138 repeats full installed CPU verification and rejects any earlier access
to the reserved test, including by study134. All fifteen shared native gate
tests and five tests for the two new data sets pass. Both study source lists
are frozen before execution. No study137 training or accuracy result is claimed
at queue time, and any earlier qualifying candidate takes priority over this
fallback.

A further audit of the nine prior study128 populations compared native-token
inputs and labels. It found no identical inputs with contradictory labels.
Romanian has 5,153 unique inputs among 5,185 rows; all 32 duplicate groups are
same-label whitespace variants within exposed regression. Ukrainian has 5,258
unique inputs among 5,258 rows. No duplicate crosses the prior evaluation
partitions. This audit does not establish semantic label correctness and does
not change any metric or gate. It performs no model inference and does not read
the reserved test. The script and input-bound result are retained as
`training/audit_evaluation_inputs_v9.py` and
`research/v9-stabilization/evaluation-input-label-audit-20261019.json`.

### Newer multilingual model capacity screen prepared (study139)

The earlier Gemma 3 12B attempt was already limited by local latency. Study139
instead prepares a separate screen of Gemma 4 E4B, whose upstream card describes
4.5B effective parameters and 8B including embeddings, multilingual pretraining,
and an Apache 2.0 license. These are upstream capabilities, not RoGuard accuracy
results. Sources: [Google model card at the conversion's source revision](https://huggingface.co/google/gemma-4-E4B-it/blob/fee6332c1abaafb77f6f9624236c63aa2f1d0187/README.md)
and [MLX conversion](https://huggingface.co/mlx-community/gemma-4-e4b-it-4bit).

The 5.18 GB download is pinned to revision
`475b9088d29754a3379866cf5aeb6b41acd313c2`, with every file hashed and upstream
large-file hashes checked. It remains in the private cache. The conversion's
card still labels its license as Gemma; any future redistribution must preserve
and resolve that metadata difference against the pinned upstream license.
No model assets or package profile are being published by this screen.

Forty already exposed scope-development examples are selected with seed1139:
two per language/scope/label group, twenty per language. The original policy
prompt is unchanged, classification uses fixed threshold 0.5, and an invalid
highest-probability output token counts as an error. At least 18/20 correct in
each language would justify a broader experiment; this small screen can never
qualify a release. The reserved test is not read. Prompt preparation verifies
one-token labels and a maximum length of 467 tokens with no truncation.

Four CPU tests on small instances of the actual Gemma 4 text architecture
verify the final-token projection and cached-prefix path for tied/untied output
weights and enabled/disabled logit caps. The real screen additionally compares
its first two results against full forward passes. It runs in float32 with a
10 GiB allocation limit, after study132 and before study133, with a fifteen-minute
resource timeout. This scheduling change leaves all encoder training sources,
data and protocols unchanged. No study139 inference result is claimed here.

### Completed scope-operator continuation (studies131–132)

Study131 selected epoch0 for Romanian and epoch1.5 for Ukrainian. Romanian did
not improve. Ukrainian development has zero misses, with 79/1,413 formal,
4/223 conversational, 34/90 scope and 26/172 stress false reviews. This lowers
the worst development false-review rate relative to study128, but calibration
does not confirm an overall improvement: formal calibration has zero misses and
258/1,386 false reviews; conversational calibration has one miss among 164 positive
cases and 29/223 false reviews. Exposed regression has one miss among 44 positive
cases and 17/44 false reviews. Accuracy prerequisites fail.

Study132 reproduced all 10,443 prior decisions with a maximum score difference
of 0.0000115634. Reserved-test inference was skipped. The prepared study139
capacity screen runs next, followed by full-encoder adaptation in studies133–134.

### Prompt mismatch diagnosed and corrected (studies139–140)

The study139 screen completed in 19.5 seconds with a peak MLX allocation of
6.26 GB and cached/full score disagreement below 3.4e-7. Its accuracy result is a
failure: all 40 highest-probability tokens were outside the permitted 0/1 labels,
so all 40 were counted as abstentions. The original results remain preserved.

Investigation found that MLX's tokenizer wrapper defaults `enable_thinking` to
its inferred model capability. The plain Hugging Face tokenizer used in the
preflight did not enable it. All 40 actual prompts therefore differed from the
preflight; their maximum length was 469 rather than 467 tokens. Study140 explicitly
sets `enable_thinking=False` for both completed-label templates. A unit test
checks that this overrides the wrapper default without reading the gold label.
An actual tokenizer-only comparison confirms identical Hugging Face and MLX
input tokens for all 40 corrected prompts. Their hashes are frozen and checked
again by the inference script before scoring.

The retry uses the same 40 examples, policy, fixed threshold and strict malformed
output rule. It also records the top token ID for diagnosis. No scoring rule or
release gate is relaxed. It is scheduled after the Romanian study133 child
process exits and before its Ukrainian sibling starts: only the parent
scheduler is paused, and the retry resumes it in a `finally` block. Active
Romanian training continues unchanged. No study140 neural result is claimed
at preparation time; the reserved test remains unread by both screens.

### Frozen contextual classifier prepared (study141)

Study141 tests a different classifier architecture: a small logistic classifier
on the final contextual representation of frozen Gemma 4 E4B. It uses the same
explicit direct-answer prompt, but learns a binary readout from training data.
This measures whether those representations support the policy better than
the model's generated answer. It does not change the scoring or results of
studies139–140. Linear probing is an established method, not a new research claim.

Training samples at most 50 examples per source and label from the study131
corpus plus the prepared study135–136 additions: 1,659 Romanian and 1,948
Ukrainian examples. The full development populations contain 2,638 Romanian
and 2,783 Ukrainian examples. Exact prompt-token audits find no overlap with
protected evaluation inputs, including all 600 reserved examples. Reserved
examples are used only for exclusion and never enter the representation input
artifact. All examples remain synthetic and project-authored.

The model and tokenizer are pinned, prompt hashes are checked again at runtime,
and no input is truncated; the maximum prepared length is 542 tokens. Each
representation is normalized independently. Five predeclared regularization
settings are fitted per language, with selection and thresholds using development
data only under the unchanged 99% recall requirement. Calibration and exposed
regression inference are deferred until development passes. A separate installed
runtime verification and the reserved test would still be required for release.

Two focused tests pass: cached versus complete features on a small instance of
the actual Gemma 4 architecture, and portable JSON readout scores versus the
fitted classifier. The real experiment also checks cached/full features on its
first two inputs. Feature chunks are resumable and bound to input, source and
weight hashes. They remain private local caches. No study141 accuracy result is
claimed at preparation time.

The idle study140 scheduler is replaced with one that runs study140 and then
study141 after Romanian study133 finishes. Its `finally` block resumes the
paused study133 parent so Ukrainian training can proceed. Active training is
not interrupted, and all GPU jobs remain serialized.

### Completed Romanian full adaptation and corrected prompt screen

The Romanian half of study133 completed both epochs and retained epoch0: none
of the fully updated checkpoints improved the development selection objective.
Its calibration therefore remains at zero misses, with 244/1,386 formal and
37/243 conversational false reviews; exposed regression has 8/44 false reviews.
Ukrainian study133 and installed verification134 are still pending at this update.

Study140 completed with correct prepared/runtime prompt tokens. It scored
15/20 correct in each language. Romanian has one missed positive and four false
reviews, including four abstentions; Ukrainian has two missed positives and
three false reviews, including two abstentions. The six invalid top tokens begin
JSON objects rather than the requested digit. They remain errors. The screen
does not qualify for expansion under its predeclared 18/20 rule.

Cached/full score disagreement is below 1.4e-8; the model stage took 18.4 seconds
with peak MLX allocation 6.25 GB. These are a local forty-case screen's measurements,
not production performance claims. Study141 proceeds as the separately declared
learned-classifier experiment. No reserved example has been inferred.

### Conditional nonlinear classifier prepared (study142)

Study142 reuses study141's verified contextual representations if the linear
classifiers fail development. It fits radial-basis support-vector classifiers
with a predeclared twelve-setting grid per language. This adds a nonlinear
decision boundary without another foundation-model inference pass or new data.
The selected linear classifier is retained unless a nonlinear candidate strictly
improves the same development objective. The 99% recall and 5% false-review
requirements are unchanged; calibration and reserved tests remain separate.

The portable scorer exports numeric support vectors and coefficients, avoiding
executable model serialization. Its test matches the fitted classifier on an
unseen nonlinear-boundary sample; each real candidate must also match full
development margins within 1e-7. Scores are bounded margins, not calibrated
probabilities. Source, feature-chunk and input hashes are verified before fitting.
This CPU-only fallback may run alongside the later Ukrainian encoder training.
No study142 accuracy result is claimed at preparation time.

### Inference batching prototype

`training/gemma4_batch_features_v9.py` groups continuations by exact token length
and clones the common prompt cache for each group. It returns features in the
original input order. Two tests on small actual Gemma 4 models cover independent
and shared key/value layers, sliding attention, unequal lengths and cache
isolation. Both pass at the original absolute feature tolerance of 1e-5.

An initial right-padding implementation exceeded that tolerance for a shared-KV
test: its maximum difference from independent full inference was 1.19e-5.
Grouping by exact length preserves the one-token attention path and fixes the
test without changing its tolerance. The prototype is not used by the frozen
study141 run. It has no real-model throughput result or deployment qualification;
actual score and decision parity must be checked before enabling it.

### Completed frozen contextual linear classifiers (study141)

All 9,028 representations were extracted, with a maximum cached/full relative
feature difference of 1.97e-5 and peak MLX allocation of 6.08 GB. Extraction,
validation and fitting took 2,899 seconds locally. Portable logistic scores match
the fitted classifiers within 1.12e-16 on every development example.

The selected Romanian classifier uses C=0.001. It misses five formal positives
and has 618/1,413 formal, 48/242 conversational and 26/90 scope false reviews.
The selected Ukrainian classifier uses C=0.01. It misses five formal positives
and one conversational positive, with 349/1,413 formal, 39/223 conversational,
29/90 scope and 52/172 stress false reviews. Both satisfy the development recall
constraint but fail false-review requirements. Neither qualifies for remaining
release verification. Calibration and reserved-test inference were not performed.

Read-only interim Romanian fits of the same predeclared linear and kernel grids
ran while Ukrainian representation extraction finished. They introduce no new
candidate settings or evaluation population. A separate diagnostic reconstructed
the pretrained 0/1 label scores from the cached features and two decoded output
rows; it matched all twenty Romanian study140 scores within 8.35e-7. Those
normalized label scores also fail the development gates. This diagnostic does
not reinterpret malformed generated answers as successful study140 predictions.

### Shared bilingual semantic classifier prepared (study143)

Study143 tests whether the existing six semantic labels provide useful additional
supervision. One shared 128-unit GELU bottleneck learns subject, time, assertion,
source, relation and concern, alongside a binary review head, from the same
3,607 training representations. Language/label weights balance the four training
strata. Training uses 100 CPU epochs, seed1143, AdamW at 0.001, weight decay 0.01
and gradient clipping. It evaluates epoch0 and the predeclared epochs10,20,40,60,100.

Three score paths are compared on development: the binary head, the existing
factorized policy calculation and that calculation with training-fitted pair
couplings. The latter encode subject/source and assertion/relation associations
already supported by `native_fields_v9`; their values use training labels only.
Scores and semantic interpretations remain model proposals, not verified facts
or calibrated safety probabilities. Each language retains its study142 selection
unless a candidate strictly improves the unchanged development objective.

The independent numeric GELU export test passes at 1e-6. Every evaluated real
checkpoint must also match all development logits within 1e-4. All evaluated
tensor checkpoints are retained privately and bound by hashes. The study adds
no foundation-model inference, calibration access or reserved-test inference.
It is queued after study142 and skips if an earlier contextual classifier passes
development. Ukrainian encoder training can continue on the GPU independently.

### Completed nonlinear and semantic-head comparisons (studies142–143)

Study142 retained the linear Romanian classifier. Its selected Ukrainian RBF
classifier has five formal misses and 368/1,413 formal, 41/223 conversational,
27/90 scope and 45/172 stress false reviews. The worst false-review rate is
30%; it still fails development. All twenty-four kernel export comparisons
passed their margin-parity checks. Fitting and evaluation took 98.3 seconds.

Study143 completed in 12.4 seconds. Romanian selected the coupled-policy score
at epoch10: five formal and two conversational misses, with 396/1,413 formal,
18/242 conversational and 20/90 scope false reviews. Ukrainian selected the
factorized-policy score at epoch20: five formal misses, with 255/1,413 formal,
32/223 conversational, 21/90 scope and 25/172 stress false reviews. Each group
still meets the recall constraint, but false-review gates remain unmet.

These are improvements within the contextual-readout experiments, not release
qualification or evidence of real-world accuracy. No calibration or reserved
examples were inferred. The next targeted data question is whether the limited
sampling of formal and indirect-distress training examples constrains these
semantic heads; only 100 examples from each of those sources per language were
included in study141's initial sample.

### Expanded contextual training prepared (studies144–145)

Study144 adds the remaining formal-composition and indirect-distress training
inputs: 2,179 Romanian and 2,167 Ukrainian examples. Together with study141's
training sample, this produces 7,953 training examples. The 5,421 development
examples and all release requirements stay fixed. Exact prompt-token audits
found no overlap with protected evaluation inputs or existing representations.
Reserved inputs were used only for overlap exclusion, never model inference.

The extractor first compares batches of four and one against 100 existing
training representations. Every relative feature difference must be at most
1e-4. Batches of four are selected only if they also run faster; otherwise the
verified single-example path is used. This is a training-extraction comparison,
not a deployment batching qualification. Features remain private, while input,
source and chunk hashes are recorded for reproducibility.

Study145 repeats study143's architecture, seed, optimizer and checkpoint schedule
on the expanded training set. Semantic couplings use the expanded training labels
only. Each language retains its study143 classifier unless the same development
objective strictly improves. Both studies are queued after the full native
encoder and its installed verification, before the next conversational encoder.
No expanded-training accuracy result is claimed at preparation time.

### Contextual complementarity and expanded scope gate (studies146–147)

Study146 compares the fixed study131 encoder and study143 semantic classifier
using probability and logit blends at five predeclared weights. Inputs, labels
and development groups are checked for identity before combining predictions.
Neither language passes every development gate. The selected Ukrainian blend
misses no positives and has 56/1,413 formal, 9/223 conversational, 11/90 scope
and zero stress false reviews. Romanian still fails formal and scope checks.
Five numerical tests cover the blend calculation and invalid scores.

Study147 revisits the quotation-abstracted scope classifier from study10 with
the expanded training corpus: 14,156 Romanian and 14,913 Ukrainian inputs.
It learns applicability separately from concern content, preserving text outside
closed quotations. Three regularization settings and three powers are compared
with each fixed baseline on development. Full normalized training inputs have
zero protected overlap. Shared abstracted scope frames are counted explicitly;
their reuse does not establish generalization to new phrasing. No hand-written
decision rule is added for the observed development errors.

The selected Ukrainian candidate passes all four development groups: four
formal misses and 24/1,413 formal false reviews; no conversational misses and
5/223 false reviews; no scope misses and 4/90 false reviews; no stress misses
or false reviews. Its gate uses C=0.1 and power2 with the study146 blend.
Independent numeric reconstruction matches every development score exactly.
Romanian still fails, with two formal misses and 264/1,413 formal, 4/242
conversational and 11/90 scope false reviews. Calibration, installed execution
and the reserved population have not qualified these candidates for release.

### Frozen Ukrainian composite qualification prepared (study148)

Study148 checks study147's selected Ukrainian candidate on the separate
calibration and exposed regression populations, with its weights and threshold
fixed beforehand. The reserved 600 examples are absent from its inference
inputs. Both languages had zero exact development/training collisions even
after quotation abstraction in study147.

The composite numeric scorer first reproduced all 2,783 Ukrainian development
decisions. Its maximum score difference was 4.32e-6, below the frozen 1e-5
tolerance. This verifies reconstruction from the selected semantic tensors,
native scores, blend and sparse scope gate; it is not installed end-to-end
parity. The separate contextual features and prior accuracy checks are queued
after studies144–145. Fresh-test access still requires both languages to pass
all preceding accuracy and installation checks.

### Joint checkpoint selection prepared (study149)

Study149 addresses a limitation of selecting each classifier in isolation. It
compares the existing semantic checkpoints from studies143 and145 by the errors
of their composition with the native encoder and scope gate. The language-specific
blend weights from study146 and the scope gate and power from study147 stay fixed.
Only the already evaluated epochs10,20,40,60,100 and three semantic score paths
are considered. Native studies131 and133 are compared, with identical tensor
checkpoints deduplicated. Study147 is retained unless the unchanged development
objective strictly improves.

Completed parent reports, predictions and tensor hashes are recorded before
comparison. Couplings are reconstructed from each classifier's training labels
with its original language/label weights. This CPU comparison adds no foundation
inference or training data. It is queued after study145 and before study148's
calibration inference, so calibration results cannot guide its selection.

### Portable sparse scope scoring

`training/portable_scope_gate_v9.py` reconstructs the exported sparse gate using
the Python standard library. It applies the frozen character/word n-grams,
sublinear counts, IDF and per-block normalization without loading scikit-learn
or executable model objects. Six tests cover multilingual held-out strings,
empty and repeated text, quotation styles and malformed numeric artifacts.

The actual selected gates reproduced all 5,421 development decisions, with
maximum score differences below 5e-16. On this machine the scope component took
approximately 0.15–0.16 milliseconds per case with native/contextual scores
already available. This timing excludes those encoders and is not an end-to-end
latency claim. Numerical CI now runs the sparse export and blend tests with
their required reference dependencies explicitly installed.

### Completed full encoder and installed replay (studies133–134)

The Ukrainian full encoder selected epoch1: development has zero misses, with
42/1,413 formal, 2/223 conversational, 25/90 scope and 35/172 stress false
reviews. Calibration has zero formal misses with 238/1,386 false reviews, and
one conversational miss with 16/223 false reviews. The exposed scope regression
has one miss and 6/44 false reviews. Romanian retained its initial checkpoint.
Both languages still fail accuracy requirements.

Study134 reproduced all 10,443 prior decisions using the installed CPU runtime.
The largest score difference was 7.46e-6. Installation parity passed, but the
reserved population was skipped because preceding accuracy gates failed.

### Complete-content and attribution views prepared (study150)

Study150 gives the native encoder the complete conversation in its first input
and the focal message with closed quotations abstracted in its second. Every
quoted word remains available in the complete view; negation and attribution
remain visible around the quote marker in the second. The transform makes no
review decision. Four tests cover content preservation, later-turn exclusion,
unclosed quotations, Ukrainian attribution and Romanian normalization.

This fallback starts from the pinned original pretrained encoders with a new
binary head, using the same expanded 14,156 Romanian and 14,913 Ukrainian
training examples. Exact two-view token audits found zero protected overlap,
no duplicate training inputs and maximum lengths of 127/133 tokens. Reserved
examples were tokenized only for exclusion. Full encoder training uses two
epochs, seed1150, top learning rate2e-5 and layer decay0.9, with the existing
paired loss and unchanged development selection requirements. Initialization,
learning rate and input view all change, so this is not an isolated view ablation.

It is queued after studies137–138 and skips when an earlier candidate should
advance to release qualification. The new preprocessing has its own artifact
format and requires an installed runtime check before any reserved inference.
There is no study150 accuracy result at preparation time.

### Conservative model-call bounds

The logit blend is monotone in the contextual score, and its result is multiplied
by the sparse scope factor. `training/contextual_bounds_v9.py` computes the
possible output interval before requesting the contextual model. A decision is
returned early only when the entire interval lies on one side of the frozen
cutoff, with a numerical margin. Otherwise the full path remains required.
These bounds preserve a classifier's decision; they do not establish that the
decision is correct.

Seven tests cover extreme and random component scores, cutoff boundaries and
invalid values. Replay against study147 avoided 1,111/2,638 Romanian contextual
calls (42.1%) and 563/2,783 Ukrainian calls (20.2%), changing no decisions. The
Ukrainian exits also avoid the native encoder. These are cached development call
counts, not end-to-end latency measurements or deployment qualification. No
training, calibration inference or reserved-test access was added.

### Direct policy supervision prepared (study151)

Study151 keeps study145's data, initialization, architecture and optimizer. It
adds binary cross-entropy on the existing integrated policy probability, averaged
over its independent and coupled forms, to the binary-head and six-field losses.
This directly trains the composed review decision while retaining semantic-field
supervision. Training uses the existing policy, not a changed routing rule.

Two gradient tests compare the differentiable calculation with the independent
numeric scorer to 1e-12, verify finite gradients in all policy-relevant fields
and confirm that a gradient step reduces the policy loss. The time field remains
trained by its field loss; it does not independently change this existing policy.
The CPU-only experiment waits for study149 and skips if a preceding contextual
candidate qualifies development. Its baseline is study145, and release gates,
calibration separation and reserved-test handling remain unchanged.

### Completed expanded and policy-supervised contextual fits

Study144 extracted all 4,346 additional training representations in 1,208 seconds,
with 6.23 GB peak MLX allocation. Batches of four matched the 100 reference
representations within 3.97e-5 relative error and took 34.6 seconds versus 39.1
seconds individually. Study145 retained the Romanian study143 head. Its selected
Ukrainian epoch10 coupled-policy head reduced the worst development false-review
rate to 18.9%, but still failed. Fitting took 18.3 seconds.

Study149 retained the existing Romanian composition. Ukrainian selected the
expanded epoch10 factorized-policy head with native131 and the unchanged scope
gate: five formal misses, with 11/1,413 formal, 4/223 conversational, 2/90 scope
and zero stress false reviews. Every group passes development. Study152 freezes
this composition for separate prior qualification using study148's feature
extraction. Its numeric replay reproduced all 2,783 development scores exactly.

Study151 completed in 27.9 seconds. Romanian retained its prior standalone head.
Ukrainian selected epoch10 coupled-policy: three formal and one conversational
miss, with 99/1,413 formal, 20/223 conversational, 7/90 scope and 13/172 stress
false reviews. This improves standalone development behavior but does not pass
every gate. Neither study145 nor151 accessed calibration or reserved inference.

### Entailment scope and native agreement (studies153–154)

Study153 combines the frozen study123 entailment scores with study147's sparse
scope gate. It compares the maximum reason score and a bounded ratio that
preserves the original two-reason decision, with scope powers0,1,2,4. Selection
uses development only. The selected maximum-score path uses power4 for Romanian
and power2 for Ukrainian. Both still fail formal and scope false-review gates,
so neither language is scored on calibration in this study.

Study154 requires both the selected native131 decision and the selected153
entailment-scope decision. Component cutoffs stay fixed; the bounded minimum of
their score/cutoff ratios exactly represents this conjunction. Separate numerical
tests verify both the OR-ratio and AND-ratio transformations, including boundaries.

Romanian passes development: zero formal misses with 10/1,413 false reviews;
two conversational misses with 2/242 false reviews; zero scope misses with
4/90 false reviews. With the composition frozen, calibration has zero misses,
but 95/1,386 formal false reviews exceed the 5% limit. Conversational calibration
has 10/243 false reviews and passes. The exposed regression has no misses or
false reviews across 88 cases. Ukrainian still fails development scope, so its
calibration is skipped. Reserved inference remains closed.

### Separate Ukrainian qualification completed (studies148 and152)

Study148 extracted all 2,475 Ukrainian calibration and exposed-regression
representations in 821 seconds, with 6.24 GB peak MLX allocation. The frozen147
composition misses 32/614 formal positives, with 6/1,386 formal false reviews.
Conversational calibration passes with one miss and five false reviews, but the
exposed regression has 10/44 false reviews. Study152's frozen149 composition
also misses 32 formal positives, with eight formal false reviews. Conversational
calibration has no misses and four false reviews; exposed regression has no
misses and eight false reviews. Neither candidate qualifies. The discrepancy
between development and calibration remains visible; reserved inference stays
closed. Native study137 training has resumed.

### Conditional contextual check (studies155–156)

Study155 keeps study154's Romanian first-stage decision and compares the already
trained143,145 and151 semantic heads only where that stage requests review.
Checkpoints, binary/policy modes and the final threshold are selected exclusively
on development with the same per-group recall constraint. The selected143
epoch20 factorized-policy head has one formal and two conversational misses,
10/1,413 formal false reviews, and no conversational or scope false reviews.
All development groups pass. The retained baseline remains available.

Study156 freezes this cascade for prior qualification. A separate implementation
reproduces all2,638 development decisions and scores exactly while evaluating the
contextual head on907 cases. Its prior input manifest contains979 of2,547 cases;
1,568 first-stage exits require no contextual inference. This is a reduction in
model calls, not a measured end-to-end latency improvement. The prior extraction
waits for active Romanian training to finish; no reserved cases are included.

### Ukrainian entailment-first cascade (studies157–158)

Study157 replaces the first-stage native blend with the frozen153 entailment
scope decision, then selects among the same existing semantic heads on
development. The expanded145 epoch10 coupled-policy head passes all development
groups: five formal and one conversational miss, with26 formal,4 conversational,
2 scope and2 stress false reviews. Numeric replay in158 reproduces all2,783
scores and decisions exactly, evaluating the contextual head on997 cases.

Separate qualification passes formal calibration (3/614 misses,23/1,386 false
reviews) and the exposed regression (no misses,2/44 false reviews). Conversational
calibration still fails recall:7/164 misses,10/223 false reviews. All seven are
stopped by the entailment first stage; they involve a transition from a fictional
exercise to a personal concern. This diagnosis informs subsequent experiments;
these outcomes are not used to choose a new cutoff. The cascade remains
unqualified, and the reserved test set has not been inferred.

### Recall-first cascade and context consistency (studies159–162)

Study159 sets the entailment first-stage cutoff to the minimum positive
development score, then selects its contextual check with the unchanged99/5
frontier. The policy-supervised151 epoch10 coupled head passes development,
including one scope false review and no scope misses. Separate160 qualification
passes formal calibration (two misses, nine false reviews) and exposed regression
(no errors), but conversational recall still fails: eight misses and seven false
reviews. Its development replay has no score or decision differences.

Study161 adds1,728 synthetic training cases and288 development cases across the
two languages. Current-message attribution is paired between personal and
explicitly fictional versions; each is repeated under three earlier topics.
Ordinary-message controls remain negative. Development uses different message
bodies and framing. Two tests verify the intended policy labels, complete
contrasts, identical focal messages across histories and disjoint wording.
Token checks find no overlap with the protected development, calibration,
exposed or reserved populations. These are project-authored examples.

Study162 prepares a Ukrainian-only continuation of the123 entailment encoder.
Its6,292 training rows include4,096 stratified replay examples,1,332 existing
indirect/freeform examples and864 new transition examples. The last four encoder
blocks, pooler and classifier are trained with balanced reason cross-entropy,
paired margins and a symmetric KL penalty for identical current messages under
different histories. Whole matched-history families stay together in batches.
The144 held-out Ukrainian transitions form an additional development gate;
calibration remains separate. Inherited training also has zero token overlap
with this new development population.

Two additional tests verify minibatch coverage and the consistency loss with
finite gradients. A preflight covers every one of12,584 training queries exactly
once, in1,121 batches of at most12 queries, with644 paired and1,728 invariance
comparisons. This research artifact is explicitly Ukrainian-only and requires
production support and installed qualification before any release. Training is
queued after156 qualification; ongoing Romanian training is uninterrupted.

### Shared-attention tail adaptation preflight (study163)

The pinned Gemma4 text model has a final18-layer block that reuses key/value
states from earlier frozen layers. For a last-token classification loss, those
shared states allow the adaptable tail to recompute only the final query at
each layer. The captured state includes complete attention keys and values,
the final-token hidden state, per-layer inputs, positions and attention masks.
It does not shorten the conversation. Only parameters after the verified
shared-KV boundary may change.

Five small-model CPU tests check output equivalence before and after tail
weight changes, gradient equivalence, sliding-window behavior, multiple batch
members and rejection of a boundary containing its own key/value projections.
These tests pass; they do not establish performance on the actual4B weights.

The frozen163 preflight uses16 existing training inputs, balanced by language
and label and spanning prompt lengths. It will compare boundaries24 and38,
measure replay time and cache size, and test one rank4 query/output adapter
update on the full shared tail. Full-sequence parity must still hold after
the update. The update is discarded; this preflight makes no accuracy claim
and accesses no development, calibration or reserved examples.

A separate cache helper stores only typed arrays and bounded JSON metadata,
with input-token and source-freeze identities. Loading rejects mismatched
identities, nonfinite states, unexpected arrays and oversized archives; it
never uses pickle. Equal-length states can be batched without padding or
changing attention positions. Two CPU tests pass for storage round trips,
identity/corruption rejection and batched full-sequence parity. The numerical
CI job also runs the NumPy cache validation test.

### Romanian conversational continuation completed

Study137's Romanian training selected the half-epoch checkpoint on development.
It still fails false-review gates:261/1,413 formal,16/242 conversational and23/90
scope false reviews, with no misses. Separate calibration has no misses but
268/1,386 formal and36/243 conversational false reviews; the exposed regression
has10/44 false reviews. All checkpoints and the completed evidence are retained.
This result does not replace the better fixed154/155 Romanian cascades.
