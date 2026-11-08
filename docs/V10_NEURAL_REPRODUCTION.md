# V10 neural research reproduction

These are failed or unqualified research experiments, not a published v10 model.
The public study directories retain protocols, source snapshots, selected
development outputs and exposed regression results. The large tensor caches and
newly trained adapters stay outside Git. Their absence does not turn a report
into an independent reproduction.

The experiments use the published v9 adapters, the pinned Gemma4 base, public
synthetic inputs, and the existing exact shared-attention cache format. They do
not require access to the older private research adapter history. The neural
backend requires Apple Silicon and the dependency versions pinned by the v9
runtime. Run commands from this repository with its `src` directory on the
Python path.

## Obtain the published starting model

Use an isolated environment with the `gemma4` and `trajectory-train` extras.
Download the unchanged public profile explicitly:

```sh
roguard-fetch fetch --profile v9-gemma4-20261019
PYTHONPATH=src python -m training.stage_public_v9_v10 --output /your/workspace/v9-start
```

Staging verifies the immutable public profile and research adapter digests. It
copies the two small adapters, writes the matching research metadata, and creates
a local link to the verified base under its revision name. It does not download
anything or copy the entire base. Use the printed `candidate` and `base` paths in
the training commands. Keep this workspace outside Git.

The staging path was exercised against a checksum-verified public-profile cache:
existing matching base/adapter files were reused locally and the small pinned
release metadata was fetched. Both staged adapter digests matched the public
release. This checks staging, not a complete clean-machine retraining.

## Rebuild the original frozen-body caches

The four source corpora contain 22,982 training/development inputs. Their source
freezes reference public files. To prepare clean cache directories without
copying the old private tensors:

```sh
PYTHONPATH=src python -m training.prepare_public_tail_cache_v10 \
  --output /your/workspace/corpora
```

Adding `--base /your/workspace/v9-start/475b9088d29754a3379866cf5aeb6b41acd313c2`
runs the existing pinned extractor sequentially for all four corpora. This is a
large local computation: the existing complete caches occupy roughly 190 GB,
in addition to the model, features and temporary files. The metadata preparation
path was exercised separately; the existing verified caches were reused for the
reported v10 experiments, rather than rebuilding that entire archive again.

`extract_semantic_readout_v10`, `cache_semantic_development_v10`, and
`train_semantic_tail_v10` accept `--cache-root /your/workspace/corpora`. Their
default remains the original local research cache location. The corpus reader
checks input, source, chunk and tensor identities before reuse.

## Experiment sequence

| Study | Entry point | Inputs and decision |
|---|---|---|
| Semantic features 11 | `training.extract_semantic_readout_v10` | Replays the published adapters on original TRAIN/DEV caches; exports private final-token features and logits. |
| Semantic readout 11 | `training.train_semantic_readout_v10` | Three registered head fits, 12 epochs each, development-only selection. |
| Semantic regression 11 | `training.evaluate_semantic_readout_v10` | Runs 6,002 exposed checkpoints and verifies original v9 score parity; creates the frozen-encoder regression cache. |
| Training expansion 12 | `training.prepare_semantic_augmentation_v10` | Composes training-only wording, adds existing trajectory TRAIN views, and audits exact overlap with recorded evaluation inputs. |
| Features/readout 12 | `training.extract_semantic_augmentation_v10`, then `training.train_semantic_readout_v10` | Adds 2,561 training prefixes; use `--amendment research/v10-evidence/protocol-amendment-09.json`. |
| Consensus 13 | `training.train_semantic_consensus_v10` | Fits nonnegative combinations of neural score modes on TRAIN and selects on DEV. |
| Attention 14 | `training.cache_semantic_tail_v10`, `training.cache_semantic_development_v10`, `training.train_semantic_tail_v10` | Replays all 2,561 new training states and 327 eligible development states exactly, then trains attention adapters and active-language heads. |
| Frozen score comparator 15 | `training.select_frozen_score_baseline_v10` | Reselects an existing v9 score mode on the same development data; no weight update. |
| Semantic probes 16/17 | `training.train_semantic_sparse_v10` | Separate semantic-only lexical representations. They do not change review scores. |
| Base-reference 18 | `training.extract_base_reference_v10`, then `training.train_base_reference_v10` | Replay published-v9 and zero-adapter base views. Four classifier designs, 40 epochs, development-only selection. The failed initial extractor attempt is retained separately from completed `base-reference-features-18b`. |
| Language specialists 19 | `training.select_language_specialists_v10`, then `training.evaluate_base_reference_v10` | Select per language on matching development IDs; fresh base-view inference for selected reference heads, bound attention-result reuse for selected attention models. |
| Scope curriculum 20 | `training.scope_curriculum_v10`, `training.audit_scope_inputs_v10`, `training.extract_scope_features_v10` | Author and screen 3,456 new TRAIN views; append paired features without changing development. Use `training.train_base_reference_v10 --curriculum` for the head fit. |
| Explicit assertion probes 22/23 | `training.probe_assertion_query_v10`, `training.assess_assertion_queries_v10` | Fixed prompts and balanced original DEV only; compare Gemma and the separately pinned Qwen model against exact v9 assertion logits. All prompts introduce new errors. |
| Semantic-query representations 24 | `training.extract_query_features_v10`, then `training.train_base_reference_v10 --query` | All original 16,641 TRAIN and 6,341 DEV inputs; pair verified v9 features with fixed semantic-query base features. Query regression uses `training.evaluate_base_reference_v10 --query`, which rejects old paired-view caches. |
| Response curriculum 25 | `training.response_curriculum_v10`, then `training.train_response_readout_v10` | Synthetic response-behavior wording with separate TRAIN/DEV families; four lexical fits; selection frozen before original diagnostic scoring. Rejected. |
| Neural response comparison 26 | `training.train_neural_response_v10` | Registered semantic encoder comparison using exactly the response-25 split, separate heads and no changes to the published review model. |
| Semantic query 24 | `training.extract_query_features_v10`, then `training.train_base_reference_v10 --query` | Extracts 22,982 original TRAIN/DEV representations under the fixed explicit assertion query. Both selected full heads retain zero binary DEV errors but leave 643 RO / 691 UK joint semantic errors. |
| Neural response 26 | `training.train_neural_response_v10` | Fixed 340 TRAIN / 80 DEV response inputs; zero-adapter semantic prompt and separate readout. Selected head gets 77/80 DEV and 10/10 original diagnostics; rejected. |
| Review agreement 27 / unanimity 28 | `training.review_agreement_v10`, then `eval.assess_review_unanimity_v10` | The two-model rule fails DEV. `--unanimous` uses the separately registered all-three-positive rule; it passes the recorded binary engineering populations and is independently recomputed from bound records. |
| Installed committee runtime 29 | `eval.run_committee_v10 --installed-wheel` | Registered fresh inference on all 6,002 checkpoints. Verifies installed package bytes, component scores and decisions, input identities and unchanged thresholds. |
| Assertion specialist 30 | `training.train_assertion_specialist_v10` | Six class-balanced assertion-only heads per language. DEV errors fall from 584 to 433 RO and 578 to 315 UK; rejected. |
| Native answer diagnostic 31 | `training.native_answer_diagnostic_v10` | Projects cached semantic features through the pinned pretrained answer-letter rows. Verifies exact dequantization and all 256 prior probe decisions; direct answer scores still make substantial errors. |
| Contrast response 32 | `training.train_contrast_response_v10` | Registered new matched TRAIN pairs (596 total TRAIN inputs), unchanged 80 DEV inputs and prompt, verified feature reuse, and separate class-balanced response heads. |
| Scope query 33 | `training.extract_scope_query_v10`, then `training.train_assertion_specialist_v10 --scope` | Registered fresh query representations for the existing 3,456 screened scope TRAIN inputs; original DEV remains fixed. |
| Response context 34 | `eval.evaluate_response_context_v10` | Freeze heads before scoring 120 unchanged drafts behind new distracting contexts. Study26 gets 101/120; study32 gets 95/120. Neither qualifies. |
| Regularized response 35 / isolated response 36 | `training.train_regularized_response_v10` (add `--draft-only` for 36) | Converged L2 fits on verified frozen features. Both select heads with 70/80 original DEV accuracy; context results are 100/120 and 105/120 respectively. Both rejected. |
| Qwen response 37 | `training.train_qwen_response_v10` | Registered comparison of the pinned Qwen encoder on the same 164 unique TRAIN and 40 unique DEV drafts. Native answer scores and six fixed regularization settings; freeze before the ten old diagnostics. |

| Committee cost 38 | `eval.benchmark_committee_v10` | Same loaded encoder, fixed length-stratified inputs, warm randomized timings; about 12% more than v9 and 63% less than separate passes on this host. |
| Probability floor 39 | `eval.audit_probability_floor_v10` | Checks direct probability cutoffs against cancellation in one-minus-probability. Existing recorded prediction sets remain unchanged; no accuracy gain claimed. |
| Evidence responses 40/41/43 | `eval.evaluate_evidence_response_v10` | Strict generated-output contracts; `--grounded` uses bound source IDs, and `--contrast-examples` selects fixed matched TRAIN demonstrations. No response model qualifies yet. |
| Committee trajectory 42 | `eval.verify_committee_trajectory_v10` | Installed wheel check, 6,002 wrapper replays and 21 fresh neural inferences. Full output parity; diagnostic semantic probabilities remain uncalibrated. |
| Trajectory archive 42b | `scripts/archive_committee_research_v10.py --trajectory-protocol ... --selector ...` | Packages the exact verified wheel, bundle and selector; checks all archived bytes. Unpublished engineering artifact only. |

| Fresh committee challenge 44 | `eval.challenge_committee_v10 author`, then `evaluate`; `eval.audit_fresh_committee_data_v10` | Freeze 72 authored checkpoints and screen enumerated historical data before scoring the unchanged relocated archive. Seven errors shared with v9; rejected. |
| Translated response 45 | `eval.evaluate_translated_response_v10` | Translation-only stage followed by separate boolean behavior judgments. Same40 DEV drafts;35/40 valid correct, rejected. No fitting or output repair. |

| Robust semantic readout 46 | `training.prepare_robust_augmentation_v10`, `training.extract_semantic_augmentation_v10`, then `training.train_semantic_readout_v10 --amendment research/v10-evidence/protocol-amendment-44.json` | Registered 1,204 additional TRAIN rows: balanced interfering assistant instructions plus separate bot-use contrast wording. Existing DEV selection and protected regressions remain fixed. Post-selection evaluation uses `training.evaluate_cached_semantic_v10` and `eval.evaluate_readout_challenge_v10`; the latter checks exact fresh v9 score/logit parity on all 72 fresh44 checkpoints. |

| Robust attention 47 | `training.prepare_robust_augmentation_v10 --exclude-protected-parents --amendment ...45.json`, then feature extraction, `training.cache_semantic_tail_v10`, and `training.train_semantic_tail_v10 --amendment ...45.json` | Registered final 18-layer q/o LoRA plus active-head update, with corrected augmentation-parent screening. Original plus eligible trajectory DEV selection; new attention needs fresh neural regression. Fresh44 comparison uses `eval.evaluate_readout_challenge_v10 --component attention --amendment ...45.json`. |

| Unchanged selection assessment 47 | `eval.assess_unchanged_attention_v10` | Checks all 80 tensors per language byte for byte and identical runtime settings, despite different serialized file hashes. Fresh44 outputs also match v9 exactly. The redundant full-baseline rerun was stopped; partial files remain marked incomplete. |
| Response attention 48 | `training.train_response_attention_v10` | Registered six-epoch shared bilingual response adapter: final 18 q/o LoRA layers plus the study26 linear head, same 596 TRAIN/80 DEV split. Frozen-body states are streamed in minibatches. Small-Gemma forward/gradient/frozen-parameter checks and input-binding preflight pass before full training. |

Each command's `--help` lists its required paths. Use new output directories;
commands refuse to overwrite studies. Historic `source.py` snapshots preserve
the implementation used for each recorded result, even as the working scripts
gain additional options.

`training.evaluate_cached_semantic_v10` accepts only the original frozen-v9
encoder cache and verifies that every non-head tensor is unchanged. Its replay
of readout 11 matched all 6,002 original predictions, with a maximum score
difference of zero. An attention update must use
`training.evaluate_semantic_tail_v10`, which performs new neural inference.
Changing a cache's name does not make it valid for a different encoder.

`eval.assess_semantic_candidate_v10` independently recomputes the attention
candidate's reports and paired error IDs, and checks every saved score and
semantic argmax against its logits. For the recorded attention run:

```sh
PYTHONPATH=src python -m eval.assess_semantic_candidate_v10 \
  --study research/v10-evidence/semantic-tail-regression-14 \
  --selected-study research/v10-evidence/semantic-tail-14 \
  --reference-study research/v10-evidence/semantic-regression-11 \
  --output /your/workspace/assessment-14
```

This needs NumPy but no neural weights or accelerator inference. It records
`REJECT`: two broad binary errors (including one new regression), remaining
semantic errors, and missing independent qualification. It performs no fitting.

All these populations have been exposed during development. The exact-overlap
audit found 21 trajectory development inputs and 44 exploratory calibration
inputs in earlier v9 fine-tuning. The 21 development overlaps are excluded from
attention-14 selection. Fresh calibration and independently controlled final
evaluation remain separate requirements. See the
[study handoff](V10_INDEPENDENT_STUDY.md) for that process.

## Larger generated-data studies 50–57

Natural-language generators50–53 each completed a16-group TRAIN pilot. They
were rejected for scaling after wording or label defects; none of their
structurally accepted text was silently promoted into training supervision.
Every attempt remains in the private generation artifacts, with public hashes
and assessments. The cross-model screening utility is available, but no model
judge result is claimed for these rejected pilots.

Composition54 uses controlled response contrasts. Its12,288 TRAIN inputs come
from192 core sentences in48 paired frames, expanded across topics and contexts.
There are512 DEV and1,536 reserved TEST inputs. All translations and variants of
a frame stay in one partition; topics are also disjoint. The source definitions
are public developer-authored test design, not independently authored tests.

Response55 adds the596 historical TRAIN inputs for12,884 total, compares three
registered head configurations over80 epochs, and selects on80 historical plus
512 generated DEV inputs. Origin/language/category weighting prevents expanded
examples from overwhelming the historical training population. The selected
checkpoint is frozen before context34 and original-diagnostic evaluation. The
reserved test opens only after every preceding population has zero candidate
errors and zero introduced errors. Feature extraction can resume using bound,
hashed chunks; interrupted optimizer runs cannot overwrite checkpoints.

Response55 completed all243 checkpoints and was rejected. The registered
selection retained the original head (warm-linear epoch0):39/592 DEV errors,
19/120 context errors and0/10 original-diagnostic errors. Some checkpoints
reached13 total DEV errors but worsened the maximum group error rate. None
was error-free. The1,536 reserved TEST inputs remain unopened.

Composition56 supplies4,608 binary TRAIN inputs from64 core sentences in16
paired frames, plus576 DEV and1,728 reserved TEST inputs. Eight concern scopes
cover direct statements, explicit denial, fiction, reported speech, uncertainty,
hypotheticals, past events and retractions. Ordinary functioning is not negated
and automatically labeled benign. Labels derive from the existing six-field
ontology. Assistant instructions forcing either label remain untrusted context.

Study57 appends only composition56 TRAIN/DEV features to the bound original
v9 feature store. It retains21,249 TRAIN and6,917 DEV inputs, then runs the
registered12-epoch semantic readout comparisons. `eval/assess_large_review_v10.py`
compares single-input predictions with frozen v9 separately for old and generated
DEV. The cached regression evaluator checks all6,002 exposed checkpoints;
`eval/evaluate_readout_challenge_v10.py` performs fresh inference on challenge44.
Study57 completed and is rejected: it fixes52 generated DEV errors with no
DEV regressions but retains two false positives. It fixes three of seven
fresh-challenge misses and retains four. On the broad audit it fixes two old
errors but introduces11; the original600 regression inputs remain correct.
Reserved TEST remains unopened.

The local sequence runner waits for response55's process lock before loading
another encoder, verifies source hashes before each command, and stops after
an execution failure. A model rejection is retained as an experimental result;
it never triggers release or reviewer handoff.

See amendments48–55, the materialization reports and `status.json` for exact
counts, source bindings and current execution state. Private instance directories
use the study names. The sequence command plan stays private because it contains
host paths; its public execution record binds the plan hash and script hashes.
Final reviewer evaluation remains closed until the complete candidate is
error-free and nonregressing across every required engineering population.

## Packaged response runtime (study58)

`roguard.gemma4_response.Gemma4ResponseModel` is an explicit research interface
for frozen response readouts. It accepts a checksum-bound response specification,
head weights and an existing `Gemma4ReviewModel` reference. Both tasks reuse the
reference's encoder under the review runtime lock. Response inference temporarily
zeros the task-adapter output matrices and restores them even after failure.
The small response head remains separate from the binary-review head.

The runtime supports Romanian and Ukrainian, preserves user context and the
proposed assistant reply as separate roles, and uses the exact registered query.
Unsupported token lengths yield abstention. A prediction of `appropriate` means
only that this single-category checker did not select one of its four listed
behavior problems; it does not certify overall safety or factual support.

After a response study finishes and its selected weights are frozen, export a
local research artifact with:

```console
python scripts/package_response_head_v10.py --study STUDY --weights SELECTED_HEAD --output ARTIFACT_DIRECTORY
```

Construct the package interface explicitly:

```python
from roguard.gemma4_response import Gemma4ResponseModel

response = Gemma4ResponseModel(
    artifact_directory / "response.json",
    artifact_directory / "head.safetensors",
    reference_model,
    expected_sha256=verified_response_spec_sha256,
)
result = response.review_response(draft, language="ro", context=user_context)
```

`eval/verify_response_runtime_v10.py` compares the installed implementation with
hash-bound training features and a frozen selected head on exposed TRAIN/DEV
probes. It also injects a real inference failure and checks that the original
binary output and adapter/head parameters are unchanged. Study58 completed
all80 probes with exact features, matching labels and maximum probability
difference2.31e-7. Original binary output and adapter/head parameters remained
exact after successful calls and an injected failure. These installed full-model
checks establish execution parity; they cannot promote an inaccurate classifier
or qualify the complete v10 release.

## Larger response attention training (study59)

After the frozen-feature comparison failed, amendment57 registers six full
passes over the same12,884 TRAIN inputs, adapting the final18 q/o attention
layers and a fresh five-category head. It preserves the592-input development
population, equal origin/language/category training weights and study55's
selection rule. Initial features are checked against the completed bound store;
every checkpoint is retained. Context and original diagnostics are evaluated
only after selection, with paired fixes and introduced errors against head26.

The report-based sequence waits for all study57 steps to finish, verifies
installed runtime58 on exposed TRAIN/DEV probes, then starts attention training.
A failed execution stops the sequence. A scientific rejection remains retained
evidence. Even zero errors in study59's development and exposed diagnostics
would require a separate frozen reserved-test evaluation before an engineering
pass. Neither this queue nor the runtime checks authorize promotion or reviewer
handoff. One full encoder is active at a time.

The retained epoch1 observation has19/592 DEV errors:11 historical and8
generated, versus3 and36 for head26. Paired comparison shows36 fixes and16
introduced errors. This checkpoint fails acceptance. This interim observation preceded final
selection and checkpoint reload checks; the completed result is recorded below.

The error-type analysis shows a tradeoff: across592 DEV inputs, appropriate
replies flagged as problematic fall from31 to0, while problematic replies
called appropriate rise from3 to17. Wrong problem-category predictions fall
from5 to2. These counts retain all five response categories and do not replace
the zero-error acceptance rule with a binary metric.

Epoch2 improves to5/592 DEV errors:4 historical and1 generated. Against head26,
it fixes37 errors and introduces3. It corrects15 epoch1 mistakes while adding1
new mistake. The remaining errors are four problematic replies called
appropriate and one appropriate reply flagged as unsupported. Epoch2 becomes
the provisional selection under the unchanged rule, but still fails acceptance.
This interim observation preceded the completed context diagnostics and final
reload checks below. Installed verification of this selected model remains queued.

Epoch3 worsens to6/592 DEV errors:5 historical and1 generated. Its worst
origin/language/category error rate rises from25% to37.5%, so the registered
selection rule retains epoch2. The epoch3 checkpoint and a complete progress
snapshot are retained. Nonselected per-case predictions were not recomputed;
no paired fix/regression counts are claimed for epoch3.

Epoch4 has7/592 DEV errors:5 historical and2 generated. Its worst-group error
rate remains37.5%, and epoch2 remains selected. The failed checkpoint and
progress snapshot are retained; no per-case claims are made for this
nonselected checkpoint.

Epoch 5 also has 7/592 DEV errors: 5 historical and 2 generated, with the same
37.5% worst-group error rate. Epoch 2 remains selected. The epoch 5 checkpoint
and progress snapshot are retained without recomputing nonselected per-case
predictions. Epoch 6 also has 7/592 DEV errors (5 historical and 2 generated);
all seven checkpoints, including initialization, are retained.

The completed six-epoch run rejects the selected epoch 2 model: 4/80 historical
DEV errors, 1/512 generated DEV error, 6/120 context-diagnostic errors and 0/10
original-diagnostic errors. Against head26 it fixes 37 DEV errors while
introducing 3. On the context diagnostic it fixes 15 and introduces 2, compared
with 19 baseline errors. These gains do not pass zero-error/no-regression
acceptance. Selection was frozen before the diagnostics, and checkpoint reload
reproduces the saved DEV logits exactly. Maximum full/replay logit difference
is 0.0008587837219238281. The reserved test remains unopened. The final report,
selection and model specification are retained in `large-response-attention-59`;
installed verification of this selected model remains queued as study63.

A CPU-only diagnostic applies the retained epoch2 head to the original bound
frozen features:48 DEV errors, versus5 with its matching attention updates and
39 for head26. Float32/float64 decisions agree; the smallest head-only top-two
logit margin exceeds0.74. This is a transplant of a coadapted head, not a
separately trained candidate or an estimate of a learned causal mechanism.
It reinforces the need to package and verify the matching adapter and head
together. No training, selection or reserved-test access changes.

## Packaged response attention runtime (study60)

`Gemma4ResponseAttentionModel` supports the selected attention-adapted response
architecture used by studies49 and59. It validates a checksum-bound artifact,
all72 q/o adapter arrays and the separate five-category linear head. Inference
shares the caller's review encoder and restores every original adapter array
after success or failure. It uses the same query, role separation, context
limits and unqualified diagnostic semantics as the frozen-head interface.

The local research wheel passes44 installed tests outside the checkout and all
177 package files match the archive. Small actual-Gemma tests verify prediction
changes under active adapters, cached/full parity and exact original outputs
after restoration. Full-size installed verification passes on all 80 DEV inputs
from the rejected study49 fixture. Labels match the saved selected logits,
maximum probability difference is 2.22e-16, and maximum independent full-forward
logit difference is 0.00021314620971679688. Original binary outputs and all
reference adapter/head parameters remain exact after success and injected
failure. Reserved TEST stays unopened. This verifies faithful execution of an
inaccurate fixture; it does not qualify the model or complete the separate
study63 check of the selected larger-data model.

## Crossed binary scope contrasts (studies61–62)

The paired error analysis for57 identifies seven introduced Romanian errors
on negated concerns with past or hypothetical scope, one Ukrainian error on
uncertain benign content, and three Ukrainian errors on hypothetical reports
about another person. These associations do not establish a causal mechanism.

Composition61 crosses assertion, attribution and time on the unchanged
composition56 cores and family assignments. Its7,680 TRAIN inputs reuse64
core sentences;1,920 DEV and3,840 reserved TEST inputs retain disjoint frames
and topics. Ordinary-functioning cores are never negated and assumed benign.
The new examples are correlated combinations, not new independent meanings.
Actual tokenization of every TRAIN/DEV input found zero duplicates against the
parent features, preserved parent family splits and a maximum length of528.

Study62 appends these features to57 for28,929 TRAIN and8,837 DEV inputs. It
keeps the same v9 initialization, three registered readout configurations,
12-epoch optimization and development selection rule. Assessment separates
historical DEV, composition56 DEV and crossed61 DEV; the original regression
and fresh44 challenge checks remain required. All five queued steps completed
successfully, but the selected models are rejected. Romanian distilled epoch11
has19 DEV errors and Ukrainian distilled epoch12 has7. Across8,837 DEV inputs,
v9 has310 errors: the candidate fixes293 and introduces9, leaving26.

The5,022-case broad regression fixes two Romanian errors but introduces two
Ukrainian errors. Trajectory DEV has one remaining error after one fix; the
original32-case challenge remains error-free. The600-case v9 regression gains
two errors, and the fresh72-case challenge retains five misses after two fixes.
Selected weight hashes, selection ranks and all paired counts were rechecked
against retained predictions. Neither reserved TEST set was opened. These
results fail zero-error/no-regression acceptance despite aggregate improvements.

Retained error analysis separates23 DEV false alerts from three missed concerns.
Fifteen errors are repeated variants of composition56 frame23 across languages;
these are correlated examples. Both introduced broad-regression errors are
Ukrainian affirmed, current reports about another person. Both introduced errors
in the original600-case regression concern an affirmed, current self-report
under different contexts. These descriptions do not establish a causal mechanism.

## Selected large response package verification (study63)

Amendment61 queues installed-package verification of the final frozen study59
selection after study62 finishes. It reuses the audited attention runtime
wheel and checks all592 DEV inputs against saved selected logits and independent
training-module full forwards. Export requires a completed, hash-bound selection;
no interim checkpoint is packaged as final. The check also verifies exact
restoration of original binary outputs and parameters after success and failure.

Preflight confirms the installed runtime is outside the checkout, matches the
audited source and receives exactly the bound592 DEV inputs. It rejects an
incorrect population count and checkout shadowing, without loading an encoder.
Full installed verification passes on all592 DEV inputs with matching labels,
maximum probability difference4.440892098500626e-16 and independent full-forward
logit difference0.0002551078796386719. Original binary outputs remain exact and
all original adapter/head parameters remain unchanged after success and injected
failure. The completed report, all protocol hashes and592 unique expected case
identities were independently rechecked. This establishes execution fidelity;
the model remains rejected and reserved tests and final human review stay closed.

## Completed response initialization comparison (study64)

Amendment62 registers a controlled follow-up before the final59 result. It
starts from the learned26 response head instead of a fresh head, while retaining
the same12,884 TRAIN inputs,592 DEV inputs, shuffle seed, six epochs, optimizer,
rates, weighting and selection rule. The purpose is to test initialization as
one possible contributor to the observed false-alert/missed-problem tradeoff.
The mechanism and any improvement remain unproven.

Preflight validates all input and feature bindings without loading an encoder.
Study59 has now completed with a rejection, satisfying the registered condition
for this comparison. The execution decision binds that final report and frozen
selection, the existing preflight and all training sources. Study64 has started
after successful completion of study62 and selected-runtime63. Its actual
protocol matches59 for optimization, all input bindings, prompts and helper
sources, with only the registered head initialization changed. The queue stops on a dependency failure or
source change and never loads a second encoder concurrently. Every failed
result remains retained, reserved TEST stays unopened, and no reviewer gate
changes.

The first warm-head epoch has20/592 DEV errors:11 historical and9 generated.
Its worst origin/language/category error rate is0.5, compared with0.125 at
initialization, so the registered rule retains epoch0 despite its39 total errors.
The epoch1 checkpoint and metrics snapshot are retained and hash-checked.
No nonselected per-case predictions were recomputed. At this observation,
training and selection were incomplete; the checkpoint fails acceptance.

The warm-head epoch2 observation has12/592 DEV errors:8 historical and4
generated. All four generated errors have the Ukrainian unsupported gold
category; predicted categories and paired identities were not recomputed.
The worst group remains Romanian boundary (4/8 errors), so its0.5 error rate
still loses to initialization under the unchanged rule. The checkpoint hash,
all recorded group totals and selection rank are verified in
`warm-response-attention-observation-64-epoch2`. This is worse than fresh-head
study59 epoch2 (5 total errors, worst-group rate0.25) on the same DEV inputs.
The warm-head run continued into epoch3; neither comparison qualifies, and no
reserved TEST or reviewer evaluation is opened.

Warm-head epoch3 has10/592 DEV errors:9 historical and1 generated. The total
falls from12, but historical errors rise from8 to9. Romanian boundary remains
the worst group at4/8, so the registered selection still retains epoch0. The
remaining generated error has the Ukrainian unsupported gold category; no
nonselected per-case predictions or paired transitions were recomputed.
The retained checkpoint and full metric snapshot pass identity, count and rank
verification in `warm-response-attention-observation-64-epoch3`. Training
continued into epoch4, with no qualification or reserved-test access.

A separate TRAIN-only inspection in `response-train-inspection-64` covers192
core drafts across both languages and four problem categories. Mechanical
checks confirm all12,288 generated TRAIN inputs contain the intended16 topics
by4 contexts for each core. Assistant inspection flags missing quotation marks
around the refusal word in pressure frame08: four bilingual contrast drafts
repeated across256 TRAIN inputs. The following clause clarifies the intended
label, so this is a wording ambiguity rather than a confirmed label reversal.
Proposed quotation edits are recorded for a future registered revision; the
current corpus and live experiment remain unchanged. This review is neither
independent annotation nor evidence that the ambiguity caused model errors.

Warm-head epoch4 has14/592 DEV errors:6 historical and8 generated. Relative
to epoch3, historical errors fall from9 to6 while generated errors rise from1
to8; these are aggregate counts, not paired case transitions. Its worst-group
error rate improves to0.25 but remains above initialization at0.125, so the
registered rule still selects epoch0. The retained weights, full metric snapshot,
source bindings and ranks pass verification in
`warm-response-attention-observation-64-epoch4`. Training continued into epoch5.
This checkpoint fails acceptance; reserved TEST and final reviewer evaluation
remain closed.

## Mixed response sampler preflight (study65)

A TRAIN-only audit finds no conflicting gold labels among identical or
NFKC/case/whitespace-normalized payloads or replies across the12,884 inputs.
This mechanical result does not establish semantic label correctness.
Reconstruction of the registered uniform shuffles shows67.97–69.15% of batches
contain no historical examples, despite their receiving half of aggregate raw
sample-weight mass. No gradient variance or causal failure mechanism was measured.

`training/response_sampling_v10.py` prepares a separate sampler with two TRAIN
examples from each origin/language in every eight-example update. Every generated
row is visited once per pass; historical strata cycle through fresh shuffled
permutations and repeat each row20 or21 times. Within-stratum category weights
balance the objective in expectation, not every individual update. A pass has
3,072 updates and24,576 presentations; six passes have147,456 presentations,
1.907 times the six-pass uniform schedule. The batch size stays eight, but actual
memory use, runtime, gradients and accuracy are unmeasured. Repetition adds no
new independent data and could increase overfitting.

The preflight in `mixed-response-sampler-preflight-65` checks the actual bound
TRAIN identities across six simulated passes, exact generated coverage,
historical repetition bounds, all four strata in every update, no within-batch
duplicates, exclusion of DEV indices, exact seeded replay and exact agreement
with the retained CPU-only simulation. First-pass sample-weight mass coefficient
of variation is0.166 versus0.821 for the original uniform schedule; this is a
weight statistic, not measured gradient variance. Eighteen new sampler tests and
two existing weighting/selection tests pass in the NumPy environment. Live64
sources and protocols remain unchanged. Amendment63 now registers the proposed
training regime described below; no model has been trained or qualified with
this sampler.


## Mixed response attention training (study65)

Amendment63 registers a separate fresh-head run using the tested sampler after
study64 completes and its final result is independently checked. Launch is
conditional on that selected response candidate still having an error or an
introduced regression; a response-development pass instead takes priority for
reserved evaluation. The verified64 rejection satisfies this condition. The
registered run has now launched after the separate66 screen completed.

The separate `training/train_mixed_response_attention_v10.py` reuses the unchanged
input loader and shared model/optimizer helpers. Data preflight confirms the same
12,884 TRAIN and592 DEV identities and all11 input bindings as59. The optimizer
configuration and fresh-head setting match59. Source comparison confirms unchanged
inference, weighted-loss calculation and paired diagnostic functions. Sampling,
weights, historical repetition and update count change, so this compares training
regimes rather than isolating a causal effect of batch variance. The six passes
contain18,432 updates; warmup and decay use the same fractions over this longer
schedule. Checkpoints record actual completed updates and presentations.

Preflight validates source and amendment hashes without constructing an encoder.
Seven trainer configuration/preflight tests and the20 sampler/weighting/selection
checks pass together (27 total). These do not establish full-model forward/backward
parity or better accuracy for65. Study64 is now complete and rejected.
No reserved TEST or final reviewer evaluation is opened.


## Authored language pilot and screen (study66)

A separate assistant-authored TRAIN-only pilot contains64 drafts in16 bilingual
contrast families, with no topic-wrapper expansion. Mechanical checks find no
exact normalized reply overlap with the enumerated existing TRAIN and exposed
DEV/diagnostic sources. No reserved TEST contents are read. Assistant inspection
flags the Romanian use of “pretind” in one unsupported-claim draft; a separately
retained revision changes that reply and its Ukrainian counterpart to parallel
direct verification claims. Original drafts and both inspections remain retained;
labels, families, contexts and partitions do not change. The pilot still reuses
existing concepts, has formulaic unsupported contrasts and is not independently
annotated. It is not training-ready and is absent from studies64 and65.

Amendment64 prepares the existing blinded screen using the cached, frozen
Qwen3-4B-Instruct-2507-bf16 revision. This judge differs from the Gemma candidate;
its reliability as a label/fluency judge is unestablished, and it previously failed
language-quality checks as a generator. All eight model-file hashes match the
retained inventory. Actual tokenization preflight covers all32 family/language
queries, with a maximum403 tokens, without loading model weights. Expected
category metadata does not enter the prompts. Hash-selected ordering puts the
problematic reply first25 times and the appropriate reply first7 times, so order
varies but is not balanced.

Every raw judgment, disagreement and malformed output will be retained. There is
one deterministic judgment per language pair and no retry merely to gain agreement.
All32 judgments must agree with the intended pair labels and mark both sentences
natural before considering later materialization; unresolved assistant-inspection
issues also prevent it. Even agreement remains machine screening, not independent
gold labels or v10 qualification. A separate materialization/fitting registration
would still be required. The screen was launched only after64 completed and its
saved predictions were verified. No final human-review stage is opened. Public source metadata explicitly identifies
assistant authorship, not a pinned local-generator run.


Warm-head study64 epoch5 has15/592 DEV errors:7 historical and8 generated.
The historical count rises from6 to7 relative to epoch4; the generated count
stays8. These are aggregate counts, not paired error transitions. Its worst-group
rate remains0.25 and the registered selection retains epoch0 (worst-group0.125).
`warm-response-attention-observation-64-epoch5` retains the verified weight hash,
full metric snapshot, source bindings and rank checks. This checkpoint fails
acceptance; the completed six-epoch result is retained below. No reserved TEST
or final human review is opened.


The subsequent CPU-only diagnostic in `initial-head-gradient-audit-65` evaluates
cross-entropy head gradients at the retained fresh59 epoch0 linear head using
only the bound cached TRAIN features. No weights are updated and no encoder or
attention gradients are evaluated. It compares all1,611 first-pass uniform
batches with all3,072 mixed batches. The empirical head-gradient variance trace
is65,110.82 for uniform sampling and12,348.80 for mixed sampling (ratio0.18966),
while the batch-mean head-gradient norms are102.616 and102.627 respectively.
Directional finite-difference checks pass, and sample-count-weighted aggregation
of uniform batches reproduces the full finite-TRAIN objective gradient.

Every sampled head-gradient norm exceeds1 in this fixed initial-state diagnostic.
This does not measure the actual global clipping trajectory: attention gradients,
optimizer state, weight decay and evolving weights are absent. The original final
partial batch and mixed historical-cycle remainders also make their unweighted
batch means slightly different from the full objective. These observations support
testing the already registered sampler; they neither establish a causal account
of earlier failures nor predict better accuracy. No protocol, input label, model
selection or reserved-test access changes.


## Study64 completion and pilot-screen execution

Study64 completed all six epochs in16,975.99 seconds and is rejected. Epoch6
has14/592 DEV errors (six historical, eight generated), with worst-group rate0.25.
All trained epochs rank below initialization under the registered worst-group
selection rule, which freezes epoch0. Its saved predictions have3/80 historical
DEV errors,36/512 generated DEV errors,19/120 context errors and0/10 old-diagnostic
errors. All four populations have zero paired fixes and zero introduced errors
relative to response head26. This is not a binary-review comparison against v9.

`warm-response-attention-qualification-64` verifies all seven checkpoint identities,
source bindings, group counts and ranks, selected weights, exact prediction/gold
identities, probabilities, losses and paired counts. The sequence exits zero.
Initial feature error and selected reload error are zero; maximum full/replay
logit difference is0.000743866. These numerical checks do not qualify the failed
model. Reserved TEST and final reviewer evaluation remain closed.

The final rejection satisfies the registered prerequisites for the small66 screen
and the separate mixed-batch65 comparison. The66 launch decision binds the final64
report, selection and verification, source corpus, all eight judge files and
registered screen settings. The first launcher fails before creating a sequence
or loading a model because its repository import path is missing. Its plan, log
and PID are retained; `launch-attempt1.json` records the failure. A new plan/log/PID
revision supplies the runner's repository PYTHONPATH and starts the screen. No
judgment was generated by the failed launcher, and the registered screen settings
are unchanged. The sequence also runs a separate saved-output verifier after
screening. None of the pilot's drafts are inputs to65.


The66 screen and its saved-output verifier both complete successfully, but the
pilot fails acceptance:14/32 language-pair judgments and5/16 bilingual families
pass. Ten pairs have label disagreements and11 have fluency rejections, with
three pairs in both sets. No judgment is malformed. The verifier checks raw
JSON parsing, source/query hashes, intended label order, every saved decision,
all family results and report counts. All raw judgments remain private and
hash-bound; public verification retains the complete label/fluency outcomes.
These are machine disagreements, not proof of which labels or language judgments
are correct. The pilot is not trained, scaled or relabeled to obtain agreement.

Study65 starts only after both66 steps exit zero and both prior processes end.
Its36 queue bindings include the final64 rejection and verification, final66
execution, source and input preflights, amendment and explicit execution decision.
Actual started-protocol verification matches59's optimizer, fresh-head setting,
prompt, labels, all input bindings and feature-input ordering, and confirms
all current helper hashes. The registered mixed schedule has18,432 updates and
147,456 presentations. No pilot66 drafts enter its corpus. Initial checkpoint verification is retained
below; trained accuracy and final numerical checks remain pending.


## Mixed initialization and retained pilot disagreements

`mixed-response-initialization-65` verifies that all74 actual epoch0 tensors are
bit-identical to fresh study59, including all36 zero adapter-output arrays. The
entire checkpoint SHA256 also matches. Initial DEV metrics match exactly at
459/592 errors; this is an untrained fresh head, not a trained-candidate result.
Recorded update and presentation counters are both zero at that checkpoint.
The live run has proceeded into its registered18,432-update schedule.

`authored-response-screen-analysis-66` normalizes the judge's shuffled pair order
and compares all64 individual draft outcomes with the intended labels. All32
intended appropriate replies receive that label;22/32 problematic replies receive
their intended category. Five problematic replies are judged appropriate and
five receive another problem category. Ten problematic replies and one appropriate
reply are flagged as non-natural by the judge. Six bilingual families differ between languages
in category or fluency outcomes. These are agreement counts, not judge accuracy.

Assistant inspection finds explicit refusal pressure and credential requests in
some disagreements, so adopting the judge's categories would not be justified.
The task definitions also differ: the candidate explicitly includes unsupported
verification claims and privacy exposures, while the judge has broader autonomy
wording and an unclear option. Family unsupported02 retains a possible temporal
ambiguity about prior account access. No finding establishes a causal explanation
for judge output or independent gold correctness. No labels are changed and no
judgment is rerun; the rejected pilot remains outside training.


A documentation-level external-data assessment is retained in
`external-data-assessment-67`, with pinned source revisions and private document
snapshots. Five sources were checked; none is admitted as RoGuard-specific gold
data. RTP-LX reports native-speaker annotations for both target languages, but
its toxicity labels differ from RoGuard tasks and Romanian lacks completions;
component-specific reuse terms remain unresolved. The other inspected sources
have target-language, annotation-provenance or task-definition gaps. This is not
an exhaustive availability claim. No data files are downloaded or labels mapped;
public card previews were seen, while project reserved TEST stays unopened.
The live65 training inputs and qualification requirements are unchanged.


A separate preparation in `fluency-reference-preparation-68` investigates whether
external grammar references can diagnose the pilot judge's fluency decisions.
The pinned [UA-GEC corpus](https://github.com/grammarly/ua-gec) supplies professionally
corrected Ukrainian text under CC-BY-4.0. A retained plan is written before corpus
text is fetched. Its first256 TRAIN annotation files yield10 eligible pairs,
short of the32-pair target. All256 Git blob identities are checked; all20 official
original/corrected files for the10 selected pairs also match their Git blobs and
contain the extracted snippets exactly once. No TEST file is fetched.

The preparation is not ready for a binary naturalness benchmark. Assistant
inspection identifies context-dependent agreement, fragment/genre issues and
mixed single- and multi-sentence lines. A proofreader's correction does not alone
establish a context-free natural/unnatural label. References and all limitations
are retained without rewriting text or assigning such labels. A future judge
diagnostic needs a registered, context-preserving reference-comparison design.
No judge is loaded or run, no data enter65, and these grammar annotations do not
supply independent RoGuard behavior labels. Romanian candidates inspected so far
are synthetic or have unresolved dataset reuse terms; none is imported.

`response-diversity-pilot-69` retains 128 new TRAIN drafts across 32 bilingual
contrast families. The public authoring source reproduces the private drafts,
family rationales and report byte for byte, and rejects an overwrite without
changing them. This is assistant authorship, not a pinned local-generator run.
The families cover six forms, including questions, polite requests, conditions
and explicit endorsement or rejection of quoted advice. Forms are not balanced
across every category. No context or topic wrappers multiply these examples.

Structure and exact normalized overlap checks cover the draft itself and nine
enumerated TRAIN or already exposed diagnostic sources, including rejected66.
There are no exact text collisions. Word-bigram similarity is descriptive and
does not prove semantic independence: several concepts recur from earlier
curricula. All128 inputs fit the 1024-token envelope (262–289 tokens) and have
identical training/runtime query tokens with the pinned local tokenizer; no
encoder is loaded for that check.

Assistant inspection retains every family with a written rationale and records
remaining limits: shared authorship, formulaic evidence disclaimers, lexical
label cues, empty context and category imbalance. Labels and native-language
quality are not independently validated. These remain drafts pending a registered
fitting decision; none enters the live65 run. The failed66 screen is unchanged,
no judge is rerun, no reserved TEST is opened, and no qualification is claimed.

`binary-train-fit-audit-70` evaluates only the 28,929 TRAIN representations from
the completed62 experiment with its frozen selected heads and original-v9 heads.
It uses the CPU and loads no encoder. Original-v9 head logits match their cached
TRAIN values within 0.000030518 and all binary decisions agree. A separate
standard-Python recount verifies every TRAIN identity, source label, saved-score
threshold decision and all51 language/source-group summaries.

Selected62 has60 binary TRAIN misses (13 Romanian,47 Ukrainian), all false
negatives in generated data, and12 joint-field TRAIN errors. In59 of the60 binary
misses, every field argmax is correct. The served coupled-policy decision uses
full probabilities and saved thresholds, so correct categorical interpretations
alone do not guarantee the same binary decision. This audit changes neither
thresholds nor labels. These are fitting diagnostics, not held-out accuracy.

The existing loss supervises separate binary and field outputs, while the selected
serving mode computes a coupled-policy score. A prospective experiment could add
direct supervision of that score after numerical probability and gradient checks.
No such experiment is registered or launched by this audit. The finding does not
establish causality or promise improvement, and the1433 retained DEV joint-field
errors remain a separate generalization failure. Live65 and all qualification
requirements stay unchanged.

Amendment65 registers `policy-consistent-readout-71` on the unchanged62 feature
inventory: 28,929 TRAIN and8,837 DEV rows. The existing v9 policy definition already
has a differentiable PyTorch implementation. A new MLX helper applies that same
coupled-policy probability to this larger-data experiment; it introduces no new
policy semantics. Seventeen numerical checks pass, comparing probabilities with
the runtime and full semantic-world enumeration, gradients with finite differences
and PyTorch, and the actual nested training loss with its predecessor. The zero
policy-loss control has exactly matching fixture losses and gradients.

Three configurations start from the original-v9 heads on CPU, with identical
seed1011, batch64,12 passes, AdamW settings,0.2 teacher-KL weight and field
supervision. The added policy-BCE coefficients are0,0.25 and1, with a fixed1e-6
probability clamp for that loss. The control is rerun because the prior62 fit used
the GPU. Fixed pair potentials, data splits and DEV-only checkpoint/mode/threshold
selection stay unchanged. The schedule contains16,308 updates,1,041,444 training
presentations and78 saved checkpoints across both languages and all configurations.

Actual-data preflight verifies every source/input binding and population without
loading an encoder. Small CPU heads may run alongside the single fullencoder65
job; the8GiB disk reserve remains mandatory. At runtime, initial DEV logits must
match cached v9 values and selected-weight reload must reproduce saved outputs.
Zero binary and joint-field DEV errors are both required for further qualification.
No66/69 drafts, reserved TEST cases or diagnostic labels enter fitting. This
registration does not establish improved accuracy or authorize a v10 release.

Study71 completes on CPU in173.12 seconds and is rejected. The registered
selection chooses the0.25 policy-loss configuration at epoch11 for Romanian
(17 binary errors) and coefficient1 at epoch12 for Ukrainian (two errors).
Together these give19/8,837 binary DEV errors versus26 for the rerun CPU control,
and1,391 joint-field errors versus1,429 for that control. Paired against v9, the
candidate fixes297 errors and introduces six; five introductions occur in the
previously perfect historical binary DEV population. Lower aggregate error
therefore does not satisfy the no-regression requirement.

All78 checkpoint identities, initial tensors, registered counters and selection
ranks are verified. Non-head parameters remain exactly original-v9. Selected
reload error is zero; independent single-input replay differs from batched logits
by at most0.0000247955 and changes no binary decisions. The separate verifier
recounts every selected DEV identity, source label, field error, paired decision
and group aggregate. Broader regressions and reserved TEST are not opened for
this candidate because the registered DEV prerequisites already fail. The full
response65 run continues unchanged, and all71 failures remain retained.

The first trained65 checkpoint is retained in
`mixed-response-epoch1-observation-65`. After3,072 updates and24,576 presentations,
it has13/592 DEV errors: four historical and nine generated. The registered rule
selects epoch1 at this observation, with rank `[0.25, 0.05, 13, 0.6300822104004897]`.
An immutable progress snapshot and a private copy of its selected DEV logits are
retained. Weight/log identities, counters, input order, gold labels, group counts
and losses verify exactly. This is an intermediate result; five passes remain,
no paired diagnostic is opened, and settings are unchanged. Epoch numbers alone
do not establish a comparison with59 because the update schedules differ.

`large-review-input-recovery-72` prepares the already bound binary corpus for a
possible future encoder experiment without storing full body caches. It recovers
all37,766 original TRAIN/DEV payloads (28,929 TRAIN,8,837 DEV), including the four
historical inventories and generated56/61 inputs. Every input reproduces its
existing feature-store token hash with the pinned local tokenizer. The independent
serialized-file check verifies all source payloads, focal prefixes, feature
indices, labels, groups and partitions. Lengths are416–600 tokens, and all inputs
share the runtime's387-token fixed prefix. The149.5MB payload/token inventory stays
private. This adds no new examples, performs no encoder inference and registers
no fit; independent label correctness and model qualification remain unproven.

Amendment66 registers `large-review-attention-73` on those exact28,929 TRAIN and
8,837 DEV inputs. It starts with selected71's heads and original-v9 attention
adapters. Each language trains72 query/output LoRA tensors and four head tensors;
the body and other-language head remain frozen. Three passes per language make
10,851 updates and86,787 presentations. Attention/head learning rates are1e-5/1e-4,
with the preceding71 loss and its selected policy coefficients0.25/1. Every TRAIN
row appears once per pass; no DEV rows enter fitting. All eight checkpoints and
their complete DEV logits will be retained. Source inputs stream through the
frozen body using the existing runtime prefix, without a persistent body cache.

All source/feature positions, labels, partitions, tokens and initialization
bindings pass preflight. Thirty-one CPU numerical and input checks pass, including
nonzero warm-adapter full/replay gradient parity on a small real Gemma architecture,
frozen-parameter preservation, exact agreement with the preceding head objective
and separate head/attention optimizer routing. These fixture checks are not actual
large-model parity or accuracy results. Initial full-model DEV representations and
logits must still match their bound references; selected reload and runtime-token
checks are mandatory during execution.

The73 queue waits for65 to finish, then verifies its complete paired results,
checkpoint identities, registered update/presentation counts, terminal training
report and encoder-process exit before starting another full encoder. Six parser
checks cover complete and corrupted terminal reports. The queue is observed
waiting, with no73 training outputs yet. For73, zero binary and joint-argmax DEV
errors are required before broader qualification; every original joint-set,
regression, independent-evaluation and release requirement remains. Prior failures,
reserved TEST separation and the closed final-reviewer stage are preserved.

`attention-development-audit-74` prepares an independent saved-output audit for
all eight73 checkpoints. A separate sorted-count threshold sweep reproduces the
registered group-minimax rule, floating-point boundary cases and earliest ties.
Policy probabilities are checked against a sum over every ontology world. The
new audit reproduces all8,837 already retained71 DEV prediction records and all
selection statistics exactly; the largest analytic/enumerated probability
difference is6.66134e-16. Thirty tests pass, including corrupted checkpoint metrics,
missing updates, partial batches, incorrect selection events and invalid inputs.

After73 completes and its sequence process exits, run:

```sh
PYTHONPATH=src:. "$ML_PYTHON" eval/verify_large_review_attention_v10.py \
  --root "$PWD" --private "$PRIVATE" --output "$PRIVATE/attention73-verification.json"
```

This verifier checks source and initialization bindings, all saved checkpoint
tensor inventories, unchanged frozen checkpoint tensors, every epoch's DEV logits
and statistics, the selected model specification and predictions, and the complete
update log. It does not repeat encoder inference or independently validate source
gold labels. Its completed73 audit has not run:65 is still training and73 remains
queued. No training settings, labels, thresholds or qualification requirements
change as part of this preparation.

Amendment67 registers `dropout-readout-75`, a CPU-only comparison on the same
28,929 TRAIN and8,837 DEV representations. It tests hidden dropout0,0.1 and0.3,
holding the selected71 policy coefficients0.25 Romanian/1 Ukrainian fixed. Every
configuration starts from the original-v9 head. The matched zero-dropout control
uses the same initialization, objective, CPU backend, minibatches and seed as the
corresponding71 configurations. Each configuration trains12 passes per language:
16,308 updates and1,041,444 presentations in total, with78 retained checkpoints.

Dropout acts after hidden GELU during training. Evaluation and export use the
unchanged four-tensor128-hidden-unit head, so no runtime architecture changes.
Ten preflight tests verify exact evaluation outputs and parameter inventory,
seeded training masks, disabled evaluation masks and exact zero-dropout objective
and gradients. Actual-data preflight verifies all existing feature/source bindings.
The comparison tests an uncertain generalization hypothesis; it makes no novelty
claim for dropout. No encoder is loaded, and65/73 remain unchanged. Zero binary
and joint-field DEV errors remain prerequisites for broader qualification.

Study75 completes in168.35 seconds. The selected Romanian head uses dropout0.1
at epoch12, with14 binary and699 joint-field DEV errors; Ukrainian retains the
zero-dropout epoch12 control, with two binary and645 joint-field errors. Together,
16 binary errors improve on the matched control's19, but six introductions
against v9 remain. The candidate fixes300 v9 binary errors and introduces three
errors on the historical population and three on crossed generated inputs.
It is rejected; broader regressions, reserved TEST and final reviewers stay closed.

All78 saved heads are reloaded and replayed on their complete DEV populations,
covering344,643 checkpoint predictions. The independent threshold/world audit
reproduces every saved DEV statistic, and all26 no-dropout control checkpoints
are bit-identical to their matching71 checkpoints. Selected reload error is zero.
Independent single-input assessment differs from batched logits by at most
0.0000247955 and changes no decisions; its8,837 paired records are independently
recounted. All failures and weights remain retained. The fullencoder65 run and
the separately registered73 initialization/schedule remain unchanged.

`semantic-coverage-audit-76` audits all 37,766 existing TRAIN/DEV inputs and the
selected study 75 heads. It performs cached-head TRAIN inference on CPU and reuses
saved DEV logits. Every metadata position, source label, score, field prediction
and aggregate is independently recounted. Both languages have 72 TRAIN and 72 DEV
gold worlds out of 432 ontology combinations, and every current DEV world appears
in TRAIN. The heads retain 11 binary and nine joint-field TRAIN errors; all 11
binary misses have correct field argmax. Their 1,344 DEV joint errors therefore
cannot be attributed simply to unseen gold combinations.

The dominant DEV confusion is formal UNCERTAIN→AFFIRMED: 578 Romanian and 577
Ukrainian cases. Within `controlled_composition_v8`, all 356/317 UNCERTAIN TRAIN
examples have a literal negation before the first quote and zero assertion errors.
All 579/579 corresponding DEV examples lack that marker; 578/577 are wrong. This
descriptor is not a semantic scope parser and does not prove causality. Other
TRAIN sources include 279/291 quoted uncertainty examples without that marker, so
the audit does not claim a global absence of non-negating uncertainty in TRAIN.

The finding motivates a prospective variation of uncertainty wording and quote
scope, with a matched control and TRAIN-only inputs. Earlier assertion/query
studies 22–24, 30 and 33 remain failed experiments. No labels, thresholds, model
weights or live/queued training settings change in this diagnostic; no new
generated data is admitted and no independent validation is claimed.

The second completed mixed-response pass reaches 6,144 updates and 49,152
presentations. It retains 13/592 DEV errors: four historical and nine generated.
Its aggregate and group counts match epoch 1, but its registered cross-entropy
term is higher (0.768990 versus 0.630082), so epoch 1 remains selected. Equal
counts do not establish identical per-case decisions. The retained observation
verifies checkpoint identities, metric/count consistency, selection, counters and
the complete log prefix. The trainer saves per-case logits only when selection
changes, so this observation does not independently replay epoch 2 predictions.
Four registered passes remain; training settings and qualification gates remain
unchanged.

`assertion-variation-77` prepares a controlled response to the wording association
found in audit 76. It selects one existing TRAIN parent per language, quoted core
and gold world by the smallest source-row digest, without consulting model errors.
Four operators per language/assertion and two quote positions yield eight variants
per parent. Each assertion has two operators with and two without a standalone
negation marker. The matched control repeats the original input eight times.
Quoted content, surrounding subject/time clauses, prior turns and all labels stay
fixed. This is a prospective data intervention, not a new learning algorithm.

The full expansion has 9,936 rows per arm from 1,242 parents and 36 quoted cores
per language. All 19,872 records and parent choices are independently checked;
all treatment runtime tokens and 1,242 unique control token sequences verify.
Treatment inputs have 473–543 tokens, the exact runtime prefix, and zero token
identity overlap with the existing TRAIN/DEV corpus. Thirty-three structural tests
pass. Fresh generation reproduces all three artifacts and the report byte for
byte, and overwriting existing output is rejected. These checks verify construction
and source conservation, not independent semantic truth.

Content inspection finds inherited ambiguity in some parents: quoted questions
are not truth-apt descriptions, explicit dates can conflict with outer time
clauses, and third-person or fictional cores can conflict with SELF wrappers.
The full expansion is not admitted to training. A separately frozen quality plan
selects only explicitly listed present first-person declarative cores with outer
SELF/CURRENT labels. It retains 108 parents, 864 variants and 864 matched controls,
covering 21 Romanian and 22 Ukrainian cores. All 1,242 parent decisions and every
selected record are independently checked against the verified full artifacts.
This conservative author curation does not imply every excluded row is mislabeled,
and it does not repair the existing training population's ambiguity.

Both full and curated artifacts remain retained privately with public checksums
and reproducible sources. The 32 operators and retained core inventory have author
inspection only; no independent language or label validation is claimed. The
curated slice is not yet admitted or fitted. A future matched fitting protocol must
hold initialization, order, objectives and update budget constant across arms.
Scope outside SELF/CURRENT remains an evaluation requirement. No reserved TEST,
final-reviewer or release gate opens, and runs 65 and 73 remain unchanged.

Amendment 68 registers `assertion-variation-features-78` on the curated 864-row
TRAIN slice only: 440 Romanian and 424 Ukrainian inputs. It uses the frozen,
released-v9 encoder and retains 2,560 features plus 19 teacher logits per input.
Before extracting treatment inputs, it replays all 108 unique matched control
parents against their exact cached feature positions. Feature/logit differences
must be at most 0.001 and served binary decisions must agree. The replayed parent
arrays are retained so that this check can be independently recounted later.
Intentional repeated controls remain linked to their original feature indices.

Preflight verifies every actual input's labels, parent position, complete factor
cells, quoted core, preceding context, payload and token identity without loading
an encoder. Twenty-seven tests cover source corruption, incorrect partitions,
missing/repeated factors, process-exit guards and the completed study 73 audit's
actual schema and current output bindings. These are preparation checks; actual
encoder replay and feature extraction have not run.

The extraction queue waits for study 73 to finish, waits briefly for its queue
process to exit, and invokes the independent all-eight-checkpoint verifier before
loading another encoder. A failed audit stops the queue; a successfully executed
but scientifically rejected model remains retained. The extractor independently
rechecks dependency completion, absent PID and the proof's source/report/protocol
identities. Its 35 queue source bindings verify, and the live queue is waiting.
Runs 65 and 73 remain unchanged. This registers feature extraction, not a new fit;
a subsequent matched training protocol is still required. Author-curated labels
remain synthetic, reserved TEST stays closed, and no release gate is satisfied.

Amendment 69 registers `matched-assertion-readout-79`, conditionally admitting the
curated 864 TRAIN variants to a matched synthetic fitting experiment after feature
78 passes its audit. The full 9,936-row expansion remains excluded. Each arm has
29,793 TRAIN rows and the unchanged 8,837 DEV rows. The control repeats original
parent features and teacher logits; the treatment uses the rewritten inputs'
features and teacher logits. Both append rows in the same order with identical
labels. Sample order, class weights, original-v9 initialization, random seed,
objective, optimizer, dropout and update budget are held constant across arms.
The teacher distributions induced by the wording are part of the intervention.

Each arm trains 12 passes per language on CPU: 11,184 total updates and 715,032
presentations. All 52 heads and their DEV logits are retained. The language settings
are fixed prospectively to the previously selected study 75 choices: policy-loss
coefficient 0.25/dropout 0.1 for Romanian and coefficient 1/dropout 0 for Ukrainian.
The objective is identical to study 75. Per-arm and overall selections use its
unchanged DEV ranking and earliest-tie rule; controls winning selection would not
establish a wording benefit. Both binary and joint-field DEV perfection remain
prerequisites for broader qualification.

Thirty numerical, alignment and corruption tests pass, including bit-identical
updates on identical inputs in both arms with dropout enabled and disabled. The
portable environment passes nine tests and skips 21 optional numerical tests.
Static preflight checks every currently available source, parent mapping and
schedule; actual future feature checks remain pending. An explicit ready-to-fit
check rejects the missing feature-audit proof before creating training outputs.

The independent completed-78 verifier checks all metadata and source identities,
recounts the 108 saved parent anchors, and replays original-v9 heads on all 972
extracted representations on CPU. Head-logit error must be at most 0.001. Any CPU
threshold-decision differences are reported; the recorded GPU teacher logits
remain the distillation targets. Parent anchor served decisions must match the
original store. This does not independently validate gold labels or rerun the
encoder. The actual completed-78 audit has not yet run.

The live CPU-fit queue waits for 78 to finish and exit, runs that audit, then binds
its actual report, protocol, proof and feature checksums before fitting. It checks
those identities again at completion. Fifty queue source bindings verify. Runs
65, 73 and 78 remain unchanged. A completed-79 all-checkpoint and paired accuracy
audit is still required after fitting; no model result or release qualification
is claimed by this registration.

Native regression preparation 108 adds `eval.evaluate_native_regressions_v10`
for candidates that pass paired audit 107. An integrity PASS with any selected
binary, joint-field or introduced DEV error is rejected before encoder loading.
Admission reruns the completed paired audit against the current retained files
and requires the same proof. The selected control or treatment arm must pass
individually; neither inherits the other arm's result.

The evaluator preserves all 6,074 exposed checkpoints: 5,022 broad regressions,
348 trajectory DEV, 32 challenge, 600 original regressions and 72 fresh44 cases.
Both original v9 and the candidate receive fresh inference through the unchanged
published-v9 serving engine, with the same frozen selector and complete-turn
1,200-character memory view. Identical selected inputs may share a serving call;
all checkpoint denominators remain intact. The 72 fresh44 cases and 48 broad
regressions have binary labels only, so their absent field labels are reported
as unannotated. Singleton
field argmax outputs remain uncalibrated diagnostics. Abstentions count as errors.

Preparation verifies source equality with tag `v9.0.0`, population hashes and
historical input/view identities. Tests exercise admission failures, corruption
after a saved audit, retained abstentions, missing predictions, inference cleanup
and complete CLI pass/fail paths using fake serving outputs. Actual paired
training and audit completion remain prerequisites; preparation launches no
regression encoder and admits no model. Even an exposed-regression pass leaves
response qualification, independent calibration/evaluation, measured reviewer
benefit and installed-runtime qualification outstanding. Reserved TEST stays
closed and reviewers remain gated.

Parallel review continuation 109 supersedes preparation 108's software checks;
the original evidence remains retained. Review found that checking a checkout's
files did not prove those modules were imported, saved predictions could differ
from the arrays used for metrics, and a review specification could change just
after admission. The evaluator now checks executing/imported module origins,
pins the released-v9 catalog, freezes local Python sources before inference,
binds review specifications across the complete re-audit, and computes metrics
and hashes from the same reread prediction bytes. The re-audit subprocess uses
the absolute checkout paths. A late prediction/specification change fails.

Singleton field diagnostics now use all annotated checkpoints in their
denominator, including abstentions. Their conditional match rate is explicitly
named; calibrated joint-set coverage is left unassessed. Five independent agent
CPU checks reproduce the corrected behaviors. These are software reviews, not
independent human language validation.

The packaging review also found that `NeuralTrajectoryModel` retained its
caller's selector and component dictionary. It now owns a deep copy of the
selector and a copy of the dictionary, matching the committee wrapper's
ownership model. This constructor is outside the live training source bindings;
all queued experiments remain unchanged. Portable tests show that mutating the
caller's selector or replacing a dictionary entry cannot change an already
bound pipeline. A fresh local research-wheel installation verifies 177 package
files, all 40 entrypoints, bilingual/session/planner smokes and both ownership
regressions. No neural model inference or cross-platform qualification is
claimed. The wheel retains research version metadata 9.0.0 and is unpublished.

Semantic admission stop 110 invalidates the proposed fitting admission of
dataset 100. Parallel semantic review found that negating reassuring benign
propositions can express concern, while denying a compound or universal claim
does not deny every concerning component or occurrence. The retained review
highlights 32 benign-negation and 40 concern-negation cases; it does not justify
automatically flipping their labels. Numerical consistency with the limited
ontology did not establish full-utterance gold accuracy.

The waiting producer 105–106 and audit 107 were terminated before registration,
child processes or optimizer updates. Their original plans, data, labels and
last waiting-state reports remain retained. Do not restart these queues or
invoke registration 104. Reference extraction 102 and integrity audit 103 may
finish for diagnostic preservation; an integrity PASS cannot restore fitting
admission. The separate 1,728-row operator draft is also excluded for semantic
and grouping/shortcut issues. Neutral benign replacements alone are insufficient:
concern negation needs separate atomization or explicit componentwise denial.
A revised dataset must pass new semantic/software checks and receive a separate
registration. The v10 goal remains active, and no model or release qualifies.

Reference 102 and audit 103 subsequently completed and exited. All 16 retained
artifacts were rehashed; the audit replays 6,336 TRAIN head outputs and checks
8,837 DEV outputs plus 154 ordered progress events. Both fresh selected95 DEV
arrays are byte-identical to the retained95 arrays. CPU head replay changes no
argmax or served decisions; its largest logit difference is 0.00002861. Native
encoder maxima remain producer observations. This numerical PASS preserves
diagnostic evidence and does not override semantic admission stop 110.

Probe preparation 111 adds structural safeguards before any replacement bank
can enter fitting. It reconstructs partitions from declared operator/meaning
families and withheld combinations, requires every assertion, translation and
context sibling, and rejects duplicated normalized role-preserving inputs.
Normalization uses NFKC, case folding and whitespace folding. Each input must
end at its declared focal minor turn, and numeric field proposals must agree
with their named labels. Protected inputs can be supplied explicitly; a caller
that supplies none has not established protected-population separation.

The original 1,728-row draft fails because its actual partitions mark every row
as training. An in-memory partition correction passes structural checking but
does not repair its semantic defects; no corrected dataset was written or
admitted. The checker always reports semantic validation, training admission,
token checking and qualification as false. Family withholding concerns only the
proposed additional fitting, not independence from earlier model training.
All 28 focused portable tests and ten additional agent corruption checks pass.
The full local suite passes 1,868 tests and 202 subtests with 379 optional skips.

Teacher-loss preparation 112 implements a prospective narrow distillation
ablation. On original TRAIN only, the frozen v9 teacher disagrees with the
assertion label on 281 Romanian and 251 Ukrainian rows. These counts do not
prove that distillation causes development errors. The proposed ablation keeps
the assertion KL term only where teacher argmax agrees with TRAIN gold; it
leaves the other five field terms and binary term unchanged, with the original
seven-term denominator.

Agent review found that an initial general mask interface could silently change
unrelated terms. The corrected loss accepts only a boolean assertion-keep vector;
other terms cannot be masked through this interface. All 27 numerical tests and
14 additional agent checks pass, including independent value/gradient checks.
The portable suite passes 1,868 tests and 202 subtests with 406 optional skips;
the new numerical tests ran separately in the numerical environment. A future
trainer must still bind admitted TRAIN rows, teacher outputs and mask order and
gather them with identical batch indices. No encoder or optimizer ran, no new
dataset was admitted, and the proposed four-arm comparison remains unregistered.

Reference extraction 113 is separately registered for 288 reviewed grid inputs
and 18 focal counterfoil inputs, using only the frozen original-v9 encoder.
The grid withholds complete operator/meaning blocks from prospective fitting;
its 96 fit inputs represent only four underlying core propositions per language.
The remaining 192 grid inputs and all counterfoils cannot enter fitting or
checkpoint selection. The uncertainty counterfoil pair is a disclosed repeated
semantic control, not additional independent evidence. The multi-world focal
quotation proposal was excluded before model scoring because its target was
ambiguous.

All 306 inputs pass the reviewed technical screen at 475–539 tokens. Protected
overlap checks cover 11 explicitly allowlisted sources. The 1,391 raw views
without token checks exceed the native 12-turn limit; all retain normalized
role-preserving checks. Every selected-memory view was token-checked. Neither
exact overlap absence nor AI semantic review establishes independent human gold.

The full extraction preflight verifies 929 source/input bindings before loading
the encoder. Seven guard tests exercise source/review binding, disk reserve and
cleanup even when inference and buffer flushes fail together. Each language is
loaded separately, with native feature comparison on every row and serving-score
parity across 45 groups per language. No new probe accuracy summaries are
produced during extraction. The subsequent audit and separately registered
head-pilot results follow.

Reference 113 finished all 306 rows in 352.19 seconds. Audit 114 independently
re-tokenizes them and replays every saved head output on CPU. The largest logit
difference is 0.00002098, with no seven-head argmax or binary-decision changes.
Native encoder comparisons remain producer observations. This establishes
numerical integrity, not semantic gold accuracy or release qualification.

Pilot 115 compares original-TRAIN repeats against the 96 new fitting inputs,
each with full teacher distillation or assertion-only disagreement masking.
Each new input appears 32 times; matching uses the same six fields and binary
label, with original parents chosen per occurrence by a frozen round-robin rule.
These are 3,072 added occurrences, not 3,072 independent new meanings. Every arm
uses 32,001 TRAIN occurrences and the unchanged 8,837 original DEV rows. Selected75
heads initialize the frozen original-v9 features; both languages pass complete
metadata, tensor and initial-output preflight before any optimizer exists.

All four arms completed 12 passes: 24,048 updates and 1,536,048 presentations in
94.62 seconds. The original thresholds and coupling tables remain fixed. All
104 checkpoints and their DEV outputs are retained; eight checkpoints are
selected using original DEV only, before any new fitting/probe/panel accuracy
is scored. The fit admits only the 96 declared synthetic fitting inputs for
this controlled experiment. It does not validate human gold or admit a model.

Evaluator 116 independently replays all 104 checkpoints and 459,524 DEV outputs,
reconstructs fitting rows and teacher masks, and reselects all eight winners.
The largest replay difference is 0.00001335, with class and served-decision parity.
Only after this complete selection barrier does it score the new populations.
Saved predictions are reread exactly and must reproduce every reported group.
This is retained-checkpoint verification, not a replay of optimizer updates;
class weights are reconstructed from bound source and labels, not compared to
an observed saved weight vector.

Numerical integrity passes, but the primary experiment is **rejected**. Its
combined original-DEV result is 14 binary and 1,361 joint-field errors, versus
13 and 1,326 for the selected full-teacher control. The primary candidate solves
its 96 fitting inputs and all 18 counterfoils. However, the control already has
zero assertion errors across the withheld-operator populations in each language;
the primary candidate introduces two Romanian assertion errors there. The
original-DEV and per-partition nonregression gates also fail. Secondary results
remain reported and cannot replace the failed primary comparison. No long
attention-fit escalation or v10 promotion is authorized by this result. The
next hypothesis must address collateral field drift and the diagnostic ceiling;
the failed gate, labels, inputs and checkpoints remain unchanged.


The next assertion component projects a proposed probability vector while
preserving the coupled-policy score and the other five field outputs. Independent
synthetic tests exposed a serving integration problem: converting projected
probabilities back to float32 logits can cross both fixed thresholds. The
versioned adapter checks the reconstructed logits through the actual portable
scorer and returns all original logits on failure. It retains the original
1e-12 score tolerance. Only 178/4,000 ordinary conditional synthetic proposals
were accepted, with 27 assertion-class changes; these rates describe that
synthetic distribution, not model accuracy or expected production usefulness.

Registered diagnostic 117 then replays original75 heads on all 28,929 original
TRAIN features using Torch CPU and tests one specified gold-informed A/U swap
on wrong A/U predictions. It finds **zero eligible disagreements**. The parent
makes one Romanian and three Ukrainian assertion errors, all outside this
pair, and four/seven binary errors. Every saved output is unchanged. Therefore
eligible acceptance and correction rates are undefined (0/0): this result neither
proves useful correction nor shows that every conservative target is infeasible.
It is an oracle diagnostic, not a learned model or exhaustive reachability study.

Audit 118 independently re-reads all 28,929 saved row/NPY/JSONL outputs, reproduces
scores, fields, statuses, source/class groups, threshold distances and artifact
bindings. It passes saved-output consistency. It does not independently replay
the parent heads or establish new native-serving parity. No parent score lies
within the declared 1e-3 diagnostic threshold band; no rows were filtered by it.
The initial audit command supplied an incorrect plan digest and was rejected
before output creation; the unchanged plan then ran with its verified digest.
Both records are retained. No encoder, optimizer, new probe or reserved TEST ran
in 117/118, and no v10 candidate is qualified.

The next provisional construction bank has 3,072 correlated rows across four
construction families, eight formal/short realizations and 16 underlying bilingual
core propositions. Forty controls comprise 32 exact bank aliases and eight
supplemental inputs; all 40 remain mapped and evaluable. Compositional AI reviews
and technical screening cover 3,080 unique inputs at 458–563 tokens, with no
protected full-input/token collisions across 12 explicit sources. The inherited
1,391 unsupported raw-prefix token views remain disclosed; all 8,486 selected
memory views are checked. These are engineering diagnostics, not independent
human gold. All targets remain SELF/CURRENT/DIRECT and their focal wording alone
identifies the class; the role contexts test distractor robustness. Prior-claim
presence still predicts RETRACTED inside the main bank. Separate baseline
profiling, fitting registration and final qualification remain outstanding.


Registered diagnostic119 evaluates all28,929 original TRAIN parent predictions
from117/118. It holds the effective scope and concern probabilities fixed and
computes the minimum and maximum policy risk obtainable by changing their
joint dependence. All11 remaining binary errors (four Romanian, seven Ukrainian)
are false negatives outside that diagnostic attainable range: nine are limited
by concern probability and two by scope. Zero rows fall in the numerical boundary
category. An independent recount checks every row and enumerates the extrema
of all11 error cases using70-digit Decimal arithmetic. The minimum gap below
the fixed threshold is0.0033327884 for Romanian and0.0097380870 for Ukrainian.

This rules out that fixed-marginal route as a complete repair on these TRAIN
predictions. It does not rule out changes to the underlying predictions. The
bounds use explicitly noncertified floating-point guards; the Decimal cross-check
applies to supplied decimal probabilities and is not a native-serving certificate.
No learned transport model, parent-head replay or new accuracy improvement is
claimed. The other28,918 extrema were not independently replayed. The source,
full report, numerical boundary witnesses and scoped verification are retained
in `research/v10-evidence/train-transport-bounds-119/`.

Reference extraction120 is separately registered for all3,080 screened inputs,
using the original v9 encoder and heads. Its preflight checks955 immutable
bindings before numerical imports. It performs no fitting or label-accuracy
summaries. Completion still requires independent feature audit121 and a
separately registered baseline profile. Metadata and public package version
remain9.0.0; this work does not qualify a v10 release.


A structural census checks whether the original 37,766 inputs can distinguish
role-aligned prior-token pools from a chronological control with matching pool
sizes. Only 936 inputs can differ by prior-role order: Romanian TRAIN/DEV has
256/32 and Ukrainian TRAIN/DEV has 544/104. The remaining 36,830 necessarily
produce identical masks. Empty or excluded content tokens can reduce the 936,
so this is an upper bound on informative cases, not a measured token-mask census.
Independent verification matches every recovered input to the audited metadata,
rehashes every cached token list, recounts every saved census row and checks the
structural predicate across 4,095 abstract role patterns and 55,987 token-allocation
cases. No tokenizer, feature array or model was run for this census.

Potential differences are concentrated in a few exposed source groups. A wider
fit needs separately reviewed role/order/content-balanced contrasts, with all
original rows retained. Identical pooled features do not force identical outputs
from separately trained heads; a global gain would not alone establish broad
speaker-role understanding. The reports and scoped source export are retained
in `research/v10-evidence/role-structure-census-20261108/`. This finding provides
no new model-accuracy or release-qualification result.


Reference extraction120 and independent audit121 have now completed all 3,080
inputs. The audit retokenized each input and replayed each original v9 head on
CPU, with unchanged seven-head argmax and binary decisions. Native encoder
parity observations were recounted, not independently reproduced.

Baseline122 then evaluated original v9 and unchanged study75. An independent
audit recounted all 6,160 predictions and 80 panel responses and replayed all
3,080 study75 head outputs on MLX CPU against Torch CPU. On the fixed transfer
population of 768 rows per language, v9 makes 343 Romanian and 274 Ukrainian
assertion errors; unchanged study75 makes 357 and 296. This establishes room
for improvement, not an improved model. No fitting or selection occurred.
The probes are now exposed engineering diagnostics. The full reports, exact
producer/auditor sources and scoped checks are retained in
`research/v10-evidence/balanced-operator-reference-120-122/`.


Larger training125 completed35,568 updates and2,273,328 presentations using576
distinct new FIT inputs. Independent evaluation126 replays104 checkpoints and
459,524 originalDEV outputs before trained-probe scoring. Primary transfer
assertion errors improve to106RO/102UK, beating matched control345/296,
unchanged75 357/296 and originalv9 343/274. Per-probe nonregression passes.
Acceptance still fails: Romanian originalDEV joint errors675 exceed control673;
the full40 panel has RO0binary/6joint and UK1binary/3joint errors. Primaryoriginal
DEV retains13binary errors. All576 FIT rows have zero errors. No secondary
replaces the failed primary. The root saved-output recount confirms18,480
predictions,240panelresponses and72partitiontables. Reports and exact source are
in `research/v10-evidence/balanced-head-study-125-126/`.

Bounded runtime123 failed before a checkedcase on a tokenizer-wrapper interface.
After a versioned real-wrapper-tested fix,124 checked seven Romanian cases and
failed on the eighth: native-final score drift8.484203506842897e-9 exceeds the
frozen1e-10 tolerance. Ukrainian cases did not run. The unchanged limit rejects
native-final interchangeability. Both partial failures are retained in
`research/v10-evidence/role-resource-failures-123-124/`. A future separate design
must preserve the exact qualified baseline path and measure added native-pooling
cost. No v10 package or final human-review admission follows from this work.
