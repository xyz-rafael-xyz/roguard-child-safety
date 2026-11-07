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
and fresh44 challenge checks remain required. The bound queue has started
feature extraction after response59 and attention-runtime60 verification
completed. Neither reserved TEST set
opens unless every required preceding error/regression gate passes.

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
Full verification remains pending. A passing execution check would not qualify
the model or open reserved tests or final human review.

## Prepared response initialization comparison (study64)

Amendment62 registers a controlled follow-up before the final59 result. It
starts from the learned26 response head instead of a fresh head, while retaining
the same12,884 TRAIN inputs,592 DEV inputs, shuffle seed, six epochs, optimizer,
rates, weighting and selection rule. The purpose is to test initialization as
one possible contributor to the observed false-alert/missed-problem tradeoff.
The mechanism and any improvement remain unproven.

Preflight validates all input and feature bindings without loading an encoder.
This comparison is prepared, not queued or launched. Execution requires the
completed59 result and completion of already queued work. If59 clears its
development and exposed-diagnostic gates, its reserved-test qualification takes
priority. All failed results remain retained and no reviewer gate changes.
