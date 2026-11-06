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
