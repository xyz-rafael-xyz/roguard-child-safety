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
