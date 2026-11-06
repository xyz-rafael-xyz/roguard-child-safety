# V10 evidence research — not a release

This branch develops long-conversation evidence selection, joint interpretation
sets, stricter evaluation, and reviewer-study tools. **V9 remains the qualified
default. No v10 model has passed the complete promotion gate.**

The owner requires zero observed errors in the required evaluations and a
measurable improvement over v9. This is a test acceptance requirement, not a
claim that any finite experiment proves universal perfection. Failed candidates
and exposed development examples remain in the record.

## Implemented

- A validated trajectory format with family, author and generator provenance;
  exact prefix identities; original Unicode evidence offsets; and rejection of
  future evidence, missing predictions and split leakage.
- A learned, bounded extractive view. Mandatory focal evidence cannot be partly
  discarded to satisfy the budget. The host retains the original transcript.
  These are relevance proposals, not verified semantic summaries.
- FlowLedger integration that inherits every consulted source, including omitted
  sources. Long lineages are grouped without losing revocation dependencies.
  The host still binds source IDs to the supplied text and authorizes use.
- A portable sparse student with review, interpretation, relevance and response
  heads. Standard-library inference works without an accelerator. Its response
  head has only 20 fictional training replies and is unqualified.
- Family-level, language-stratified split-conformal joint sets bound to the model,
  selector, representation, budget and policy. Insufficient calibration returns
  the full ontology; unsupported/empty sets cannot silently become confidence.
  Exchangeability and ontology assumptions remain unverified in deployment.
- An experimental neural bridge retaining the v9 decision path on supported
  prefixes and applying the learned selector to longer inputs. Its added field
  output is checked numerically against the unchanged score computation.
- An explicit session review floor with host-reviewed, context-bound resolution.
  A benign topic change does not close an earlier concern.
- A durable SQLite session API that checks every appended minor prefix, retains
  unresolved positive predictions and abstentions across restarts, and records
  host reviewer resolutions. Atomic revisions prevent concurrent writes or stale
  resolutions from losing history; request IDs make retries idempotent. Model
  failure aborts the entire append and must be surfaced by the host.
- Blinded annotation packets, disagreement detection, counterbalanced reviewer
  packets and elapsed-time summaries. No participants were contacted or invented.
- A single-opening final-test workflow that binds artifacts before exposing test
  content. A failed attempt still consumes that test. Hashes do not prove blindness.
- A development loop that retains rejected candidates and a promotion assessment
  that rejects missing evidence. No automatic release or default-profile switch.

## Evidence so far

The engineering corpus reuses **only the old paired-scope-117 training partition**,
with synthetic distracting turns. Its 36 bilingual-aligned families are assigned
as 20 training, 6 development and 10 calibration families. There are 1,832 training,
348 development and 580 calibration prefixes. The distinct 32-case diagnostic has
only four developer-authored families and is **exposed development data**, not a
sealed final test. The published 600-case v9 test is an exposed regression.

| Candidate | Observation | Disposition |
|---|---|---|
| Student 01, pooled lexical features | 32/32 diagnostic binary decisions, but joint coverage 75% RO / 93.75% UK | Rejected |
| Student 02, joint head | 32/32 diagnostic decisions and complete diagnostic joint coverage; 14 development binary errors | Rejected |
| Student 03, balanced selection and separate focus/history | One development binary error; substantial regression on the old 600-case v9 test | Rejected |
| Further registered sparse configurations | Results in the development-loop ledger; a binary-perfect development score alone does not qualify interpretation | Research only |
| Frozen v9 | Correct on all 16 supported diagnostic cases; abstains/rejects the other 16 because they exceed its input contract | Baseline |
| V9 with simple lexical retrieval | 32/32 diagnostic binary decisions | Stronger baseline; defeats a claim that diagnostic binary success alone is novel |
| Neural bridge 04 | 344/348 development decisions; 32/32 diagnostic and 600/600 regression decisions; two Ukrainian regression interpretations omitted | Rejected |
| Score combinations 07 and 08 | 347/348 family-cross-validated development decisions; 32/32 diagnostic and 600/600 regression decisions; inherited interpretation errors | Rejected |
| Uniform memory 09 | 346/348 development decisions; 32/32 diagnostic and 600/600 regression decisions; 22 Ukrainian regression interpretations omitted | Rejected |
| Score combination 10 | 346/348 family-cross-validated development decisions; 32/32 diagnostic and 600/600 regression decisions | Rejected |
| Semantic readout 11, retrained neural heads | Broad binary audit changes from 5,020/5,022 to 5,021/5,022, fixing two errors but introducing one; 600/600 original binary decisions retained; 346/348 trajectory development decisions; structured argmax regressions remain | Rejected |
| Semantic readout 12 and monotone consensus 13 | Expanded training retains 5,020/5,022 broad and 600/600 original binary decisions; trajectory development stays 346/348; structured errors remain | Rejected |
| Attention training 14 | Zero binary errors on 6,668 eligible development inputs; 348/348 trajectory development, 32/32 diagnostic and 600/600 original regression decisions; broad audit remains 5,020/5,022, fixing one old error and introducing one new false alert | Rejected |
| Frozen score comparator 15 | Selecting an existing v9 score mode/threshold on the same development data still leaves two Romanian development errors; broad audit worsens to 5,009/5,022 | Comparator; rejected for promotion |
| Semantic lexical probes 16 and 17 | Plain and quotation-separated features leave 3,235 and 3,150 joint interpretation errors respectively on 6,668 development inputs; neither changes the binary review decision | Rejected as standalone interpretation models |
| Base-reference heads 18 | Four classifier designs compare pretrained-only and paired pretrained/v9 representations. Selected Romanian head has zero binary and 632 joint semantic development errors; Ukrainian head has one binary and 611 joint semantic errors | Rejected as a bilingual replacement |
| Frozen v9 with a 4,096-token full-prefix comparator | 32/32 diagnostic decisions with complete annotated-evidence retention | Stronger baseline; extended context is outside v9's qualified envelope |

Combination development is **meta-training**. Its family cross-validation chooses
regularization and is not an unbiased final evaluation. The calibration partition
has also been exposed during repeated exploratory work. A final study needs fresh
calibration for the selected pipeline and an independently held final test; these
exploratory sets do not establish a coverage guarantee.

The new `semantic-training-12/report.json` also audits exact normalized input
overlap with v9's earlier neural fine-tuning. Some trajectory development and
calibration inputs were already in that fine-tuning corpus: splitting the later
trajectory dataset does not undo this exposure. These partitions are exploratory
for the neural model. The report preserves affected checkpoint IDs. Fresh neural
calibration must exclude the complete prior training history.

Attention experiment 14 uses 19,202 training inputs and updates the last 18
layers' query/output adapters and the active language head. Its development
selection excludes the 21 exact prior-training overlaps, leaving 6,668 inputs
across the original and trajectory development sets. The headline 348-case
trajectory score also includes those 21 exposed overlaps and is descriptive.
Selecting only an existing frozen-v9 score mode cannot remove the two remaining
Romanian development errors; the attention update does. This is a development
gain, not independent superiority.

Fresh attention inference on all four exposed evaluation populations reduces
broad-audit semantic argmax errors from 133 to 73, but still leaves 21 Ukrainian
semantic errors on the original 600 cases. These are **uncalibrated singleton
predictions**, not a coverage guarantee. The independently recomputed
[`semantic-assessment-14/report.json`](../research/v10-evidence/semantic-assessment-14/report.json)
retains the exact fixed/new binary error IDs and rejects the candidate. Saved
logits reproduce every recorded score and binary decision exactly. No ensemble
or threshold is fitted to these regression failures.

The next registered comparison, language specialists 19, chooses between
attention-14 and base-reference-18 separately for each language using identical
development IDs and labels. It selects Romanian base-reference and Ukrainian
attention; regression evaluation is separate from selection. Scope curriculum 20
adds 3,456 synthetic TRAIN examples in 864 matched assertion families. The
effective-input audit retains every focal text and finds no exact protected-prefix
overlap. Retractions include a prior affirmative statement; the other variants
share neutral history. These developer-authored frames still lack independent
language review. See the machine-readable status for current run completion.

The uniform-memory experiment computed 481 changed neural inputs and reused
1,079 results after checking input identities and frozen component bindings.
This reuse does not measure end-to-end latency. Selecting evidence uniformly
reduced two binary errors, but its narrower calibrated sets worsened Ukrainian
coverage. That tradeoff is a failure under the owner's acceptance rule.

The older **5,022-case v9 prior audit** is a separate regression from the 600-case
test. V9 recorded two Romanian false alerts there. The new broad audit preserves
its labels and verifies whether inference inputs changed before reusing scores;
its results are in `broad-comparison-10`. Do not describe v9 as error-free on every
recorded population.

The first response head correctly classifies only 3 of its 10 separate fictional
diagnostic replies. It must not be presented as a reliable semantic response
checker. Existing response lint remains separate.

The v9 runtime enforces both a **12-turn prefix limit** and a **1,024-token model
limit**. Long-input abstentions are visible in comparisons. Gains from accepting
longer inputs are distinguished from gains in understanding supported inputs.

Results, model artifacts, protocols and amendments live in
[`research/v10-evidence`](../research/v10-evidence). Large upstream model weights
remain outside Git. No new real-child data is collected.

For the persistent host API, see [durable sessions](V10_DURABLE_SESSIONS.md).
For neural training and public starting artifacts, see
[neural reproduction](V10_NEURAL_REPRODUCTION.md).

## Reproduce the portable experiment

From the repository root, using a Python environment with the project on its path:

```sh
PYTHONPATH=src python -m training.build_trajectory_v10 --output /tmp/v10-data
PYTHONPATH=src python -m training.train_trajectory_v10 \
  --data /tmp/v10-data --output /tmp/v10-student \
  --min-relevance 0.5 --joint --balanced-relevance \
  --representation focus-context --review-mode joint-policy
PYTHONPATH=src python -m eval.run_trajectory_v10 \
  --cases /tmp/v10-data/challenge.jsonl \
  --model /tmp/v10-student/student.json.gz \
  --calibration /tmp/v10-student/calibration.json --plan --output /tmp/v10-evaluation
```

Training requires NumPy, SciPy and scikit-learn. Inference for the exported sparse
artifact uses only the standard library. Record exact dependency versions for
each run; numerical differences can change selected thresholds. Commands refuse
to overwrite study directories.

Install the optional training dependencies with `pip install '.[trajectory-train]'`.

```sh
roguard-v10 conversation.json --model student.json.gz \
  --model-sha256 ACTUAL_ARTIFACT_SHA256 --calibration calibration.json --plan
```

The CLI labels proposals as unqualified and grants no permissions. `--draft`
invokes the explicitly experimental response head. Do not connect it to automatic
response delivery.

`--neural` uses the already-installed immutable `v9-gemma4-20261019` profile with
the supplied student artifact as its learned selector. It requires the existing
`gemma4` extra and Apple Silicon. Fetching remains explicit through `roguard-fetch`;
inference does not download anything. Supply the **neural pipeline's** calibration
artifact, not a sparse student's calibration. Mismatched identities are rejected.

The current extractor also retains unpunctuated lines and punctuation-only turns,
and verifies that mandatory evidence is present. Its `v2` memory contract rejects
old calibration identities. For the checked-in Student 02 artifact, the updated
portable calibration is `memory-contract-v2/portable-calibration.json`; the
matching neural bridge calibration is `memory-contract-v2/neural-calibration.json`.
The migration audit verified unchanged selected text on all 1,560 existing
prefixes. The frozen original study sources and results remain in `neural-04`.
The uniform-memory variant remains an experiment invoked through
`training.run_uniform_memory_v10`, rather than the default neural CLI path.

`training/run_v10_development_loop.py` evaluates its registered configurations on
development data and exits rejected if none meets the development prerequisite.
It never opens a sealed test. New experiments require a prospective amendment;
the final test cannot be recycled as a stopping criterion for repeated tuning.

## What remains before promotion

Independent fluent authors and adjudicators must supply the final evaluation.
The study needs a prespecified sample size, independent label provenance, the
larger-context comparator on that independent population, response-model
qualification, a measured human reviewer study, and installed runtime checks on
target platforms. Local software verification passed 1,151 tests (79 optional
tests skipped) and installed-wheel smoke checks on Python 3.9, 3.10 and 3.12.
Those checks establish software behavior, not model accuracy. A compact
student that regresses against v9 cannot replace it merely because it is faster.

See [independent study handoff](V10_INDEPENDENT_STUDY.md). Runtime controls do not
establish interpretation truth, legal authority, calibrated individual safety,
or a measured benefit to children.
