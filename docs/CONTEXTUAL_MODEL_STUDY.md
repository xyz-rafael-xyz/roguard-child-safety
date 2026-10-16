# Contextual instruction-model study and review evaluation

## Status

This is an experimental development study. It does not replace the published
model or establish real-world child-language accuracy. The stable-promotion
requirements remain at least 99% concern recall and at most 5% benign false
reviews in each language and formal/conversational group, followed by fresh
assessment and integrated runtime checks.

## Instruction tuning

`training.prepare_instruction_sft_v8` creates a 2,400-row bilingual training set,
a 160-row validation subset and a 400-row development screen. Training uses only
previous training inputs; exact normalized overlaps with existing development and
calibration are rejected. Available annotated concern families are disjoint.
Half the selected training rows lack those family annotations, so this pilot does
not claim complete family independence. Validation is part of the development
screen, not an independent test.

The local experiment adapts the last eight layers of the cached Qwen3-4B-Instruct
2507 quantized model with rank-eight LoRA. The registered pilot uses 200 steps,
batch size two, gradient accumulation two, and assistant-only loss. In this
trainer, 200 iterations represent 400 example presentations and 100 optimizer
updates, not a full pass over the 2,400-row training set. Validation loss uses
eight batches per check; the classification screen evaluates all 400 selected cases. Maximum
prepared sequence length is 582 tokens, below the 1,024-token limit. No real
children's messages are used. The frozen legacy 48-case probe is not used.

`training.evaluate_instruction_sft_v8` reads the complete conversation prefix and
focal turn. It scores the two label tokens; the pilot cutoff is fixed at 0.5.
If the most likely next token is neither label, the result is unresolved and
counts as an error in strict metrics. Scores are not safety probabilities. The
script refuses silent input truncation and retains per-case outputs and hashes.

## Separate paired-family benchmark

`training.context_families_v8` defines 16 fictional situation families, assigned
as whole units to training (eight), development (four), or reserved test (four).
Both languages and all scope/format variants stay with their family. The 672
rows comprise 336 training, 168 development and 168 reserved-test rows. Variants
are correlated; they are not 672 independent situations.

Seven conditions cover personal statements, fictional framing, explicit denial,
past events, another person's account, a topic reset, and a shift from fiction
to a personal statement. Original, Unicode-decomposed and whitespace variants
measure invariance. Family isolation applies within this benchmark; it does not
prove absence from historical corpora or model pretraining. The language and
labels are project-authored and have not received independent review.

The reserved test file is frozen before model predictions. It must not be used
for training or checkpoint selection. Its hash is in the benchmark manifest.

## Evaluate your own predictions

After installing the development build, use:

```sh
roguard-evaluate-review --expected expected.jsonl --predictions predictions.jsonl --output report.json
```

Expected rows:

```json
{"case_id":"example-1","language":"ro","kind":"conversational","gold":1,"pair_group":"scenario-1","invariant_group":"scenario-1-personal"}
```

Prediction rows:

```json
{"case_id":"example-1","language":"ro","predicted":null}
```

`predicted` must be integer `0`, integer `1`, or `null` for unresolved output.
Missing, unknown and duplicate case IDs are rejected. Add `--require-targets`
to exit nonzero unless all four language/kind groups meet the observed-count
accuracy targets. This is a prerequisite check, not automatic stable promotion. The report includes:

- Concern recall counting abstentions as unresolved concerns.
- Benign review-or-abstention rate and direct false reviews separately.
- Coverage, accuracy among decided cases, and strict accuracy over all cases.
- Fully correct paired scenarios and invariant groups, plus inconsistent groups.

This prevents a model from appearing accurate simply by declining difficult
cases. `training.evaluate_evidence_runtime_v8` produces these files from the
actual evidence API, including focal source references and interpretation
summaries. Its binary task is whether review was requested; uncertain attribute
predictions can request review. That is distinct from attribute accuracy.

## Uncertainty and explanations

```python
from roguard.review_evaluation import triage

result = triage(0.45, low=0.2, high=0.8, language="ro")
assert result["decision"] == "clarify"
assert result["permits_external_action"] is False
```

The thresholds above illustrate API usage; they are not recommended calibrated
cutoffs. Callers must validate thresholds for their model and use case. Scores
below `low` yield `no_signal`, scores at or above `high` yield `review`, and the
remaining interval yields `clarify`. Unsupported languages or invalid model
outputs cannot produce `no_signal`. No outcome establishes safety or grants
permission to notify someone.

Evidence API reports additionally contain `interpretation_summary`: a Romanian
or Ukrainian label, unresolved attributes, possible actions, and whether a
retained-model review floor changed the action. Existing code-point source
references remain available. These explanations describe model proposals; they
are not verified accounts of what happened to a person.

## Upstream references

The [Qwen model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
identifies the base as a 4B causal instruction model under Apache-2.0. This study
uses its separately pinned MLX quantization and the locally recorded
[MLX-LM](https://github.com/ml-explore/mlx-lm) version. General upstream benchmark
results are not evidence of Romanian/Ukrainian child-safety accuracy.

Studio displays the interpretation summary before expandable attribute details;
source-fragment links remain available. Reports and the new evaluation CLI use
UTF-8 explicitly for Romanian and Ukrainian text across platforms.

## Scoring correction and targeted continuation

The cached tokenizer inserts an empty reasoning block before completed assistant
answers. The first evaluator mistakenly scored immediately after the generation
header. That position was not the supervised class-token position for the adapter.
The bare-prefix baseline and the initial adapted family run are retained with a
correction record and are not used for the matched comparison.

The corrected scorer derives a common prefix from two constant completions, `0`
and `1`, and requires that they differ at exactly one known class token. It does
not use the gold label to construct the prompt. Float16 cached inference differed
from full inference by about 0.003 on a diagnostic row; float32 arithmetic reduced
the discrepancy to about 0.000004 on the two-row check. Packed base weights remain
quantized. Each run retains its own cache/full check and precision metadata.

The matched unadapted float32 model has 300/400 correct decisions (75%).
The corrected float32 step-200 pilot has 339/400 (84.75%), a 9.75 percentage-point
improvement on this development screen. It still fails the release targets:

| Group | Missed concerns / 50 | False reviews / 50 |
|---|---:|---:|
| Romanian formal | 19 | 3 |
| Romanian conversational | 5 | 1 |
| Ukrainian formal | 16 | 5 |
| Ukrainian conversational | 5 | 7 |

On family development, the same checkpoint misses 0/48 concern variants in each
language but falsely reviews 12/36 benign variants in each. Explicit denials
remain a major error source. These results fail the promotion criteria.

A registered continuation, `instruction-sft-02`, uses 800 balanced training rows,
including 112 original paired-family rows, a compact policy, and one complete
training pass. Its custom loss supervises only the class-token position. Template
markers, end markers and padded tokens no longer dilute the classification loss.
It computes vocabulary logits only at the supervised position rather than at
every input position. Prompt and data both change, so its matched comparison must
use an unadapted model with the same compact policy. The final test remains unused.

## Reproduce the adapter

Research adapters contain only LoRA deltas in a bounded JSON format, with model
identity and revision. They do not include base weights or executable objects.
The step-200 export was restored and all 112 arrays matched exactly. These are
experimental study artifacts, not approved runtime replacements.

```sh
python -m training.archive_instruction_adapter_v8 restore \
  --input research/v8-stabilization/instruction-sft-01/step200-adapter.json.gz \
  --output /tmp/roguard-study-adapter
```

Use the pinned local base snapshot and the recorded MLX environment. For the
second study, pass its `policy.txt` to `evaluate_instruction_sft_v8 --policy-file`.
The training runner is `training.train_instruction_labels_v8 --config ... --base ...`.

## Additional isolation and robustness checks

The 400-case screen has only 50 positives per language/kind group. It ranks and
diagnoses candidates; the definitive observed-count gate uses the full existing
development/calibration groups. A single miss on a small screen is not treated
as proof that a model must fail the larger registered gate.

An automated audit found two repeated benign focal strings in the original
family test, despite zero full-context overlap. The original is preserved.
`test-lexical-v2.jsonl` replaces those topic-reset questions with different benign
questions, before any final-test predictions. Its 168 rows have zero normalized
full-context or focal-text overlap with the 3,102-row union of both instruction
studies' recorded training pools. This is exact isolation, not proof of semantic
independence or absence from upstream pretraining.

A separate development challenge contains 278 rows covering lowercase text,
missing Romanian diacritics, limited authored typing errors, an English discourse
marker, and an English ordinary-task turn before native-language messages. Its
545 training-side counterparts remain separate and were not included in the
running second study. These are controlled perturbations, not a representative
sample of informal child language.

Instruction prompts now canonicalize Unicode to NFC while preserving original
input strings for references. None of the 400 screen inputs changes under NFC.
A 16-row batch check found no decision differences between batch size four and
recorded sequential scoring; the largest score difference was 0.0000132. This is
an implementation check, not a universal equivalence or speed guarantee.

## Completed class-token continuation

The 800-example continuation finished all 200 optimizer updates. Validation loss
was 0.431 initially, 0.391 at step 100, and 0.410 at step 200. Its best observed
checkpoint on the 400-case screen was step 100: 350/400 correct (87.5%), compared
with 341/400 (85.25%) for the unadapted model using the same compact policy.
Step 200 regressed to 339/400 (84.75%). These are inspected synthetic development
results, not independent final-test or real-world accuracy estimates.

| Checkpoint | RO formal miss / false | RO conversational miss / false | UK formal miss / false | UK conversational miss / false |
|---|---:|---:|---:|---:|
| Step 100 | 1 / 15 | 3 / 3 | 3 / 14 | 4 / 7 |
| Step 200 | 16 / 1 | 10 / 0 | 15 / 1 | 15 / 3 |

Each entry uses 50 positive and 50 negative cases. Neither checkpoint qualifies.
Retrospective threshold checks also cannot satisfy both targets on these rows;
no deployment threshold was selected from them. On paired-family development,
step 100 has RO 2/48 misses and 22/36 false reviews, and UK 0/48 misses and 12/36
false reviews. Step 200 has RO 6/48 misses and 10/36 false reviews, and UK 5/48
misses and 10/36 false reviews. Neither solves any complete seven-condition
paired scenario. The new adapters therefore remain research artifacts.

`training.analyze_instruction_errors_v8` checks prediction coverage and reports
errors by annotated context without printing messages. Its slices overlap and
are not independent statistical tests. The exported step-200 adapter restores
all 112 arrays exactly; training results, predictions, and export hashes are
retained with the study.

The installed-package Unicode smoke check also exercises `roguard-evaluate-review`
with bilingual identifiers and verifies that an unresolved benign prediction
returns a failing target-gate exit status.
