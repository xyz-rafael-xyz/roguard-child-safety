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
batch size two, gradient accumulation two, and assistant-only loss. Maximum
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
Missing, unknown and duplicate case IDs are rejected. The report includes:

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
