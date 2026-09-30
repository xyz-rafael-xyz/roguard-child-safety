# Ukrainian prompt-only model-family diagnostic on a consumed probe

**Registered before the three new inference runs.** This is a descriptive comparison on the already consumed, unreviewed `uk-contrast-0001` abstract-card probe. It is **not** an independent test, a Ukrainian training result, or evidence about child messages. It must not be used to claim that the best-looking model generalizes to new cards.

## Fixed inputs and comparison

Use all 48 frozen Ukrainian cards in their original order, preserving their reference labels. The source SHA-256 is `87e974bf932620f71159c8b4f17cb02dbc9fa2c1c145656771aa6cbc40b28ef2`. Correct only the seven known numeral–noun agreement errors in five A1 cards **in memory**, using the same substitutions as the [registered correction diagnostic](UK_NUMERAL_DIAGNOSTIC.md). No other card text, label, generator, manifest, or original prediction file changes. The corrected A1 diagnostic result SHA-256 is `9f12b6415acca383fe667e73d88bc70e9a4f9a1d15ba8556649ec93a1e3c4e9d`.

Score each corrected card once, without fitting, examples, tuning, or retries, with the unchanged [Ukrainian prompt](../eval/uk_prompt_probe.py) (SHA-256 `d35ea6cccd404ad49b5c011d7b4faddbdeb66cc78be71d80cfae34f7be6def50`), strict code parser, unchanged MLX generation defaults, and a 24-token cap. The [model registry](../eval/models.json) has SHA-256 `bc089718f74389a252eec9cd30d4a9ecbc7dd136b482f3536239e52a3775ac79`. Run exactly these already registered 4-bit MLX snapshots, in this order, in separate processes on the local 16 GB Apple Silicon machine:

1. `phi4_mini` — `mlx-community/Phi-4-mini-instruct-mlx-4Bit`, revision `d848c30f6d5419b9892433cf6b1062626d15340e`.
2. `gemma3_4b` — `mlx-community/gemma-3-text-4b-it-4bit`, revision `4f665a4c50ecfe4ecdc34056ab52fe3e3c4abf9e`.
3. `mistral7b` — `mlx-community/Mistral-7B-Instruct-v0.3-4bit`, revision `a4b8f870474b0eb527f466a03fbc187830d271f5`.

The prior Qwen3 result is a **historical comparator**: its original 40 non-A1 decisions plus the separately saved corrected A1 decisions. Do not rerun or silently replace it. The same cards informed the choice to run this diagnostic, so no four-model result is a fresh held-out selection.

## Predeclared report

Save raw model outputs, prompt hashes, strict parse flags, and code decisions. Report parsed cards, exact cards, complete negative-to-positive pairs, correct-direction flips, per-category TP/FP/FN/TN and separate positive/negative abstentions on applicable cards. Count a parse failure as an abstention and as a missed exact card/pair; do not turn malformed output into `NONE`. Show the Qwen historical comparator alongside the three new runs. Any observed gain is a hypothesis for a **future** independently authored, fluent-reviewed Ukrainian study. The Ukrainian taxonomy and training gate remain pending independent language and child-safety review.

## Saved result

| Prompt-only model | Parsed cards | Exact cards | Exact pairs | Correct-direction flips |
|---|---:|---:|---:|---:|
| Historical Qwen3 with corrected A1 decisions | 46/48 | 21/48 | 0/24 | 0/24 |
| Phi-4-mini | 16/48 | 4/48 | 0/24 | 0/24 |
| Gemma 3 4B | 48/48 | 20/48 | 0/24 | 0/24 |
| Mistral 7B v0.3 | 22/48 | 11/48 | 0/24 | 2/24 |

The three newly run models all failed to resolve a complete pair. Phi and Mistral often returned strings containing multiple listed answer choices, which the strict parser correctly rejected. Gemma parsed every card, but mostly chose the category named in the prompt on both sides; all four S1 positives were missed. A corrected numeral form alone did not fix Qwen's A1 behavior. These are failure modes of this prompt and consumed draft-card surface, not a ranking of Ukrainian language capability.

The [Phi](../eval/runs/uk-family-phi4-mini-corrected.json), [Gemma](../eval/runs/uk-family-gemma3-4b-corrected.json), and [Mistral](../eval/runs/uk-family-mistral7b-corrected.json) records contain raw outputs, exact corrected-prompt hashes, per-category TP/FP/FN/TN, and separate parse abstentions. The [verifier](../eval/verify_uk_family.py) replays decisions and summaries from those saved outputs and reconstructs the historical Qwen comparator. `mlx-lm` 0.31.3 was used for the new runs. No model was selected as validated; a changed prompt or model needs a future independent test for a generalization claim.
