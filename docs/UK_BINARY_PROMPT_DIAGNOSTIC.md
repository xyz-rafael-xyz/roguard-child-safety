# Ukrainian single-rule binary prompt diagnostic

**Registered before inference.** The first Ukrainian prompt-only [four-model comparison](UK_MODEL_FAMILY_DIAGNOSTIC.md) produced no complete contrast pair on the consumed draft probe. Phi and Mistral often copied a list of answer choices, while Gemma parsed all cards but largely chose the named category on both sides. This experiment changes the **output task**, not the data or weights: ask one category question at a time in Ukrainian and require exactly the single character `0` or `1`. It is a post hoc development diagnostic on the same consumed cards, not a new held-out accuracy result.

Use the same 48 frozen `uk-contrast-0001` cards with only the seven already documented Ukrainian numeral corrections applied in memory. The [source and correction hashes](UK_MODEL_FAMILY_DIAGNOSTIC.md) remain binding. No card, label, taxonomy, or model weight changes. The [runner](../eval/diagnose_uk_binary.py) asks exactly one question for each applicable category: D1, R1, P1, or G1 on their respective eight cards; both A1 and S1 separately on each of the 16 response cards. This is 64 tasks per model. The category rules come from the six draft Ukrainian definitions. There are no worked answers or few-shot examples in the prompt.

Run the same pinned registry revisions in the order `qwen3_4b`, `phi4_mini`, `gemma3_4b`, `mistral7b`, each in a separate process, with the existing MLX generator's default decoding and `max_tokens=8`. Parse a task only if the stripped output is exactly `0` or `1`. A card is parse-valid only if **every** applicable task parses. For any invalid card, preserve all raw task outputs, record a card-level abstention, and give it no category decision. No repairs, alternate parsers, retries, threshold changes, or examples are allowed after seeing outputs.

Predeclared diagnostics: task parse coverage (out of 64), card parse coverage (out of 48), exact cards, exact negative-to-positive pairs, correct-direction flips, and per-category TP/FP/FN/TN with positive and negative abstentions. Compare descriptively with the original list-output prompt. If binary output improves a consumed score, that is a **prompt-format hypothesis only**. Selection or an accuracy claim needs independent fluent review and newly authored, adjudicated Ukrainian abstract cards; the taxonomy and training gate remain pending.

## Saved result

| Model | Parsed tasks | Parsed cards | Exact cards | Exact pairs | Direction flips |
|---|---:|---:|---:|---:|---:|
| Qwen3 4B | 64/64 | 48/48 | 20/48 | 0/24 | 2/24 |
| Phi-4-mini | 64/64 | 48/48 | 23/48 | 4/24 | 7/24 |
| Gemma 3 4B | 64/64 | 48/48 | 26/48 | 5/24 | 7/24 |
| Mistral 7B v0.3 | 64/64 | 48/48 | 25/48 | 5/24 | 9/24 |

The changed output contract removed the observed parse failure for all four models. It did **not** yield reliable paired decisions. Qwen still marked both sides of every pair in most categories. Phi recovered the A1 rule on 4/4 positives with one false A1 review among 12 applicable negatives, but missed all four S1 positives and falsely reviewed seven S1 negatives. Mistral recognized all four S1 positives without an S1 false review but marked A1 on all 16 response cards. Gemma's strongest category was D1 at 3/4 positives and 0/4 false reviews; it still resolved only five complete pairs overall. These differences motivate category-specific hypotheses for a future study; choosing a combination by this consumed probe would overfit it.

The [Qwen](../eval/runs/uk-binary-qwen3-4b-corrected.json), [Phi](../eval/runs/uk-binary-phi4-mini-corrected.json), [Gemma](../eval/runs/uk-binary-gemma3-4b-corrected.json), and [Mistral](../eval/runs/uk-binary-mistral7b-corrected.json) records contain all 256 raw task outputs and prompt hashes. The [verifier](../eval/verify_uk_binary.py) replays every strict parse, card decision, pair result, and category count. All runs used `mlx-lm` 0.31.3. No model or prompt is promoted to a validated Ukrainian classifier.
