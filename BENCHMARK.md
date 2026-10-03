# RoGuard benchmark

## Romanian abstract-card pilot, 29 September 2026

The first local RoMistral adapter was trained for 200 steps on 88 approved Romanian training cards, with 40 separate development cards. The final step-200 checkpoint had the lowest logged validation loss (0.260). One frozen evaluation used the 40 approved cards in batch 0007. [The recorded predictions and counts](eval/runs/ro-v1.json) are the source for this table. All 40 outputs passed the strict code parser. Exact card-label match was **18/40 (45%)**. The base model is pinned to `OpenLLM-Ro/RoMistral-7b-Instruct` revision `fc097a5fdc2a5a689df084c76ffacf2cfb9f6550`; training used a 4-bit MLX conversion, rank 16, scale 2, eight adapter layers, batch size 1, learning rate `1e-5`, and prompt masking.

| Language | Category | Support | TP | FP | FN | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|---:|
| Romanian | D1 | 4 | 0 | 0 | 4 | undefined | 0% |
| Romanian | R1 | 4 | 4 | 4 | 0 | 50% | 100% |
| Romanian | A1 | 4 | 0 | 0 | 4 | undefined | 0% |
| Romanian | P1 | 4 | 0 | 0 | 4 | undefined | 0% |
| Romanian | G1 | 4 | 4 | 4 | 0 | 50% | 100% |
| Romanian | S1 | 5 | 0 | 0 | 5 | undefined | 0% |

“Undefined” means the model predicted no positives for that category; assigning it a 0% precision would misstate the denominator. It missed all positive `D1`, `A1`, `P1`, and `S1` cards. It recovered each positive `R1` and `G1` card, with four false positives in each. These are poor pilot results and do not support a high-accuracy claim. The model emits hard category codes; API values of 0 or 1 are **not calibrated probabilities**. No threshold calibration was performed.

The predictions reveal a type-only shortcut: all eight `routing_card` rows received `R1`, all eight `gate_card` rows received `G1`, and all 24 remaining rows received `NONE`. The model made no within-type distinction on this test. A [post-pilot development diagnosis](eval/runs/ro-v1-dev.json) found the same 18/40 exact-match count and category-level pattern. This is evidence that the pilot did not learn the intended category boundaries, even in its abstract-card domain.

The test cards are fictional, abstract descriptions of decisions. They contain no real or realistic child disclosure text. Their rule structures overlap training and development while their wording and symbolic values differ. Only four or five positive examples occur per category, so these numbers have high uncertainty even for this narrow format. They say nothing about detection of real disclosures, routing rights, Romanian child comprehension, or safe deployment. Structured `R1`, `P1`, and `G1` decisions still require the separate deterministic checks and human review. At this first pilot stage, the Ukrainian taxonomy and model had not been evaluated. No prior paper's numbers were substituted for RoGuard measurements.

## Romanian abstract-card challenge, batch 0008

After the type-only shortcut was found, a second adapter used a card-specific prompt on the **same 88 training and 40 development cards**. Batch 0008 was generated separately, [reproduced and validated automatically](data/synthetic/batch-0008.review.json), then opened once for this fixed comparison. The adapter made **23/40 exact matches (57.5%)** with 40/40 strict parses. The pinned, quantized RoMistral base model with the same prompt made **25/40 (62.5%)** with 38/40 strict parses. See the [adapter predictions](eval/runs/ro-v2-test-0008.json), [base predictions](eval/runs/romistral-base-v2-test-0008.json), and [run manifest](eval/runs/ro-v2-test-0008-manifest.json). Fine-tuning did not improve exact-match accuracy on this challenge.

| Category | Support | Adapter TP/FP/FN | Adapter precision | Adapter recall | Base TP/FP/FN |
|---|---:|---:|---:|---:|---:|
| D1 | 4 | 4/0/0 | 100% | 100% | 4/1/0 |
| R1 | 4 | 4/4/0 | 50% | 100% | 4/4/0 |
| A1 | 4 | 1/0/3 | 100% | 25% | 2/2/2 |
| P1 | 4 | 1/1/3 | 50% | 25% | 4/4/0 |
| G1 | 4 | 2/2/2 | 50% | 50% | 4/2/0 |
| S1 | 4 | 4/3/0 | 57.1% | 100% | 4/0/0 |

Four positive cards per category make these estimates unstable. `D1` here means recognizing a **description of an abstract pattern**, not a child's message. The generated challenge shares the taxonomy's decision structures with training. The automated audit verifies exact generation, hashes, label balance, and surface separation; it cannot validate authentic language use or practical child-safety outcomes. Batch 0008 is now consumed as a test set and must not be used for another model-selection claim. The toolkit's declared-contract checks have a different evidence basis: they evaluate caller-supplied structured facts against explicit rules, without granting access or performing external actions.

### Post-hoc comparator diagnostic

After seeing the batch-0008 results, I ran the pinned Qwen3-4B-Instruct-2507 4-bit base model with the same v2 Romanian prompt. [Its exploratory output](eval/runs/qwen3-base-v2-0008-exploratory.json) parsed 40/40 cards and matched 20/40 complete label sets. It predicted positives for all applicable `R1`, `A1`, `P1`, and `G1` cards, including every negative for each of those categories. This diagnostic was **not** fixed before the test was opened, so it is not a new held-out model-selection result. It reinforces the need for contrastive evaluation and independent data design before more fine-tuning. At this point no Ukrainian model had been measured; the later draft-taxonomy probe is reported below.

## Frozen one-factor diagnostic, contrast 0001

The [48-card contrast set](data/synthetic/contrast-0001.jsonl) contains 24 negative/positive pairs with source kind fixed and one described fact changed. The [protocol](docs/CONTRAST_PROTOCOL.md) and data were committed before these model runs. All models used the Romanian v2 prompt with a 24-token generation cap. Exact pair means **both** full label sets were correct. Correct-direction flip asks only whether the expected category changed from absent to present.

| Model | Strict parse | Exact cards | Exact pairs | Correct-direction flips |
|---|---:|---:|---:|---:|
| RoMistral v2 adapter | 48/48 | 28/48 | 6/24 | 6/24 |
| RoMistral prompt-only base | 48/48 | 25/48 | 2/24 | 2/24 |
| Qwen3 4B prompt-only base | 48/48 | 25/48 | 1/24 | 4/24 |

The [adapter](eval/runs/contrast-0001-ro-v2.json), [RoMistral base](eval/runs/contrast-0001-romistral-base.json), and [Qwen3 base](eval/runs/contrast-0001-qwen3-base.json) predictions show the main failure: the category often stays positive even when the decisive condition flips. The adapter's `S1` recall was 0/4; it also missed 3/4 `P1` positives. The RoMistral base marked all four `R1`, `P1`, and `G1` negatives positive. Qwen3 did the same for those categories. This set is consumed and cannot support selection of the later v3 adapter. It measures abstract rule descriptions, not real child language or a deployment operating point.

## Frozen Romanian paired-card comparison, batch 0011

The [v3 protocol](docs/V3_PROTOCOL.md), 288 training cards, 36 development cards, and 48 test cards were fixed before the final 600-step fit and this comparison. The final adapter's validation loss was 0.279 at step 600, up from 0.184 at step 400. The [comparison record](eval/runs/ro-v3-test-0011-comparison.json) binds exact train, development, test, prompt, adapter, and prediction hashes. Each comparator used the same v3 rule prompt and 24-token cap. Exact pairs require both negative and positive rows to have the complete correct label set; a correct-direction flip requires the category to appear only after its decisive fact changes.

| Model | Strict parse | Exact cards | Exact pairs | Correct flips | False review on 24 negatives |
|---|---:|---:|---:|---:|---:|
| RoMistral v3 adapter | 48/48 | 25/48 | 2/24 | 2/24 | 1/24 |
| RoMistral base | 48/48 | 20/48 | 0/24 | 0/24 | 24/24 |
| Qwen3 4B base | 48/48 | 24/48 | 1/24 | 2/24 | 22/24 |
| Phi-4 mini base | 38/48 | 23/48 | 3/24 | 3/24 | 15/24 |
| Gemma 3 4B base | 48/48 | 25/48 | 2/24 | 2/24 | 20/24 |
| Mistral 7B v0.3 base | 32/48 | 22/48 | 0/24 | 4/24 | 14/24 |
| Always review reference | 48/48 | 24/48 | 0/24 | 0/24 | 24/24 |

The adapter learned an almost-always-`NONE` shortcut. It had **zero recall** for five categories and only two true `S1` positives. The prompt-only models generally showed the opposite shortcut: high apparent positive recall accompanied by many false reviews. The adapter reached only 19/36 exact cards on the development split after fitting, with the same five categories at zero recall. This is a model failure on the abstract task, not a safety threshold or useful operating point.

| Romanian adapter category | Support | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 4 | 0 | 0 | 4 | undefined | 0% |
| R1 | 4 | 0 | 0 | 4 | undefined | 0% |
| A1 | 4 | 0 | 0 | 4 | undefined | 0% |
| P1 | 4 | 0 | 0 | 4 | undefined | 0% |
| G1 | 4 | 0 | 0 | 4 | undefined | 0% |
| S1 | 4 | 2 | 1 | 2 | 66.7% | 50% |

The test is now consumed. Category support is only four positives each and the cards are symbolic rule descriptions. The `D1` rows do not contain child utterances. These results cannot estimate performance on children, natural Romanian, or real routing decisions.

## Ukrainian draft-taxonomy prompt-only probe

The separately frozen [Ukrainian probe](docs/UK_PROBE_PROTOCOL.md) used 48 Ukrainian symbolic cards in 24 one-factor pairs and a pinned, prompt-only Qwen3 4B base. It parsed 46/48 outputs, matched **21/48** card labels, **0/24** exact pairs, and **0/24** correct-direction flips. See the [predictions and metrics](eval/runs/uk-contrast-0001-qwen3-base.json). The model largely flagged both sides of a pair. This is an exploratory measurement under an unreviewed Ukrainian taxonomy, not a Ukrainian trained classifier.

A [post hoc grammar audit](docs/UK_NUMERAL_DIAGNOSTIC.md) found seven numeral agreement errors in five of the eight A1 cards. Correcting only those phrases in memory and rerunning the same pinned Qwen model left every A1 decision unchanged: **4/8 exact cards, 0/4 exact pairs**. The original probe and its reported metrics remain frozen. Neither the corrected diagnostic nor the original draft set establishes Ukrainian language validity.

A separately registered [post hoc Ukrainian model-family diagnostic](docs/UK_MODEL_FAMILY_DIAGNOSTIC.md) scored the same consumed 48 cards, with only those seven in-memory grammar corrections, using three more pinned prompt-only bases. Phi-4-mini parsed 16/48 cards and matched 4/48, Gemma 3 parsed 48/48 and matched 20/48, and Mistral 7B parsed 22/48 and matched 11/48. **All three got 0/24 exact pairs**; Mistral had 2/24 correct-direction flips without an exact pair. The historical Qwen comparator with corrected A1 decisions remained at 21/48 cards and 0/24 pairs. The [saved raw outputs and replay verifier](eval/verify_uk_family.py) preserve strict parse and category-level errors. This is failure analysis on a consumed, unreviewed probe, not model selection or new held-out Ukrainian accuracy.

A targeted [single-rule binary-prompt diagnostic](docs/UK_BINARY_PROMPT_DIAGNOSTIC.md) then removed the format failure: Qwen, Phi, Gemma, and Mistral each parsed **64/64 category tasks and 48/48 cards**. Complete pairs remained weak on the same consumed cards: **0/24**, **4/24**, **5/24**, and **5/24**, respectively. Phi's A1 count judgments improved while its S1 judgments failed; Mistral had the opposite pattern. The [saved raw outputs and replay](eval/verify_uk_binary.py) preserve these errors. This result cannot select a Ukrainian model or establish held-out accuracy.

| Ukrainian category | Support | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 4 | 4 | 4 | 0 | 50% | 100% |
| R1 | 4 | 4 | 4 | 0 | 50% | 100% |
| A1 | 4 | 4 | 9 | 0 | 30.8% | 100% |
| P1 | 4 | 4 | 4 | 0 | 50% | 100% |
| G1 | 4 | 4 | 4 | 0 | 50% | 100% |
| S1 | 4 | 0 | 0 | 4 | undefined | 0% |

## Frozen Romanian binary paired-card comparison, batch 0012

The [v4 protocol](docs/V4_PROTOCOL.md) changed the model task to one strict `da`/`nu` decision for each applicable category. Its 480 training tasks were balanced by repeating positive A1 and S1 cards; the 48 development tasks were not repeated. The first uppercase-target fit was stopped after 60 steps when a tokenizer check found unequal target lengths; no test inference was run. The restarted fit saved six checkpoints. The [development-only record](eval/runs/ro-v4-dev-selection.json) selected step **500** by the frozen rule: 3/18 exact development pairs, 20/36 exact cards, and 9/18 false reviews. The selected weight hash was committed before batch 0012 was opened.

The [frozen comparison](eval/runs/ro-v4-test-0012-comparison.json) binds the selected weight, development choice, train and development exports, new test, prompt, and all six prediction files. The five prompt-only baselines used the same binary prompt and an eight-token cap per category task. An unparsed row counts as no predicted labels and remains visible in the parse column.

| Model | Strict parse | Exact cards | Exact pairs | Correct flips | False review on 24 negatives |
|---|---:|---:|---:|---:|---:|
| RoMistral v4 adapter, selected step 500 | 48/48 | 28/48 | 4/24 | 5/24 | 3/24 |
| RoMistral base | 45/48 | 27/48 | 6/24 | 8/24 | 12/24 |
| Qwen3 4B base | 48/48 | 29/48 | 5/24 | 5/24 | 17/24 |
| Phi-4 mini base | 47/48 | 17/48 | 0/24 | 1/24 | 23/24 |
| Gemma 3 4B base | 35/48 | 26/48 | 2/24 | 2/24 | 7/24 |
| Mistral 7B v0.3 base | 0/48 | 24/48 | 0/24 | 0/24 | 0/24 |
| Always review reference | 48/48 | 24/48 | 0/24 | 0/24 | 24/24 |

The adapter detected 8/24 positive cards and missed **all** P1, G1, and S1 positives. Its small false-review count cannot compensate for those misses. Qwen3 detected 22/24 positives but also reviewed 17/24 negatives. Mistral's 24 exact cards came entirely from treating its 48 malformed outputs as empty predictions; it is not a useful 50% classifier. The tests for v3 and v4 use different cards, so their scores do not establish a clean gain from the binary intervention.

| Romanian v4 adapter category | Support | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 4 | 2 | 0 | 2 | 100% | 50% |
| R1 | 4 | 2 | 2 | 2 | 50% | 50% |
| A1 | 4 | 4 | 1 | 0 | 80% | 100% |
| P1 | 4 | 0 | 0 | 4 | undefined | 0% |
| G1 | 4 | 0 | 0 | 4 | undefined | 0% |
| S1 | 4 | 0 | 1 | 4 | 0% | 0% |

Batch 0012 is consumed. The cards describe imaginary rules, not child messages or assistant replies. Four positives per category, repeated training facts, and rule overlap cannot support a real-world child-safety accuracy claim. The deterministic contract checker remains the usable route for complete declared R1, P1, G1, and word-cap A1 facts; its output is still a suggestion for human review, never an external action.

## Frozen Romanian token-margin comparison, batch 0013

The [v5 protocol](docs/V5_PROTOCOL.md), [96-card symbolic test](data/synthetic/batch-0013.preview.md), and evaluator were committed before development calibration or test inference. The new cards form 48 adjacent one-condition pairs, eight per category. The [surface audit](data/synthetic/batch-0013.audit.json) found zero exact text duplicates and maximum same-kind character similarity 0.583 to earlier batches; the underlying policy rules still overlap. No card contains a child utterance or realistic assistant reply.

V5 kept the selected v4 step-500 adapter and prompt fixed. It read the next-token logits for lowercase `da` and `nu` as a two-choice score, then fitted six thresholds on **batch-0010 development cards only**. The [committed threshold record](eval/runs/ro-v5-dev-thresholds.json) shows 29/36 exact development cards and 11/18 exact development pairs. A leave-one-pair-out diagnostic dropped to 24/36 cards and 8/18 pairs; P1 and G1 each recalled only one of three development positives. The threshold record was committed at `0d0a2b6` before batch 0013 was opened. The score is **not a calibrated child-risk probability**.

The [frozen comparison](eval/runs/ro-v5-test-0013-comparison.json) binds test, attestation, threshold, and prediction hashes. The comparators were the same adapter with its original hard generation and the pinned prompt-only RoMistral with the same category prompt. An unparsed generation counts as no labels and remains visible in the parse column. The always-review and never-review references each get 48/96 cards and 0/48 pairs.

| Model | Strict parse | Exact cards | Exact pairs | Correct flips | False review on 48 negatives |
|---|---:|---:|---:|---:|---:|
| V5 token-margin adapter | 96/96 | 50/96 | 2/48 | 2/48 | 6/48 |
| V4 hard-output adapter, same weights | 96/96 | 56/96 | 8/48 | 8/48 | 1/48 |
| Prompt-only RoMistral | 89/96 | 47/96 | 9/48 | 11/48 | 31/48 |

Every category has eight positive cards. `TP/FP/FN` gives raw counts; `P/R` gives precision and recall. Undefined precision means the model predicted no positives for that category.

| Category | V5 TP/FP/FN | V5 P/R | Hard adapter TP/FP/FN | Hard P/R | Base TP/FP/FN | Base P/R |
|---|---:|---:|---:|---:|---:|---:|
| D1 | 0/0/8 | undefined / 0% | 0/0/8 | undefined / 0% | 6/2/2 | 75% / 75% |
| R1 | 0/0/8 | undefined / 0% | 1/0/7 | 100% / 12.5% | 8/8/0 | 50% / 100% |
| A1 | 8/6/0 | 57.1% / 100% | 8/1/0 | 88.9% / 100% | 6/9/2 | 40% / 75% |
| P1 | 0/0/8 | undefined / 0% | 0/0/8 | undefined / 0% | 3/6/5 | 33.3% / 37.5% |
| G1 | 0/0/8 | undefined / 0% | 0/0/8 | undefined / 0% | 6/5/2 | 54.5% / 75% |
| S1 | 0/0/8 | undefined / 0% | 0/0/8 | undefined / 0% | 5/11/3 | 31.3% / 62.5% |

The preregistered success rule required more exact pairs than **both** comparators, at least four of eight positives recalled in every category, and at most two false reviews per category. V5 failed. The development gain did not transfer to the new card surface: V5 detected only A1 positives and had six A1 false positives. The hard adapter mostly abstained from review; the prompt-only base over-reviewed negatives. The [model-agnostic pair evaluator](docs/CLI.md#compare-a-model-on-abstract-contrast-pairs) can recompute these diagnostics from saved predictions. Batch 0013 is consumed and cannot be used to select a revised threshold or model. None of these results estimates detection on real child language, safe routing, or deployed child-safety accuracy.

## Frozen Romanian compositional comparison, batch 0016

The [v6 protocol](docs/V6_PROTOCOL.md) fixed 576 training cards, 96 development cards, and a new 144-card test before fitting. Every adjacent negative/positive pair changes one declared fact; the [generator](training/generate_compositional.py) checks its label against a typed rule. The [attestation](data/synthetic/batch-0016.review.json) and [surface audit](data/synthetic/batch-0016.audit.json) bind the exact test bytes. The development-only [selection record](eval/runs/ro-v6-dev-selection.json) chose step 1200 because its 10/48 exact pairs exceeded steps 300, 600, and 900 (6, 0, and 6); it still falsely flagged 32/48 negative development cards. The selected weight hash was committed at `56b456d` before test inference.

The [frozen comparison](eval/runs/ro-v6-test-0016-comparison.json) verifies test, prompt, selection, model, and prediction hashes. All four runs used the same v4 category-specific binary prompt. A complete pair requires both cards to have their exact full label sets.

| Model | Strict parse | Exact cards | Exact pairs | Correct flips | False review on 72 negatives |
|---|---:|---:|---:|---:|---:|
| RoMistral v6 adapter, selected step 1200 | 144/144 | 78/144 | **9/72** | 9/72 | 39/72 |
| RoMistral v4 adapter, selected step 500 | 144/144 | 84/144 | **14/72** | 14/72 | 22/72 |
| Prompt-only RoMistral | 144/144 | 66/144 | **6/72** | 7/72 | 56/72 |
| Prompt-only Qwen3 4B | 144/144 | 66/144 | **3/72** | 7/72 | 65/72 |
| Always review reference | 144/144 | 72/144 | 0/72 | 0/72 | 72/72 |
| Never review reference | 144/144 | 72/144 | 0/72 | 0/72 | 0/72 |

Every category has 12 positive cards. This table reports the selected v6 adapter's category errors; undefined precision means it predicted no positives.

| Category | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| D1 | 12 | 12 | 0 | 50% | 100% |
| R1 | 1 | 0 | 11 | 100% | 8.3% |
| A1 | 11 | 6 | 1 | 64.7% | 91.7% |
| P1 | 0 | 0 | 12 | undefined | 0% |
| G1 | 9 | 12 | 3 | 42.9% | 75% |
| S1 | 12 | 9 | 0 | 57.1% | 100% |

V6 **failed** its preregistered symbolic-card target of at least 58/72 exact pairs, at least 9/12 positive recalls in every category, at most 7/72 false reviews, and more exact pairs than all three comparators. It was worse than the prior v4 adapter on this fresh test. Strong positive recall for D1 and S1 came with many false reviews; P1 was entirely missed. Lower training loss and more examples did not yield reliable one-fact discrimination. Batch 0016 is consumed and cannot be used for prompt, threshold, or checkpoint selection. These are synthetic state descriptions, never child messages or realistic response text; the numbers do not estimate live child-safety accuracy.

## Frozen Romanian Qwen transfer comparison, batch 0017

The [v7 protocol](docs/V7_PROTOCOL.md) pinned the Qwen3 4B 4-bit snapshot, the same 576 training and 96 development cards used by v6, and an independently generated 144-card test. All four saved Qwen checkpoints failed the literal `da`/`nu` output contract on every development card. The [development selection](eval/runs/ro-qwen-v7-dev-selection.json) therefore chose step 300 by its fixed tie rule and was committed at `cb85d61` before test inference. A raw development generation showed an empty `<think>` block followed by `nu`; the adapter learned a format the literal parser rejected. The original parser remained fixed for this test.

The [frozen comparison](eval/runs/ro-qwen-v7-test-0017-comparison.json) binds test, attestation, prompt, selected weights, and all prediction files. An unparsed card counts as no labels while parse failure is reported separately. All three systems saw the same v4 prompt and eight-token cap.

| Model | Parsed cards | Exact cards | Exact pairs | Correct flips | False reviews on 72 negatives |
|---|---:|---:|---:|---:|---:|
| Qwen v7 adapter, selected step 300 | 0/144 | 72/144 | 0/72 | 0/72 | 0/72 |
| RoMistral v6 adapter | 144/144 | 85/144 | 17/72 | 17/72 | 14/72 |
| Prompt-only Qwen3 4B | 144/144 | 76/144 | 5/72 | 5/72 | 65/72 |
| Always review reference | 144/144 | 72/144 | 0/72 | 0/72 | 72/72 |
| Never review reference | 144/144 | 72/144 | 0/72 | 0/72 | 0/72 |

The Qwen v7 adapter **failed** the preregistered symbolic-card target. Its 72 exact cards are the empty-label negatives produced by parse failure, not evidence of useful classification; it recalled zero of the 12 positives in every category. The v6 comparator improved from 9/72 pairs on batch 0016 to 17/72 on this different batch, but still missed all P1 positives and fell far short of the target. Prompt-only Qwen over-reviewed 65 negatives. The learned systems remain research artifacts. Batch 0017 is consumed and cannot be used for selecting the parser repair. A [prospective v8 amendment](docs/V8_PROTOCOL.md#prospective-decoder-amendment-before-v8-fitting-or-batch-0018-inference) registered a narrow empty-wrapper parser after the v7 development diagnosis and before any batch-0018 inference.

## Frozen Romanian balanced Qwen comparison, batch 0018

The [v8 protocol and prospective amendment](docs/V8_PROTOCOL.md) registered two changes before the new test: equal training-task exposure across categories and a strict parser that also accepts Qwen's empty reasoning wrapper. The v7 comparator was reselected from its **unchanged** weights on batch 0015 with the same parser. Its [selection record](eval/runs/ro-qwen-v7-wrapper-dev-selection.json) chose step 300, 14/48 development pairs, and 18/48 false reviews. The [v8 development record](eval/runs/ro-qwen-v8-dev-selection.json) chose step 1200, 18/48 pairs, and 16/48 false reviews. Both records and selected-weight hashes were committed before any batch-0018 inference. The [attested test](data/synthetic/batch-0018.review.json) contains 144 symbolic cards in 72 one-fact pairs, 12 per category; its exact bytes and generator were validated before use.

The [frozen comparison](eval/runs/ro-qwen-v8-test-0018-comparison.json) verifies the test, prompt, parser, chat template, base snapshot, selections, weights, and prediction files. All three systems used the same v4 prompt, eight-token cap, and amended parser.

| Model | Parsed cards | Exact cards | Exact pairs | Correct flips | False reviews on 72 negatives |
|---|---:|---:|---:|---:|---:|
| Balanced Qwen v8, selected step 1200 | 144/144 | 80/144 | **13/72** | 13/72 | 18/72 |
| Original-exposure Qwen v7, parser-matched step 300 | 144/144 | 85/144 | **15/72** | 15/72 | 14/72 |
| Prompt-only Qwen3 4B | 144/144 | 79/144 | **9/72** | 9/72 | 59/72 |
| Always review reference | 144/144 | 72/144 | 0/72 | 0/72 | 72/72 |
| Never review reference | 144/144 | 72/144 | 0/72 | 0/72 | 0/72 |

Every category has 12 positive cards. This is v8's category error profile:

| Category | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| D1 | 4 | 4 | 8 | 50% | 33.3% |
| R1 | 5 | 3 | 7 | 62.5% | 41.7% |
| A1 | 7 | 1 | 5 | 87.5% | 58.3% |
| P1 | 2 | 1 | 10 | 66.7% | 16.7% |
| G1 | 8 | 9 | 4 | 47.1% | 66.7% |
| S1 | 0 | 0 | 12 | undefined | 0% |

The v8 adapter **failed** the registered symbolic-card target of at least 58/72 complete pairs, at least 9/12 positive recalls in every category, at most 7/72 false reviews, and more pairs than both comparators. The parser repaired output validity, but equal category exposure did not improve fresh-test pair accuracy over the matched v7 adapter. On the same 72 pairs, v8 alone got 3 right and v7 alone got 5 right; the two-sided exact McNemar probability is 0.727, a descriptive check that does not support a meaningful difference here. V8 and v7 both missed all S1 positives. Prompt-only Qwen recalled more positives but over-reviewed 59 negatives. Batch 0018 is consumed and cannot be used for another selection. These invented state descriptions do not measure authentic child-language understanding, child-safety accuracy, or Ukrainian-language performance.

## Frozen Romanian pair-trained encoder comparison, batch 0019

The original [v9 full-weight fit](docs/V9_PROTOCOL.md) was interrupted under local memory pressure after one complete epoch and part of a second; it never opened the test. A [prospective pretest amendment](docs/V9A_AMENDMENT.md) changed to a rank-16 mmBERT LoRA adapter and a one-fact pair ranking loss. Its first launch caught an exposure-count mistake before any optimizer step; the corrected five-epoch fit and both failures remain documented. The [development selection](eval/runs/ro-mmbert-v9a-dev-selection.json), committed at `411eefa` before test inference, chose epoch 5 and one shared threshold `0.2561211958527565` on batch 0015. It reached 35/48 exact development pairs and 81/96 exact cards. These softmax scores are not calibrated child-risk probabilities.

The [frozen comparison](eval/runs/ro-mmbert-v9a-test-0019-comparison.json) binds the [attested test](data/synthetic/batch-0019.review.json), selected adapter hash, threshold, raw predictions, and the unchanged v8 Qwen comparator. Both models used the same Romanian v4 category prompt on the new 144-card symbolic holdout. Every category has 12 positive cards. A complete pair requires the exact full label set on both sides.

| Model | Valid cards | Exact cards | Exact pairs | Direction flips | False reviews on 72 negatives |
|---|---:|---:|---:|---:|---:|
| mmBERT v9a adapter | 144/144 | 88/144 | **28/72** | 44/72 | 26/72 |
| Qwen v8 adapter, unchanged | 144/144 | 92/144 | **22/72** | 22/72 | 21/72 |
| Always review reference | 144/144 | 72/144 | 0/72 | 0/72 | 72/72 |
| Never review reference | 144/144 | 72/144 | 0/72 | 0/72 | 0/72 |

| Category | v9a TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| D1 | 11 | 3 | 1 | 78.6% | 91.7% |
| R1 | 12 | 0 | 0 | 100% | 100% |
| A1 | 12 | 20 | 0 | 37.5% | 100% |
| P1 | 10 | 8 | 2 | 55.6% | 83.3% |
| G1 | 9 | 7 | 3 | 56.3% | 75% |
| S1 | 12 | 20 | 0 | 37.5% | 100% |

V9a **failed** the preregistered 58/72-pair, per-category recall, and false-review targets. On the same pairs, v9a alone got 20 right and Qwen alone got 14 right; the descriptive two-sided exact McNemar probability is 0.392. There is no strong paired evidence of a difference from this small symbolic test. A1 and S1 tended to fire on each other's response cards because pair-only training omitted those cross-category negatives. Batch 0019 is consumed. The [v10 protocol](docs/V10_PROTOCOL.md) registers that targeted correction on a new test before fitting; no batch-0019 threshold or checkpoint will be retuned. None of these results estimates accuracy on authentic child language.

## Frozen Romanian cross-category encoder comparison, batch 0020

The [v10 protocol](docs/V10_PROTOCOL.md) added negative A1 tasks on S1 response cards and negative S1 tasks on A1 response cards to a pairwise LoRA fit. Its [development selection](eval/runs/ro-mmbert-v10-dev-selection.json), committed at `3d1ae66` before test inference, chose epoch 2 and one shared threshold `0.391207292675972` after 45/48 exact development pairs, 92/96 exact cards, and 3/48 false reviews on negative development cards. The [fresh test](data/synthetic/batch-0020.review.json) uses a new Romanian field vocabulary and Boolean markers. Its [frozen comparison](eval/runs/ro-mmbert-v10-test-0020-comparison.json) verifies selected weights, thresholds, test, attestation, predictions, and the unchanged v9a and Qwen comparators.

| Model | Valid cards | Exact cards | Exact pairs | Direction flips | False reviews on 72 negatives |
|---|---:|---:|---:|---:|---:|
| mmBERT v10 cross-category adapter | 144/144 | 99/144 | **28/72** | 28/72 | 40/72 |
| mmBERT v9a adapter, unchanged | 144/144 | 86/144 | **24/72** | 40/72 | 23/72 |
| Qwen v8 adapter, unchanged | 144/144 | 86/144 | **17/72** | 17/72 | 18/72 |
| Always review reference | 144/144 | 72/144 | 0/72 | 0/72 | 72/72 |
| Never review reference | 144/144 | 72/144 | 0/72 | 0/72 | 0/72 |

| Category | v10 TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| D1 | 12 | 7 | 0 | 63.2% | 100% |
| R1 | 12 | 0 | 0 | 100% | 100% |
| A1 | 9 | 9 | 3 | 50% | 75% |
| P1 | 10 | 6 | 2 | 62.5% | 83.3% |
| G1 | 12 | 12 | 0 | 50% | 100% |
| S1 | 12 | 6 | 0 | 66.7% | 100% |

V10 **failed** its registered symbolic-card criterion. The development improvement did not transfer: its false reviews rose from 3/48 development negatives to 40/72 test negatives. On the same test, v10 alone got nine pairs right that v9a missed and v9a alone got five; the descriptive two-sided exact McNemar probability is 0.424. Against Qwen those counts were 25 and 14, probability 0.108. Neither paired check rescues the failure. A1 ordered only 5/12 test pairs in the correct score direction when arbitrary word counts and field wording changed; G1 scored 12/12 positives but also flagged 12 false positives. Batch 0020 is consumed and will not select another threshold or checkpoint. This result measures only invented state descriptions, not a child's language or a real workflow.

## Frozen Romanian surface-generalization comparison, batch 0023

The [v11 protocol](docs/V11_PROTOCOL.md) generated 2,304 training cards by rendering 288 typed state pairs four ways, plus 144 development cards and a separately rendered 144-card test. Training incorporated *vocabulary families* diagnosed in consumed tests 0018–0020, never their rows or labels. The development choice, committed at `02e7653`, selected epoch 3 and shared threshold `0.9835666418075562` after 54/72 exact development pairs, 126/144 exact cards, and 5/72 false reviews. A [pretest metadata amendment](docs/V11_PRETEST_AMENDMENT.md) records a saved development-file batch tag error caught by the test gate before any test inference; approved batch-0022 row IDs, labels, metric recomputation, selected weights, and their hashes were verified without changing the fit or threshold. The [frozen comparison](eval/runs/ro-mmbert-v11-test-0023-comparison.json) binds all three encoder runs to the same [attested batch 0023](data/synthetic/batch-0023.review.json).

| Model | Valid cards | Exact cards | Exact pairs | Direction flips | False reviews on 72 negatives |
|---|---:|---:|---:|---:|---:|
| mmBERT v11 broad-surface adapter | 144/144 | 117/144 | **47/72** | 47/72 | 8/72 |
| mmBERT v10 adapter, unchanged | 144/144 | 121/144 | **53/72** | 53/72 | 10/72 |
| mmBERT v9a adapter, unchanged | 144/144 | 90/144 | **22/72** | 35/72 | 23/72 |
| Always review reference | 144/144 | 72/144 | 0/72 | 0/72 | 72/72 |
| Never review reference | 144/144 | 72/144 | 0/72 | 0/72 | 0/72 |

| Category | v11 TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| D1 | 9 | 3 | 3 | 75% | 75% |
| R1 | 12 | 0 | 0 | 100% | 100% |
| A1 | 0 | 0 | 12 | undefined | 0% |
| P1 | 8 | 2 | 4 | 80% | 66.7% |
| G1 | 12 | 3 | 0 | 80% | 100% |
| S1 | 12 | 0 | 0 | 100% | 100% |

V11 **failed** the fixed ≥58/72 pairs, ≥9/12 recall in every category, ≤7/72 false reviews, and superiority over both comparators. It selected a high shared cutoff and missed all A1 positives. On these same pairs v11 alone got four right and v10 alone got ten; the descriptive two-sided exact McNemar probability is 0.180. Relative to v9a, v11 alone got 25 and v9a alone got none; this establishes a difference on this synthetic sample, not deployment validity. The older v10 adapter reached 53/72 pairs on batch 0023, above its own batch-0020 result but still missed seven A1 and six G1 positives and exceeded the false-review target. The contrast between development and test across all three studies shows that wording and number comparison remain unresolved. Batch 0023 is consumed. Neither model is a validated child-language detector.

## Frozen Romanian category-threshold comparison, batch 0024

The [v12 protocol](docs/V12_PROTOCOL.md) kept v11 epoch-3 weights and the prompt fixed, then selected one cutoff per category on **batch 0022 development scores only**. The byte-identical [development score copy](eval/runs/ro-mmbert-v11-epoch-3-dev.json) retains the documented v11 metadata-tag error; row IDs, labels, metrics, and the original hash were checked before selection. The [committed threshold record](eval/runs/ro-mmbert-v12-dev-selection.json) chose D1/R1/A1/S1 at 0.5, P1 at 0.9843, and G1 at 0.99975. It reached 60/72 complete development pairs, 132/144 cards, and 6/72 false reviews. That apparent gain met the registered development gate, then failed on the independently rendered [attested batch 0024](data/synthetic/batch-0024.review.json), whose maximum nearest same-kind character similarity to prior batches was 0.50.

The [frozen row scores and comparison](eval/runs/ro-mmbert-v12-test-0024.json) apply both cutoff policies to **the same v11 inference**. [Independent metric recomputation](eval/verify_mmbert_v12.py) verifies the saved decisions and reports. Every category has 12 positive test cards.

| Cutoff policy | Valid cards | Exact cards | Exact pairs | False reviews on 72 negatives |
|---|---:|---:|---:|---:|
| V12 per-category development choice | 144/144 | 117/144 | **46/72** | 7/72 |
| Frozen v11 shared cutoff | 144/144 | 128/144 | **56/72** | 2/72 |

| Category | V12 TP/FP/FN | V12 precision/recall | V11 TP/FP/FN |
|---|---:|---:|---:|
| D1 | 12/6/0 | 66.7% / 100% | 12/0/0 |
| R1 | 12/0/0 | 100% / 100% | 12/0/0 |
| A1 | 4/1/8 | 80% / 33.3% | 0/0/12 |
| P1 | 10/0/2 | 100% / 83.3% | 10/0/2 |
| G1 | 2/0/10 | 100% / 16.7% | 12/2/0 |
| S1 | 12/0/0 | 100% / 100% | 12/0/0 |

V12 **failed** the registered ≥58/72 pair, ≥9/12 recall in each category, ≤7 false reviews, and improvement over the unchanged cutoff. It rescued four A1 positives but lost ten G1 positives and added six D1 false reviews. V12 alone got four pairs right that v11 missed; v11 alone got 14. The descriptive paired exact McNemar probability is 0.031 on this synthetic sample, favoring the frozen cutoff, not the proposed calibration. Batch 0024 is consumed and cannot select another cutoff. The v11 shared cutoff's 56/72 pairs on this new surface also failed the six-category recall target because A1 remained 0/12. Neither policy establishes child-language accuracy.

## Frozen Romanian D1/S1 advisory transfer, batch 0025

The [v13 protocol](docs/V13_PROTOCOL.md) narrowed the research question before opening a new test. It kept the committed v11 adapter, prompt, and high shared cutoff unchanged, and evaluated only symbolic D1 messages and S1 response-field cards. The [attested batch 0025](data/synthetic/batch-0025.review.json) has 48 adjacent one-fact pairs, 24 per category, with a separate Romanian renderer; its maximum nearest same-kind character similarity to prior batches was 0.449. The [frozen predictions](eval/runs/ro-mmbert-v13-test-0025.json) and [recomputation script](eval/verify_mmbert_v13.py) bind scores and decisions to the adapter and test hashes.

| Frozen v11 adapter on batch 0025 | Result |
|---|---:|
| Valid cards | 96/96 |
| Exact cards | 95/96 |
| Complete exact pairs | **47/48** |
| False reviews on 48 negative cards | **0/48** |
| D1 true positives / false positives / missed positives | 24 / 0 / 0 |
| S1 true positives / false positives / missed positives | 23 / 0 / 1 |

V13 **passed its preregistered narrow symbolic target** of at least 42/48 complete pairs, at least 21/24 positives in both categories, and at most four false reviews. Always-review and never-review references each get zero complete pairs. The model missed one S1 positive and made no false review on this generated sample. The result supports using this adapter as a research advisory scorer on **abstract pattern and response-field descriptions** alongside exact declared-contract checks. It does **not** validate D1 recognition in a child's natural Romanian, S1 judgment on a real response, Ukrainian text, or any live safety decision. Batch 0025 is consumed and cannot select another cutoff.

## Frozen Romanian mixed-input workflow, batch 0026

The [v14 protocol](docs/V14_PROTOCOL.md) was committed before the first inference on [attested batch 0026](data/synthetic/batch-0026.review.json): 144 new Romanian abstract cards in 72 one-fact pairs, 12 per category. The unchanged, committed v11 adapter and cutoff scored all card text. For the combined workflow, D1 and S1 used those model decisions; R1, A1, P1, and G1 instead used separately supplied, typed [contract sidecars](data/synthetic/batch-0026.contracts.jsonl) through the read-only checker. The generated sidecars are **privileged input** that the text-only comparator did not receive. The exact symbolic labels were fixed by a separate generator rule before calling the checker. [Raw scores and decisions](eval/runs/ro-mmbert-v14-test-0026.json) and an [independent recomputation](eval/verify_mmbert_v14.py) bind the test, sidecars, protocol, and adapter hashes.

| Input condition on the same cards | Exact cards | Complete pairs | Direction flips | False reviews on 72 negatives |
|---|---:|---:|---:|---:|
| D1/S1 model plus R1/A1/P1/G1 declared contracts | **144/144** | **72/72** | **72/72** | **0/72** |
| Frozen v11 text-only model | 127/144 | 55/72 | 55/72 | 2/72 |

The combined result splits into **24/24 D1/S1 advisory pairs** and **48/48 declared-contract pairs**. It passed the preregistered mixed-input synthetic target of at least 66/72 pairs, at least 18/24 advisory pairs, all 48 contract pairs, at least 9/12 positives in each category, and at most four false reviews. On these same pairs the combined workflow alone got 17 right and the text-only model alone got none. That difference measures the value of giving the checker exact facts; it is **not** an equal-input model comparison. The text-only model again missed all 12 A1 positives. For the contract categories, perfect results mean the checker applied generated facts correctly, not that it extracted those facts from text or verified that a caller told the truth. For D1/S1, the cards describe abstract patterns and response fields, not authentic child messages or responses. Batch 0026 is consumed. These counts do not measure live Romanian or Ukrainian child-safety accuracy.

## Frozen benign Romanian request probe, neutral 0001

The [prospective protocol](docs/NEUTRAL_PROBE_PROTOCOL.md) registered 48 original, harmless Romanian requests across arithmetic, science, language, creative work, general knowledge, and computer tasks before any inference. Each was prefixed with the same fictional declared minor role, then scored once as a `message` by the **unchanged** committed v11 adapter and cutoff. The [raw scores](eval/runs/ro-mmbert-neutral-0001.json) and [recomputation script](eval/verify_mmbert_neutral.py) bind the data, protocol, runner, adapter, and decision rule. No request was flagged: **0/48 false D1 reviews**, with 0/8 in each group. This passed the registered target of at most two overall and at most one per group. The highest raw D1 score was **0.755**, below the frozen cutoff of **0.984**.

This is a **negative-only, out-of-domain stress probe**. It shows abstention on these authored benign requests, including ordinary requests for help. It gives no positive D1 recall, no balanced accuracy, no verified child-source identity, and no representative estimate of real Romanian or Ukrainian message specificity. A classifier that always abstains would also score 0/48 here. The earlier abstract D1 positives and this probe answer different questions; they cannot be combined into a live accuracy claim. The 48 texts are consumed and will not select a new cutoff.

A [post-hoc score diagnostic](eval/analyze_mmbert_d1_shift.py) verifies all three frozen runs before comparing D1 scores on **negative** inputs. It uses no new threshold or model selection.

| Consumed set | D1 negatives | Median D1 score | Maximum D1 score |
|---|---:|---:|---:|
| Abstract v13 | 24 | 0.000202 | 0.406 |
| Abstract v14 | 12 | 0.000235 | 0.513 |
| Benign requests | 48 | 0.196 | 0.755 |

The benign-request scores are much higher in this small authored sample, though all remain below the frozen 0.984 decision cutoff. The sets differ in wording and structure, so the table cannot isolate a causal source of the shift or estimate a population rate. It is a concrete reason to keep this adapter out of live message screening and to avoid calling its softmax output a calibrated risk probability.
## Frozen Romanian D1/S1 prose transfer, batch 0027

The [V15 protocol](docs/V15_PROTOCOL.md) and [attested batch 0027](data/synthetic/batch-0027.review.json) were committed before inference. The unchanged v11 mmBERT adapter, v4 prompt, and shared cutoff scored 96 new abstract Romanian prose cards in 48 one-fact pairs: 24 D1 and 24 S1. The generator retained typed reference states but changed clause wording and ordering. The nearest same-kind character similarity to earlier batches was at most 0.509. These cards still describe invented fields, not child messages or proposed replies.

| Frozen v11 adapter | Result |
|---|---:|
| Strictly parsed cards | 96/96 |
| Exact cards | 70/96 |
| Exact complete pairs | **23/48** |
| False reviews on negative cards | **9/48** |
| D1 positive recall | 23/24 |
| S1 positive recall | **8/24** |

V15 **failed** its preregistered ≥42/48 pair, ≥21/24 recall per category, and ≤4/48 false-review target. All nine false reviews were on D1 negative cards; four of six cards lacking the contextual anchor were falsely flagged. S1 missed 16 of 24 positives across several required fields. The same frozen model passed V13 at 47/48 pairs on a different symbolic renderer, so the drop is evidence of wording sensitivity in this invented domain. The [saved row scores](eval/runs/ro-mmbert-v15-test-0027.json) and [independent recomputation](eval/verify_mmbert_v15.py) preserve the failure. Batch 0027 is consumed and cannot select a new cutoff or training checkpoint. Neither result estimates accuracy on authentic Romanian child language.

## Frozen Romanian D1/S1 prose repair, batch 0030

The [V16 protocol](docs/V16_PROTOCOL.md) registered a continuation of v11 on four new abstract D1/S1 prose forms, with a separate development form and a sealed test form. The first launch hit a [prefit loader mismatch](docs/V16_PREFIT_AMENDMENT.md) before any optimizer step; the narrow category gate was corrected and committed. The completed three-epoch fit tied at **24/24 development pairs** each epoch, so the fixed tie rule chose epoch 1 and cutoff 0.5. The [selected development record](eval/runs/ro-mmbert-v16-dev-selection.json), [row scores](eval/runs/ro-mmbert-v16-epoch-1-dev.json), and [private selected adapter](models/ro-mmbert-v16-abstract/README.md) were committed before opening [batch 0030](data/synthetic/batch-0030.review.json).

| Model on the same 48 abstract pairs | Exact cards | Exact pairs | D1 TP/FP/FN | S1 TP/FP/FN | False reviews on 48 negatives |
|---|---:|---:|---:|---:|---:|
| V16 selected prose repair | 81/96 | **33/48** | 24/15/0 | 24/0/0 | **15/48** |
| Unchanged v11 adapter and cutoff | 69/96 | 22/48 | 23/18/1 | 16/0/8 | 18/48 |

V16 **failed** its registered ≥42/48 pair and ≤4/48 false-review requirements despite recovering every S1 positive and outperforming v11 by 11 exact pairs on this sample. V16 alone got 11 pairs right and v11 alone got none; the descriptive exact two-sided McNemar probability is 0.00098 for this invented test. D1 falsely flagged 5/6 negatives lacking the safety anchor, 6/6 lacking the indirect pattern, and 4/6 lacking the explicit request; adult-source negatives were 0/6. The result suggests the model learned the source-role distinction but remains unreliable at recognizing absent support signals in unfamiliar wording. The [frozen predictions](eval/runs/ro-mmbert-v16-test-0030.json) and [recomputation](eval/verify_mmbert_v16.py) preserve both models' scores and decisions. Batch 0030 is consumed. The symbolic prose cannot estimate recognition of real child disclosures or response quality.

## Prospective Romanian D1 cutoff transfer, batch 0032

The [V17 protocol](docs/V17_PROTOCOL.md) registered a calibration-only attempt after V16's D1 false-review failure. V16 weights, prompt, and S1 cutoff remained frozen. A new [development batch 0031](data/synthetic/batch-0031.review.json) selected D1 cutoff **0.7286992073059082** from saved scores: 43/48 complete development pairs, 22/24 D1 pairs, and two false reviews. The [selection record](eval/runs/ro-mmbert-v17-dev-selection.json) was committed and pushed before test inference. The separate [attested batch 0032](data/synthetic/batch-0032.review.json) has 96 abstract Romanian cards in 48 one-fact pairs using another clause vocabulary.

| Frozen V16 weights on batch 0032 | Exact cards | Exact pairs | D1 TP/FP/FN | S1 TP/FP/FN | False reviews / 48 negatives |
|---|---:|---:|---:|---:|---:|
| V17 development-selected D1 cutoff | 72/96 | **24/48** | 24/23/0 | 24/1/0 | **24/48** |
| Original V16 shared cutoff 0.5 | 72/96 | **24/48** | 24/23/0 | 24/1/0 | **24/48** |

V17 **failed** its registered ≥42 pairs, ≤4 false reviews, and improvement requirements. D1's 24 test negatives had median score **0.9902**, versus **0.9990** for positives; both groups reached almost 1. A diagnostic enumeration after the failure found that even the best **post-hoc** cutoff retaining at least 21 D1 positives would still flag at least 12 D1 negatives, and a cutoff allowing at most four D1 false reviews would retain at most seven positives. These retrospective bounds are failure analysis, not a selected policy. The [frozen score vectors](eval/runs/ro-mmbert-v17-test-0032.json) and [recomputation](eval/verify_mmbert_v17.py) preserve the result. Batch 0032 is consumed. This result rules out a cutoff-only repair on this invented surface; it does not measure authentic Romanian disclosures or Ukrainian performance.

## Prospective Romanian factorized D1 study, batch 0034

The [V18 protocol](docs/V18_PROTOCOL.md) registered four separately supervised abstract D1 fields and the fixed rule `minor AND anchor AND (indirect OR explicit)` after V17's cutoff failure. Attested batches 0028, 0031, and 0032 supplied 288 D1 training cards; earlier consumed batches were used **only for training** in this new study. [Batch 0033](data/synthetic/batch-0033.review.json) supplied 24 D1 development pairs, and separately attested [batch 0034](data/synthetic/batch-0034.review.json) supplied 48 D1 and 24 S1 sealed-test pairs on two new symbolic surfaces. The first trainer launch had an [import error before model loading](docs/V18_PREFIT_AMENDMENT.md); the fix was committed before a full three-epoch fit.

The [development selection](eval/runs/ro-mmbert-v18-dev-selection.json) chose epoch 3 by the registered rule, but **failed the development gate** at 11/24 D1 pairs, two false reviews, and 137/192 correct field tasks. The [selected research adapter](models/ro-mmbert-v18-factor-abstract/README.md) and all three epoch scores were committed before test inference. It loads with the pinned base and reproduces a saved development score. In the candidate workflow, V18 scores D1 fields; unchanged V16 scores S1. The comparator uses unchanged V16 for both categories on the **same** test cards.

| Batch 0034 model workflow | Exact cards | Exact pairs | D1 TP/FP/FN | S1 TP/FP/FN | False reviews / 72 negatives |
|---|---:|---:|---:|---:|---:|
| V18 factor D1 plus frozen V16 S1 | 104/144 | **32/72** | 31/22/17 | 24/1/0 | **23/72** |
| Unchanged V16 D1/S1 | 118/144 | **46/72** | 46/23/2 | 24/1/0 | **24/72** |

V18 **failed** its preregistered ≥64 pairs, D1 recall ≥42/48, ≤6 false reviews, and improvement requirements. Across 96 D1 cards, the factor model got minor-source 89/96, anchor 57/96, indirect pattern 46/96, and explicit request 49/96 field judgments right. In particular, it missed 35 present anchors and falsely marked 43 absent explicit requests. The fixed conjunction suppressed some false D1 reviews but also missed 17 positives. The [saved scores and decisions](eval/runs/ro-mmbert-v18-test-0034.json) and [offline recomputation](eval/verify_mmbert_v18.py) preserve the failure. Batch 0034 is consumed; neither workflow validates authentic child-language interpretation. Further synthetic wording exposure alone cannot establish the deployment accuracy the project needs.

A [full inference replay](eval/replay_mmbert_v18.py) subsequently reloaded the pinned V16 and V18 adapters, rescored all 144 test cards and 576 applicable model outputs, and matched every saved score exactly on the local MPS system (maximum absolute difference 0.0). The registered failure remains **32/72 pairs**. This is a reproducibility check on the consumed test, not a new independent result or a way to select a revised model.

## Frozen benign Romanian near-domain probe, 0002

The [prospective protocol](docs/NEAR_DOMAIN_PROBE_PROTOCOL.md), [36 authored benign questions](data/probes/ro-near-domain-0002.json), and [runner](eval/run_mmbert_near_domain.py) were committed as `886d305` before inference. Six groups cover general education about online safety, privacy, support services, boundaries, guardian roles, and health or emergency terms. Every row has an empty D1 reference label and the same fictional minor-source prefix. Unchanged V11, V16, and V18 research adapters were scored once with their frozen decisions. The [raw scores](eval/runs/ro-mmbert-near-domain-0002.json) and [independent recomputation](eval/verify_mmbert_near_domain.py) bind source and adapter hashes.

| Frozen model | False D1 reviews | Group maximum | Registered target met |
|---|---:|---:|---|
| V11 shared cutoff | 0/36 | 0/6 | Yes |
| V16 shared cutoff | 0/36 | 0/6 | Yes |
| V18 four-factor conjunction | 0/36 | 0/6 | Yes |
| Never review reference | 0/36 | 0/6 | Trivial |
| Always review reference | 36/36 | 6/6 | No |

V11's maximum D1 score was 0.681 below its 0.984 cutoff; V16's was 0.294 below its 0.5 cutoff. V18's minor-source factor crossed 0.5 on 31/36 requests, consistent with the supplied role, while its support-anchor, indirect-pattern, and explicit-request factors never crossed 0.5. This is a useful negative-only stress result for these authored generic questions. It does **not** resolve the high false-review counts on unfamiliar abstract D1 counterfactuals, reveal D1 recall, or measure performance on authentic Romanian or Ukrainian child language. The 36 rows are consumed and cannot be used to choose a changed model.

A separate [full inference replay](eval/replay_mmbert_near_domain.py) reloaded each frozen adapter and reproduced every saved score on these 36 rows exactly on the local MPS system. All five committed mmBERT research adapters contain fitted classifier-head tensors; the loader now checks their presence and shape before inference. The base model's initialization warning refers to its unfitted temporary classification head, which the adapter supplies when loaded.

## Prospective Romanian joint-field D1 study, batch 0036

The [V19 protocol](docs/V19_PROTOCOL.md) registered a four-output D1 encoder with joint field prediction and a one-fact pair penalty. Training used 432 abstract D1 cards from previously consumed, attested batches 0028 and 0031–0034. The fresh [development batch 0035](data/synthetic/batch-0035.review.json) contained 24 D1 pairs on a new surface. The [frozen development record](eval/runs/ro-mmbert-v19-dev-selection.json) chose epoch 1 by the registered rule, but **failed the development gate**: 0/24 complete pairs, 24/24 false reviews, and 156/192 correct field judgments. All three epoch scores are preserved. The [selected private adapter](models/ro-mmbert-v19-joint-abstract/README.md) was committed before batch 0036 was generated and scored.

The separately attested [batch 0036](data/synthetic/batch-0036.review.json) supplied 48 one-fact D1 pairs on two further invented Romanian surfaces. The frozen V19 decision was compared with unchanged V18 and V16 on the **same 96 cards**:

| Frozen D1 model | Exact cards | Complete pairs | TP / FP / FN | False reviews on 48 negatives |
|---|---:|---:|---:|---:|
| V19 four-output joint encoder | 58/96 | **14/48** | 25 / 15 / 23 | **15/48** |
| V18 four-question factor encoder | 73/96 | 25/48 | 26 / 1 / 22 | 1/48 |
| V16 direct D1 encoder | 88/96 | 40/48 | 48 / 8 / 0 | 8/48 |

V19 **failed** its registered ≥42 complete pairs, ≥42 D1 true positives, ≤6 false reviews, and superiority requirements. On test, the joint head predicted minor-source and support-anchor present for all 96 cards, while it predicted explicit request absent for all 96. The training distribution was also skewed: 378/432 positive minor-source and support-anchor fields, versus 54/432 positive explicit requests. The unweighted joint loss with a pair penalty did not prevent majority-state collapse. These facts motivate a future class-balanced study; they do not license retuning on batch 0036. The [raw three-model scores](eval/runs/ro-mmbert-v19-test-0036.json) and [independent recomputation](eval/verify_mmbert_v19.py) preserve the failure. Batch 0036 is consumed. All cards describe invented fields, not children’s messages; none of these numbers estimates authentic-language sensitivity or Ukrainian performance.

A [full inference replay](eval/replay_mmbert_v19.py) reloaded all three pinned adapters and matched all **864 saved scores on 96 cards exactly** on the local MPS system. The registered failure remains 14/48 pairs. This replay checks reproducibility on the consumed test; it is not a new held-out result.

## Prospective class-balanced Romanian D1 study, batch 0037

The [V20 protocol](docs/V20_PROTOCOL.md) targeted V19's majority-state collapse with positive-class weights computed from training only and a stronger one-fact pair margin. It trained on 528 abstract D1 cards from already consumed batches, including batch 0036 as **training**, and reused consumed batch 0035 for **development**. The [five-epoch development record](eval/runs/ro-mmbert-v20-dev-selection.json) chose epoch 4 at 15/24 complete pairs, 0/24 false reviews, and 164/192 correct field judgments. That improved on V19's 0/24 pairs on the same development set but **failed** the registered ≥21 pairs and ≥170 field judgments gate. The [selected private adapter](models/ro-mmbert-v20-balanced-joint-abstract/README.md) and scores were committed before the new test was generated.

The separately attested [batch 0037](data/synthetic/batch-0037.review.json) has 48 one-fact D1 pairs on two further invented Romanian surfaces. All four frozen adapters scored the same 96 cards:

| Frozen D1 model | Exact cards | Complete pairs | TP / FP / FN | False reviews on 48 negatives |
|---|---:|---:|---:|---:|
| V20 balanced joint-field encoder | 66/96 | **18/48** | 31 / 13 / 17 | **13/48** |
| V19 unweighted joint-field encoder | 51/96 | 4/48 | 32 / 29 / 16 | 29/48 |
| V18 four-question factor encoder | 83/96 | 36/48 | 41 / 6 / 7 | 6/48 |
| V16 direct D1 encoder | 78/96 | 33/48 | 39 / 9 / 9 | 9/48 |

V20 **failed** its registered ≥42 pairs, ≥42 D1 true positives, ≤6 false reviews, and superiority requirements. Class weighting reduced the complete field collapse seen in V19: V20 recognized 9/12 adult-source negatives as adult and all 12 explicit-request positives. It also falsely marked explicit request on 22/84 cards without it and missed 22/60 present indirect patterns. Thus better field balance did not yield stable final decisions on the new wording. V18 performed best on this sample, yet still missed the pair and recall targets; its 36/48 result is descriptive on this consumed batch, not a development selection. The [raw four-model scores](eval/runs/ro-mmbert-v20-test-0037.json) and [independent recomputation](eval/verify_mmbert_v20.py) preserve the outcome. A [full inference replay](eval/replay_mmbert_v20.py) matched all **1,248 saved scores exactly** on the local MPS system. Batch 0037 is consumed. The study remains limited to invented state descriptions and cannot estimate authentic child-language or Ukrainian performance.

Post-test diagnosis, **not a new selection result**: on the first 24 pairs (the “registru / marcaj pozitiv-negativ” surface), V20 got 11 complete pairs, 24/24 positives, and 13/24 false reviews. It marked explicit request on 22/42 cards where that field was absent. On the other 24 pairs (the “încadrată / valabilă-anulată” surface), V20 got 7 complete pairs, 7/24 positives, and 0/24 false reviews; it missed 21/30 present indirect patterns. V18 went 15/24 then 21/24 complete pairs on these same surfaces, and V16 went 10/24 then 23/24. The opposite V20 errors across two invented phrasings show why another threshold or an overall average would hide the failure. A future study should register genuinely new, independently authored language surfaces and a polarity-sensitive diagnostic before another model choice; these consumed rows cannot select that choice.

## Prospective Romanian NLI-adapted D1 study, batch 0038

The [V21 protocol](docs/V21_PROTOCOL.md) changes the encoder to a pinned multilingual NLI model with four fixed Romanian hypotheses. Training used 528 consumed abstract D1 cards; batch 0037 was reused **only for development**. The three epoch records are retained in `eval/runs/ro-nli-v21-epoch-{1,2,3}-dev.json`, and the [selection record](eval/runs/ro-nli-v21-dev-selection.json) chose epoch 2 by the registered order: **23/48 complete pairs**, **19/48 false reviews**, and **308/384 correct field judgments**. That **failed** the ≥42 pairs, ≤6 false reviews, and ≥340 field judgments development gate. The [selected adapter](models/ro-nli-v21-abstract/README.md) and [independent selection recomputation](eval/verify_committed_v21_selection.py) were committed before generation of the new test.

The separately [attested batch 0038](data/synthetic/batch-0038.review.json) has 48 one-fact pairs on two further invented Romanian abstract surfaces. The runner and batch were pushed before any inference. All four frozen models scored the same 96 cards:

| Frozen D1 model | Exact cards | Complete pairs | TP / FP / FN | False reviews on 48 negatives |
|---|---:|---:|---:|---:|
| V21 NLI four-hypothesis adapter | 56/96 | **9/48** | 33 / 25 / 15 | **25/48** |
| V20 balanced joint-field encoder | 67/96 | 19/48 | 34 / 15 / 14 | 15/48 |
| V18 four-question factor encoder | 79/96 | 31/48 | 46 / 15 / 2 | 15/48 |
| V16 direct D1 encoder | 90/96 | **42/48** | 46 / 4 / 2 | **4/48** |

V21 **failed** its registered test gate and was far worse than unchanged V16 on this new surface. Its four fields were correct on 73/96 minor-source, 87/96 support-anchor, 61/96 indirect-pattern, and 49/96 explicit-request judgments. On the first invented phrasing it got 7/24 pairs, 23/24 positives, and 16/24 false reviews; on the second, 2/24 pairs, 10/24 positives, and 9/24 false reviews. A natural-language-inference backbone did not resolve the polarity problem; the same-author synthetic phrasing remains a major source of variance. The [raw scores](eval/runs/ro-nli-v21-test-0038.json) and [offline recomputation](eval/verify_nli_v21.py) preserve the failed test. Batch 0038 is consumed and cannot select a new model. V16's 42/48 is a descriptive comparator result on these abstract cards, not evidence of dependable detection on Romanian child messages. Ukrainian linguistic accuracy remains unevaluated.

A [full inference replay](eval/replay_nli_v21.py) reloaded both pinned base models and all four frozen adapters, then reproduced all **1,248 saved scores exactly** on this machine. That confirms the recorded failure is repeatable; it adds no new held-out accuracy evidence.

## Retrospective fixed-V16 D1 transfer audit

The [cross-batch audit](eval/runs/ro-v16-d1-transfer-audit.json) recomputes one unchanged V16 cutoff from its saved scores in the four consumed tests 0034, 0036, 0037, and 0038. Each batch has two invented D1 phrasings with 24 one-fact pairs each. The audit first runs every original test verifier and checks model and source hashes. It neither fits a cutoff nor selects a model.

| Consumed batch | Surface 1: exact / false reviews | Surface 2: exact / false reviews | Positive score above paired negative |
|---|---:|---:|---:|
| 0034 | 4/24 · 20/24 | 19/24 · 3/24 | 48/48 |
| 0036 | 18/24 · 6/24 | 22/24 · 2/24 | 48/48 |
| 0037 | 10/24 · 8/24 | 23/24 · 1/24 | 43/48 |
| 0038 | 18/24 · 4/24 | 24/24 · 0/24 | 46/48 |

Across these **192 pairs of abstract cards**, V16 produced 138 complete pairs, 179/192 positive reviews, and 44/192 false reviews on negatives. The positive card scored above its paired negative in 185/192 pairs, but complete-pair accuracy ranged from 4/24 to 24/24 by surface. This separates two observed defects: a fixed score cutoff is unstable under invented wording changes, and seven pair orderings reverse even before cutoff selection. Pair ordering assumes both counterfactual cards are available; a real single-message screen would not have that information. These repeated same-author surfaces are correlated and do not yield a child-language accuracy estimate or a basis to retune V16 on their consumed labels.

A separate [retrospective cutoff feasibility calculation](eval/runs/ro-v16-cutoff-feasibility.json) evaluated 769 candidate cutoffs spanning every decision region induced by the saved scores. Even an oracle using **all consumed labels** could get at most 141/192 complete pairs with one common cutoff, and could make the worst surface no better than 7/24. The frozen 0.5 cutoff got 138/192 and 4/24 at worst. An individually chosen cutoff could not exceed 12/24 on one surface because several pair score orderings reversed. These are post-hoc bounds, **not** new held-out scores or a permissible cutoff choice. They show that another global cutoff alone cannot meet a robust cross-surface target for this frozen model.

## V22 balanced D1 development candidate; no new test

The [V22 protocol](docs/V22_PROTOCOL.md) fixed a continuation of unchanged V16 on seven previously consumed Romanian D1 surface families. Each epoch drew 24 pairs per family, trained at the original `0.5` cutoff, and selected the checkpoint by the **worst** of four development surfaces. The two development batches, 0037 and 0038, were already consumed in earlier model tests. This reuse is explicitly development work, not fresh evaluation. The protocol and trainer were pushed before fitting.

| Frozen development workflow | Surface pairs (out of 24 each) | Total exact pairs | Positive reviews | False reviews on 96 negatives |
|---|---|---:|---:|---:|
| V22 selected epoch 2 | 8, 24, 17, 24 | **73/96** | 75/96 | **3/96** |
| Unchanged V16 | 10, 23, 18, 24 | 75/96 | 85/96 | 13/96 |

V22 **failed** its registered development gate of at least 18/24 on every surface, 84/96 overall, 90/96 positive reviews, at most 8/96 false reviews, and superiority to V16. Group-balanced training reduced false reviews but also missed ten additional positives and worsened the worst surface. The [epoch 1](eval/runs/ro-mmbert-v22-epoch-1-dev.json), [epoch 2](eval/runs/ro-mmbert-v22-epoch-2-dev.json), and [epoch 3](eval/runs/ro-mmbert-v22-epoch-3-dev.json) score streams and the [selection record](eval/runs/ro-mmbert-v22-dev-selection.json) remain available; the [selected private adapter](models/ro-mmbert-v22-abstract/README.md) reproduces a saved score with the pinned base. The [selection verifier](eval/verify_committed_v22_selection.py) recomputes every development decision and the failed gate from raw scores. There is no new held-out V22 test. V22 is not an operational D1 detector, and its development scores say nothing about authentic child messages.

A [full development inference replay](eval/replay_mmbert_v22.py) reloaded the selected adapter and rescored all 192 development cards. Every score matched within `1.5×10⁻⁶` on the local MPS system. This confirms reproducibility of the failed development result; it adds no independent test evidence.

The [post-hoc V16/V22 fusion audit](eval/runs/ro-v22-ensemble-development-audit.json) rechecked these 96 consumed development pairs. Every V22 positive decision was already a V16 positive decision: V22 corrected ten V16 false reviews while dropping ten V16 true reviews. Both ranked the positive above its negative in 89/96 pairs. Averaging their scores at the unchanged `0.5` cutoff yielded 75/96 complete pairs, 80/96 positive reviews, and 7/96 false reviews, with only 8/24 pairs on the weakest surface. This simple fusion did not resolve the gate failure. It is a retrospective diagnosis on correlated cards, **not** a selected model or a new test. The [audit code](eval/audit_v22_ensemble.py) recomputes the figures from frozen score streams.

## V23 group-weighted D1 development candidate; no new test

The [V23 protocol](docs/V23_PROTOCOL.md) fixed one continuation of V16 with balanced exposure to the same seven consumed training families, positive-class weight `1.5`, and online group-loss weighting. The protocol and trainer were pushed before fitting. Batches 0037 and 0038 were already consumed and were used only as development surfaces. The selected epoch was fixed by the worst-surface rule, then total pairs, false reviews, and earliest epoch.

| Development workflow | Surface pairs (out of 24 each) | Total exact pairs | Positive reviews | False reviews on 96 negatives |
|---|---|---:|---:|---:|
| V23 selected epoch 2 | 11, 24, 17, 24 | **76/96** | 81/96 | **6/96** |
| V22 selected epoch 2 | 8, 24, 17, 24 | 73/96 | 75/96 | 3/96 |
| Unchanged V16 | 10, 23, 18, 24 | 75/96 | 85/96 | 13/96 |

V23 gained three complete pairs and six positive reviews over V22, with three more false reviews. It exceeded V16 by **one** complete pair while missing four more positives. It **failed** the unchanged development gate: the weakest surface reached 11/24 rather than 18/24, the total 76/96 rather than 84/96, and positive recovery 81/96 rather than 90/96. The 6/96 false-review count met that one criterion. Batch 0032 carried about 31% of the selected epoch's learned group weight; this is a training diagnostic, not evidence that the model identified a reliable cross-surface rule.

The [epoch 1](eval/runs/ro-mmbert-v23-epoch-1-dev.json), [epoch 2](eval/runs/ro-mmbert-v23-epoch-2-dev.json), and [epoch 3](eval/runs/ro-mmbert-v23-epoch-3-dev.json) score streams and the [selection record](eval/runs/ro-mmbert-v23-dev-selection.json) are preserved. The [verifier](eval/verify_committed_v23_selection.py) recomputes their decisions, metrics, choice, and failed gate. The [selected private adapter](models/ro-mmbert-v23-abstract/README.md) replayed all 192 saved development scores within `1.4×10⁻⁶` on local MPS using the [inference replay](eval/replay_mmbert_v23.py). There is **no new held-out V23 test** and no evidence of accuracy on authentic child language. The next valid model comparison still needs independently authored, fluent-reviewed, adjudicated abstract cards frozen after the model and evaluation code.

## V24 Romanian BERT D1 development candidate; no new test

The [V24 protocol](docs/V24_PROTOCOL.md) and [trainer](training/train_robert_v24.py) were committed and pushed before fitting. V24 tested the pinned Romanian-specific cased BERT encoder with a new two-class head and pairwise LoRA training. It reused seven consumed training batches and the four consumed development surfaces in batches 0037 and 0038. The `0.5` cutoff, six epochs, selection order, and numeric gate were fixed before training. No cards or held-out test were added.

| Candidate | Surface pairs (out of 24 each) | Total exact pairs | Positive reviews | False reviews on 96 negatives |
|---|---|---:|---:|---:|
| V24 epoch 1 | 0, 0, 0, 0 | 0/96 | 4/96 | 5/96 |
| V24 epoch 2 | 0, 0, 1, 0 | 1/96 | 15/96 | 18/96 |
| V24 epoch 3 | 7, 0, 2, 0 | 9/96 | 13/96 | 10/96 |
| V24 epoch 4 | 8, 0, 4, 0 | 12/96 | 48/96 | 36/96 |
| **V24 selected epoch 5** | **17, 0, 8, 0** | **25/96** | **47/96** | **22/96** |
| V24 epoch 6 | 15, 0, 10, 0 | 25/96 | 48/96 | 23/96 |
| V23 selected, consumed development | 11, 24, 17, 24 | 76/96 | 81/96 | 6/96 |
| Unchanged V16, same consumed cards | 10, 23, 18, 24 | 75/96 | 85/96 | 13/96 |

V24 **failed** every numeric gate component: its weakest surface scored 0/24 versus the required 18/24, total 25/96 versus 84/96, positive recovery 47/96 versus 90/96, false reviews 22/96 versus the maximum 8/96, and total pairs were below V16's 75/96. This experiment does not support substituting the Romanian BERT adapter for the existing research baseline. The selected model ranked the positive above its negative on 24/24, 24/24, 18/24, and 24/24 pairs respectively, but its score ranges shifted across the four surfaces. A **post hoc diagnostic** using every consumed development label found that even the best common cutoff for the weakest surface would reach only 1/24 complete pairs. That oracle cutoff is not a validated threshold or a new test result.

All [six raw epoch score streams](eval/runs/ro-bert-v24-epoch-5-dev.json), the [selection record](eval/runs/ro-bert-v24-dev-selection.json), [selected private adapter](models/ro-bert-v24-abstract/README.md), and [verifier](eval/verify_committed_v24_selection.py) are preserved. The verifier recomputes every metric, epoch choice, hash, failure, and cutoff diagnostic from the committed cards and scores in CI. The [inference replay](eval/replay_robert_v24.py) reloaded the packaged adapter with the pinned base and reproduced all 192 saved selected-epoch scores exactly on local MPS. No new held-out V24 test exists. Lower training loss over six epochs alongside weak development decisions suggests poor cross-surface transfer, but these invented cards cannot establish behavior on authentic child language. Further fitting to these consumed surfaces would not provide independent accuracy evidence.

## Bilingual same-origin metadata studies, V25–V29

The [V25 protocol](docs/V25_SYNTHETIC_PROTOCOL.md) registered separate Romanian and Ukrainian abstract wording banks, one-fact D1/S1 pairs, a hidden test seed, fixed thresholds, and a per-cell target before fitting. The [seed commitment](eval/prospective/v25-synthetic-seed-commitment.json) and trainer were pushed at `573267a`; the selected four-epoch mmBERT adapter was pushed at `2cc00b2` before test reveal. The [once-scored test](eval/prospective/v25-synthetic-result.json), [predictions](eval/prospective/v25-synthetic-predictions.jsonl), and 384 [test rows](data/synthetic/v25/test.jsonl) were pushed at `3b14348`. The test is consumed. A complete pair requires the negative and positive variant to be correct under one fixed cutoff, with no pair context supplied to the model.

| V25 sealed test cell | Complete pairs | TP / FP / FN | Precision | Recall | Specificity |
|---|---:|---:|---:|---:|---:|
| Romanian D1 | 43/48 | 43 / 0 / 5 | 100% | 89.6% | 100% |
| Romanian S1 | 48/48 | 48 / 0 / 0 | 100% | 100% | 100% |
| Ukrainian D1 | 36/48 | 48 / 12 / 0 | 80% | 100% | 75% |
| Ukrainian S1 | 38/48 | 41 / 4 / 7 | 91.1% | 85.4% | 91.7% |

V25 reached **165/192 complete pairs (85.9%; descriptive 95% Wilson interval 80.3–90.2%)**. The fixed character 3–5-gram comparator reached 35/192; always-review and never-review each reached 0/192 because neither can get both variants of a pair right. The encoder beat those baselines but **failed** the registered requirement of at least 44/48 complete pairs, 44/48 positives, and 44/48 negatives in *every* cell. Ukrainian D1 made 12 false reviews on 48 negatives; Ukrainian S1 missed seven positives. Even Romanian D1 missed its target by one. These are same-author generated **metadata** sentences, never child utterances; their high-scoring cells cannot be promoted into a real-message accuracy claim.

The next studies preserve failed selections and leave their test seeds unopened. Their development rows use new wording families, not a continuation of the V25 test. Each selected rule was frozen before its own development scoring; each protocol requires at least 168/192 complete development pairs **and** 40/48 in every cell before test reveal.

| Study and selected development rule | RO D1 | RO S1 | UK D1 | UK S1 | Total | Test status |
|---|---:|---:|---:|---:|---:|---|
| [V26 latent facts](docs/V26_FACT_PROTOCOL.md), epoch 5 | 36/48 | 43/48 | 36/48 | 48/48 | 163/192 | Unopened; gate failed |
| [V27 class-balanced facts](docs/V27_BALANCED_PROTOCOL.md), epoch 4 selected by worst cell | 26/48 | 25/48 | 26/48 | 27/48 | 104/192 | Unopened; gate failed |
| [V28 entailment hybrid](docs/V28_HYBRID_PROTOCOL.md) | 26/48 | 48/48 | 46/48 | 48/48 | 168/192 | Unopened; RO D1 gate failed |
| [V29 anchor-routed hybrid](docs/V29_HYBRID_PROTOCOL.md) | 31/48 | 48/48 | 48/48 | 48/48 | 175/192 | Unopened; RO D1 gate failed |

The [V26](models/bi-mmbert-v26-facts/development-selection.json), [V27](models/bi-mmbert-v27-balanced/development-selection.json), [V28](models/bi-v28-hybrid/development-selection.json), and [V29](models/bi-v29-hybrid/development-selection.json) records contain raw development scores, per-factor errors, component hashes, and the explicit `development_eligible_for_test_reveal: false` flag. V27's registered worst-cell selector chose epoch 4 at 104/192, although a later epoch reached 135/192 overall with a worse 15/48 weakest cell. V28 and V29 improved three cells but repeatedly missed Romanian D1 under different wordings. Development-only probes of alternative fact readers and entailment hypotheses informed later rules; they are not held-out results. The V26–V29 hidden test seeds remain local and unspent.

The bilingual wording was drafted by this project and has no independent fluent review. These measurements concern controlled abstract fields only. They cannot establish authentic disclosure detection, safe child-facing responses, or deployable Romanian/Ukrainian language accuracy.

Subsequent private, preregistered synthetic studies V30 and V31 passed their development gates but failed separate once-scored tests: **129/192** and **143/192** complete pairs, respectively. V30's Ukrainian D1 reached only **2/48** pairs on new wording; V31's Romanian D1 reached only **13/48**. Exact cards, predictions, seed reveals, and replay checks are retained in the private research archive rather than this public package. These failures reinforce the need for independently reviewed taxonomy wording and newly authored, adjudicated abstract cards before any v1 model claim.
