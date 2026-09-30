# Frozen Romanian compositional study, v6

This protocol, generator, labels, split assignment, and evaluation code must be committed before training. Batch 0014 contains 576 training cards in 288 one-fact pairs, batch 0015 contains 96 development cards in 48 pairs, and batch 0016 contains 144 test cards in 72 pairs. Each category has equal positive and negative pair sides. All cards are symbolic descriptions of typed facts, never child utterances, response text, reported cases, or paraphrases. The test uses separate framing and field-name variants, but the same category logic; it is a surface-transfer test, not a sample of real child language.

## Fixed intervention

Use the same pinned RoMistral 4-bit base, v4 category-specific `da`/`nu` prompt and parser, rank-16 scale-2 QLoRA, eight adapter layers, dropout 0.05, prompt masking, batch size 1, 512-token cap, learning rate `1e-5`, and seed 20260929. Train on **batch 0014 only** for 1,200 steps. The exporter repeats positive response-category tasks three times on training only, as in v4; development and test have no repeats. No prior test batch enters training. The primary change is compositional, reference-checked training data with varied order and terminology, not a new model or prompt.

Select one checkpoint from steps 300, 600, 900, and 1200 on batch 0015 only. Maximize complete exact pairs, then exact cards, then minimize false reviews on negative cards, then choose the earlier step. Strict parse failures count as empty predictions and remain visible. Commit the checkpoint-selection record and selected weight hash before any batch-0016 inference. Do not fit thresholds on the test.

## Test and decision rule

Open batch 0016 once for the selected adapter. Run three frozen comparators on the same cards: the previous selected v4 adapter, prompt-only RoMistral, and prompt-only Qwen3 4B, all with the same v4 binary prompt and eight-token cap. Report strict parse, exact cards, exact complete pairs, correct-direction flips, false reviews on 72 negative cards, and per-category TP/FP/FN, precision, and recall. Preserve raw prompt-only outputs and hashes of all test, model, prompt, and prediction inputs. Include always-review and never-review references.

The v6 intervention meets its **symbolic-card** accuracy target only if it reaches at least 58/72 exact pairs, recalls at least 9/12 positives in every category, has at most 7/72 false reviews on negative cards, and exceeds all three learned comparators on exact pairs. Any failure is reported, not repaired by tuning on batch 0016. Even success here cannot establish accuracy on real child messages, Ukrainian language understanding, child comprehension, or deployment safety. Routing, permission, and external-action decisions continue to require the declared-contract checker and a human decision.
