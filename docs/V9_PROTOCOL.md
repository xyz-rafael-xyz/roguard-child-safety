# V9 multilingual encoder study

**Status:** full-weight fitting was interrupted before test inference because of local memory pressure. See the prospective [v9a amendment](V9A_AMENDMENT.md). The original fixed procedure below remains for audit.

This study asks whether a discriminative multilingual encoder can learn Romanian symbolic contract decisions better than the prior generative adapter. The already consumed batches 0016–0018 remain failures in the record. The new [batch 0019](../data/synthetic/batch-0019.preview.md) was generated and attested before fitting or test inference. It has 144 abstract cards in 72 one-fact pairs, with distinct field wording and `adevărat`/`fals` values. Its typed reference rules are shared with the training set, so it tests surface transfer inside a synthetic rule domain, not authentic child language. No child utterances, responses, cases, or realistic paraphrases appear.

## Fixed fit and choice

Use [JHU mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base) at revision `c5955035435e2bf121cde7f3c8863ef52ff35d82`, with hashes fixed in `src/roguard/mmbert_study.py`. Initialize a two-class sequence classification head and fine-tune all weights using the unchanged Romanian v4 category-specific prompt. Use only batch 0014 for training and batch 0015 for development. Repeat D1, R1, P1, and G1 tasks three times in training, as in v8; there are 1,728 task exposures but only 576 unique cards. Use seed 20260930, batch size 8, maximum length 256, AdamW with learning rate 2e-5, linear 10% warmup and decay, and three epochs. Save every epoch. Class 1 at softmax probability at least 0.5 means `da`; no threshold fitting. Nonfinite scores or inference errors fail the run and must be recorded and repaired before a new test attempt.

On batch 0015, choose among epochs 1–3 by complete exact pairs, then exact cards, then fewer false reviews on negative cards, then earlier epoch. Commit the selection record and selected weight hash before any batch-0019 inference. Do not edit the prompt, threshold, training data, or holdout after inspecting test output.

## Frozen test and decision

Evaluate the chosen encoder once on batch 0019. Evaluate the previously selected v8 Qwen adapter on the same cards with its frozen parser and prompt as a descriptive comparator; include always-review and never-review references. Preserve row-level scores, exact cards, complete pairs, correct-direction flips, false reviews on 72 negatives, per-category precision and recall, model/prompt/test hashes, and paired discordance. The encoder reaches its **symbolic-card** target only at ≥58/72 complete pairs, ≥9/12 positive recall for each category, ≤7/72 false reviews, and more complete pairs than v8. A failed run stays visible and motivates a new predeclared experiment on a new holdout. Neither result establishes child-language detection, Ukrainian performance, or operational safety; the structured checker and human review remain the usable path.
