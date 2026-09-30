# V28: frozen evidence hybrid with a new sealed wording family

Three prior synthetic studies exposed a recurring error. V25's sealed test reached 165/192 complete pairs, below its fixed target. V26 and V27 failed their pretest development gates and left their test seeds unopened. V27's best balanced-fact epoch reached only 104/192, weakest cell 25/48. A development-only diagnostic on V27 style 8 then tested a **frozen component hybrid**: the V26 fact model plus a locally cached multilingual entailment model. Entailment read source role and S1 applicability well; using it for those facts raised the hybrid to 182/192 on that diagnostic. Romanian D1 remained 38/48 there. Replacing all four Romanian D1 fact estimates with entailment raised that cell to 44/48 on the same diagnostic. These figures are model-design evidence, not prospective test results.

## Fixed rule and scope

V28 freezes that rule before seeing its own development or test cards:

| Cell | Fact reader | Decision composition |
|---|---|---|
| Romanian D1 | Multilingual entailment for source role, anchor, indirect pattern, direct request | `minor × anchor × (indirect OR direct)` |
| Ukrainian D1 | Frozen V26 mmBERT for anchor and support patterns; entailment for source role | Same D1 rule |
| Romanian/Ukrainian S1 | Frozen V26 mmBERT for pass status; entailment for applicability | `applicable × NOT passed` |

The entailment score is `P(entailment) / [P(entailment) + P(contradiction)]`, discarding the neutral logit. Each hypothesis is written in the card's language and is fixed in `src/roguard/v28_hybrid.py`. Only the semicolon-delimited body of invented metadata is read; the leading card title/marker and closing boilerplate are excluded. This is a research reader for abstract cards only. It is not a free-form child-message classifier, a permission router, or an automatic escalation path.

V28 uses the pinned V26 adapter at `models/bi-mmbert-v26-facts` and the locally cached `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` revision `b5113eb38ab63efdd7f280f8c144ea8b13f978ce`. Their exact file hashes are registered in code and the selection record. V28 fits **no new model weights**. It chooses only four decision cutoffs on its new development set. The model and composition rule were chosen after examining earlier development sets; the V28 test is the prospective check of that choice.

## Data and chronology

Each language/category has 192 paired training cards, 48 development pairs, and 48 sealed test pairs. Training wording styles 0–8 support the fixed character n-gram comparator only. The hybrid and both frozen encoder comparators use their already selected weights. V28 development uses style 9, which was reserved for an unopened V27 test; its cards use a new public seed and do not use V27's private seed. V28 test uses newly written style 10 and a new hidden 192-bit seed. All cards are invented annotation metadata. They contain no child utterance, real or realistic disclosure, case narrative, or drafted response. Labels derive from the typed D1/S1 policy oracle, and one fact changes per pair. Romanian and Ukrainian wording was drafted separately, but has no independent fluent review.

Run `python -m training.generate_v28_hybrid prepare` once. Commit and push this protocol, generator, wording, hybrid code, selector, scorer, tests, training/development data, and seed-hash commitment before cutoff selection. Unit tests use an unrelated dummy seed. The test seed remains only under ignored `review_runs/`; test rows do not exist at registration. Select cutoffs from the fixed 0.05 grid on V28 development only, maximizing complete pairs, then correct cards, then fewer false reviews. Preserve raw development scores and evidence. The development gate is **168/192 complete pairs and at least 40/48 in every cell**. If it fails, keep the test closed and preserve the failure. If it passes, commit and push the cutoffs and full development record before reveal, then generate and score the test exactly once without changing the hybrid.

## Target and interpretation

Use the unchanged V25 target: **at least 44/48 complete pairs and at least 44/48 positive and negative decisions in each cell**, at least 9/12 per D1 changed factor, at least 20/24 per S1 changed factor, and all scores finite. Report all cells and factors, style 10, false reviews, recall, specificity, precision, Wilson intervals, six fact accuracies, the frozen V25 and V26 models with original cutoffs, the V28-trained character comparator with development-selected cutoffs, and always/never review. Mark failure even if aggregate performance is high.

A pass supports only transfer among same-origin synthetic metadata phrasings. It cannot validate idiomatic Romanian/Ukrainian, independent label reliability, recognition in children's messages, response safety, or real-world deployment. The independent review and source-audit release gates remain closed.
