# V29: anchor-routed bilingual hybrid with a new sealed test

V28's development result was 168/192 complete pairs but failed its pretest per-cell gate: Romanian D1 was 26/48 because its entailment anchor hypothesis missed all 12 changed-anchor pairs. V28's test remains sealed. A development-only repair probe replaced **only the Romanian D1 anchor estimate** with the frozen V26 fact-head estimate. With the rest of V28's evidence rule unchanged, that probe reached 41/48 Romanian D1 and 183/192 overall on V28 development. Those are diagnostic numbers; they are not the V29 result.

## Fixed reader

V29 freezes this exact fact routing before generating V29 development or test rows:

| Cell | Source of each fact | Composition |
|---|---|---|
| Romanian D1 | Multilingual entailment: source role, indirect pattern, direct request. Frozen V26 mmBERT: safety/support anchor. | `minor × anchor × (indirect OR direct)` |
| Ukrainian D1 | Entailment: source role. Frozen V26 mmBERT: anchor, indirect pattern, direct request. | Same D1 rule |
| Romanian/Ukrainian S1 | Entailment: field applicability. Frozen V26 mmBERT: pass status. | `applicable × NOT passed` |

The entailment score is `P(entailment) / [P(entailment) + P(contradiction)]`. The bilingual hypotheses, body extraction rule, and component model hashes are inherited from the committed V28 implementation and frozen by source hash. V29 fits no new weights. It selects only one decision cutoff for each language/category cell from the fixed 0.05 grid on V29 development. This research component accepts only invented abstract metadata cards. It does not process free-form child messages or take an escalation action.

## Data and chronology

Each language/category has 192 paired training cards, 48 development pairs, and 48 sealed test pairs. Training styles 0–9 support the character n-gram comparator only. Development uses style 10, which was V28's unopened test style but is generated here with a new public seed. Test uses newly authored style 11 with an independently generated hidden 192-bit seed. No V28 or V27 private test seed is used. All cards are synthetic annotation metadata, not child utterances, case narratives, realistic disclosures, or response drafts. Labels come from the typed D1/S1 policy oracle. Romanian and Ukrainian wording is separately drafted but still awaits independent fluent review.

Run `python -m training.generate_v29_hybrid prepare` once, then commit and push this protocol, code, wording, tests, training/development rows, source hashes, and the test seed hash before cutoff selection. Unit tests use a dummy seed. The V29 test file must not exist at registration. Use V29 development only to fix cutoffs and report full raw scores and evidence. The predeclared gate to open its test is **at least 168/192 complete pairs and at least 40/48 in each cell**. Preserve a failed gate and leave the test unopened. If the gate passes, commit and push the selected cutoffs and complete development record before reveal. Score the unchanged hybrid once after reveal.

The test target remains **44/48 complete pairs and 44/48 correct positives and negatives in every cell**, with at least 9/12 complete D1 pairs per changed factor, at least 20/24 complete S1 pairs per changed factor, and finite scores throughout. Report pair and card metrics, false reviews, recall, specificity, precision, Wilson intervals, latent-fact accuracy, and fixed V25/V26 model and character n-gram comparators. A pass is evidence only for same-origin synthetic metadata transfer to style 11. It does not establish fluent language correctness, independent label agreement, or safe deployment for real children. The release gates in `docs/RELEASE_GATES.md` remain closed until independently reviewed.
