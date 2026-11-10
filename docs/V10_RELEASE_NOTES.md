# RoGuard v10 candidate: what changed and what the evidence says

**Status:** draft release candidate. Version 10.0.0 is built as a wheel and source archive, but is not a stable PyPI release. The v9.0.0 package remains the stable public version. An informal reviewer approval endorsed transparent reporting; no independently labeled case set or measured reviewer result has been supplied for the release gate.

## The model

V10 reuses the exact pinned v9 Gemma 4 backbone and v9 review head. It extracts two frozen features: the final classifier token and the focal minor message's attention-state mean. Four portable radial-basis-function (RBF) heads are included in the wheel: binary review and concern for Romanian and for Ukrainian. A guarded decision rule uses their margins to correct the v9 binary proposal; it retains the parent's decision when the new evidence is insufficient. The existing v9 command stays the default; `roguard-review --v10` selects the new model. The other five structured fields still come from v9.

This is a new feature extraction and head-training pipeline, not a retraining of Gemma 4. The published v9 runtime file remains byte-for-byte identical to the v9.0.0 tag. The v10 wheel pins all four model-head hashes and the v9 base-bundle identity. Inference is local after the base model is fetched, on the validated Apple Silicon and Python 3.12+ runtime.

## Demonstrated wins

| Comparison | V9 | V10 | Scope |
| --- | ---: | ---: | --- |
| Binary errors, 60 authored Romanian cases | 6 | 0 | Same policy-corrected panel |
| Concern errors, 60 authored Romanian cases | 6 | 0 | Same panel |
| Binary errors, 60 authored Ukrainian cases | 6 | 0 | Same panel |
| Concern errors, 60 authored Ukrainian cases | 6 | 0 | Same panel |

The frozen candidate also had zero binary and concern errors on 39,010 exposed training and diagnostic cases. A fresh installation of the rebuilt wheel replayed all 120 policy-corrected panel cases with zero v10 errors in those two outputs. All six v9 error cases per language were safe statements that v9 falsely flagged in both outputs; v9 had no missed positive cases on this panel. The measured gain here is fewer false alerts, with no demonstrated recall gain. The panel was not used to fit the four heads. The original panel and first evaluation are retained: 20 hypothetical personal concerns were originally mislabeled negative, and a separate manifest records the correction against the preexisting policy.

## Losses, rejected approaches and limits

- Earlier candidate heads failed fresh generated panels. Query-only features did not generalize well enough; the joint final-token and focal-message feature was selected after these failures.
- A shortcut SVC fit on a smaller active set introduced a harmful-case regression and was rejected. The final four heads use the full fit.
- A Gemma 4 prompt-based pilot missed 16 of 40 mixed-scope cases and was rejected.
- The 120-case comparison is authored, synthetic, and finite. Its zero-error result does not prove perfect accuracy on authentic child language or unseen distributions. A fresh, independently labeled comparison remains missing.
- Five structured fields are inherited from v9 without a new v10 accuracy claim. The review margins are uncalibrated. The native model requires a separate, large v9 base download and the validated Apple Silicon runtime.
- The opt-in v10 path has not replaced the default v9 review path. No automatic external action or notification is authorized by either output.

The [candidate evidence](V10_RELEASE_CANDIDATE.md), [independent review protocol](V10_INDEPENDENT_REVIEW_PROTOCOL.md), [earlier committee record](V10_COMMITTEE.md), and [research history](V10_RESEARCH.md) retain the rationale and failures. Promotion requires the exact installed wheel to score with zero errors and at least one v9 error corrected on a new, frozen, independently annotated bilingual set. A tie does not demonstrate improvement. An informal approval without case-level results cannot establish that measured gate.
