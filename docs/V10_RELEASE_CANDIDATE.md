# RoGuard v10 qualification and release evidence

RoGuard 10.0.0 provides an opt-in local joint-feature reviewer for Romanian and Ukrainian. It reads the pinned v9 Gemma 4 final-token and focal-message features, applies two checksum-bound RBF heads per language, and combines their margins with the v9 decision. The v9 model and threshold remain available through the default interface. The other five structured fields use v9 logits with explicit v10 scope-cue corrections; they have no newly trained heads or standalone accuracy claim.

## Install and run

The wheel includes the four RBF heads. Fetch the separate pinned v9 base model on Apple Silicon with Python 3.12 or later:

```sh
pip install 'roguard-child-safety[gemma4-v10]==10.0.0'
roguard-fetch fetch
roguard-review --v10 --language ro --text 'Pot cere ajutor când sunt speriat.'
roguard-review --v10 --language uk --text 'Можу попросити допомоги, коли мені страшно.'
```

The Python API is `Gemma4RBFV10Model.from_installed_v9(language='ro')` from `roguard.gemma4_rbf_v10`. Its output is an uncalibrated review proposal. It does not establish that an event occurred or authorize a notification or other external action.

## Measured engineering gate

The exact locally built and freshly installed v13e wheel has SHA-256 `269e756dfccd731ebc76dec867aad9b802f601fa5cb6b3363ba3bc583a73920b`. Its packaged model manifest has SHA-256 `7f6ac23d90d2620224434b248f502df85497b89afcf16df5e40ca1969963997c`. On the 120-case policy-corrected authored panel it made zero binary and concern errors, while v9 made six errors in each output per language. On 40 newly authored, separately frozen challenge O cases it made zero binary and concern errors, while v9 made five binary and four concern errors across both languages. The [per-case sources and results](../research/v10-evidence/fresh-self-challenges-20261110/README.md) retain the exact source and wheel hashes.

The original 120-case panel had 20 hypothetical personal concerns mislabeled negative. The separate [correction manifest](../research/v10-evidence/fresh-self-challenges-20261110/gemma4-final-mixed-panel20b-policy-correction-20261110.json) records the affected case IDs and policy basis; both source versions remain in the evidence directory. The panel was not fitted to the v10 heads. Challenge O was frozen before challenge N results were inspected and was not used to fit or adjust v13e.

The four head hashes, training report hash and parent-bundle identity are in `src/roguard/v10_models/manifest.json`. The [head fit report](../research/v10-evidence/fresh-self-challenges-20261110/gemma4-joint-feature-full-v13-probe-20261110.json) records 18,331 Romanian and 21,345 Ukrainian fit rows per head, with zero fit errors. Its raw heads have diagnostic hold errors; the final guarded runtime cleared that exposed set. Fourteen preceding authored challenges A–N had first-pass errors and remain public. Their later zero-error replay measures development repair, not fresh generalization.

## Interpretation and open validation

The finite zero-error comparisons establish a reproducible improvement on authored synthetic inputs. They do not prove perfect real-world accuracy. The most recent fresh challenge O is 40 cases; all other test and training populations were also project-authored. No independently labeled case-level human evaluation record was provided. The user reported informal reviewer approval and requested publication on the strength of the transparent engineering result. This release makes no independent-human-accuracy claim.

Two fluent reviewers per language should still annotate a new blinded sample independently, with separate adjudication and a recorded v9/v10 comparison. See the [independent review protocol](V10_INDEPENDENT_REVIEW_PROTOCOL.md). The v9 default path remains available, and neither path grants external-action permission.
